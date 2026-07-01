"""
Correctness test for Fused QKV GEMM + RMSNorm + A2A-transpose (Ulysses SP pre-attn).

Tests:
  1. norm_enabled=True,  MHA (q_nheads == kv_nheads)
  2. norm_enabled=False, MHA (degenerates to GEMM + bias + A2A)
  3. norm_enabled=True,  GQA (q_nheads > kv_nheads)
  4. norm_enabled=False, GQA

Ground truth: all_gather(x) → global GEMM → split Q/K/V → RMSNorm(Q,K) → A2A scatter by head → BSHD
Torch baseline: local GEMM + bias + RMSNorm + dist.all_to_all_single

Usage:
    python tests/comm/test_fused_qkv_norm_a2a.py [num_gpus]        # default: 2
    python tests/comm/test_fused_qkv_norm_a2a.py [num_gpus] --all  # extended shapes
"""

import os
import sys
import math
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deep_gemm.utils.dist import init_dist


# ─── Test shapes: (bs, local_seq, q_nheads, kv_nheads, head_dim, K) ───
SHAPES_MHA = [
    # (bs, local_seq, q_nheads, kv_nheads, head_dim, K) — MHA: q==kv
    (1, 1024, 32, 32, 128, 4096),    # THD, MHA, N=3*4096
    (1, 2048, 64, 64, 128, 8192),    # THD, MHA, N=3*8192
    (2, 512,  32, 32, 128, 4096),    # BSHD bs=2, MHA
    (1, 4096, 32, 32, 128, 5120),    # Wan2.1-like: dim=5120, nheads=40 (here 32), K=5120
]

SHAPES_GQA = [
    # GQA: q > kv
    (1, 1024, 40, 8,  128, 5120),    # Wan2.1 14B: q=40, kv=8, dim=5120
    (2, 512,  32, 8,  128, 4096),    # GQA bs=2
    (1, 2048, 64, 8,  128, 8192),    # GQA, kv=8
]

SHAPES_EXTENDED = [
    (1, 4096, 64, 64, 128, 8192),
    (4, 512,  64, 8,  128, 4096),
    (2, 2048, 40, 8,  128, 5120),    # Wan2.1 14B BSHD
]


def rmsnorm_ref(x, weight, eps):
    """RMSNorm in fp32 (matching WanRMSNorm). x: [..., dim] bf16 → bf16."""
    xf = x.float()
    norm = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * norm).to(x.dtype) * weight.to(x.dtype)


def compute_ground_truth(x_full, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                         num_ranks, rank_idx, local_rank,
                         norm_q_weight, norm_k_weight, eps, bias):
    """Global reference: all_gather → global GEMM → split → norm → scatter → BSHD."""
    seq = local_seq * num_ranks
    local_m = bs * local_seq

    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim  # [Q | K | V]

    # Global GEMM (single process, full seq)
    x_global = x_full.reshape(bs * seq, k_dim)
    d_global = torch.matmul(x_global, b.t())  # [bs*seq, N_total]
    if bias is not None:
        d_global = d_global + bias
    d_global = d_global.view(bs, seq, n_total)

    # Split Q/K/V
    q = d_global[:, :, :q_dim]                    # [bs, seq, q_dim]
    k = d_global[:, :, q_dim:q_dim + kv_dim]      # [bs, seq, kv_dim]
    v = d_global[:, :, q_dim + kv_dim:]            # [bs, seq, kv_dim]

    # RMSNorm on Q and K (optional)
    if norm_q_weight is not None:
        q = rmsnorm_ref(q, norm_q_weight, eps)
    if norm_k_weight is not None:
        k = rmsnorm_ref(k, norm_k_weight, eps)

    # Reassemble [Q | K | V]
    d_normed = torch.cat([q, k, v], dim=-1)  # [bs, seq, n_total]

    # Scatter by head groups (each rank takes its head group of Q, K, V)
    local_q_nheads = q_nheads // num_ranks
    local_kv_nheads = kv_nheads // num_ranks
    local_q_n = local_q_nheads * head_dim
    local_kv_n = local_kv_nheads * head_dim
    local_n = local_q_n + 2 * local_kv_n

    # Reshape to extract per-rank head groups
    q_view = d_normed[:, :, :q_dim].view(bs, seq, num_ranks, local_q_n)
    k_view = d_normed[:, :, q_dim:q_dim + kv_dim].view(bs, seq, num_ranks, local_kv_n)
    v_view = d_normed[:, :, q_dim + kv_dim:].view(bs, seq, num_ranks, local_kv_n)

    # rank_idx's head group
    out = torch.cat([
        q_view[:, :, rank_idx, :],   # [bs, seq, local_q_n]
        k_view[:, :, rank_idx, :],   # [bs, seq, local_kv_n]
        v_view[:, :, rank_idx, :],   # [bs, seq, local_kv_n]
    ], dim=-1).contiguous()          # [bs, seq, local_n]

    return out


def compute_torch_baseline(a, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                           num_ranks, local_rank, group,
                           norm_q_weight, norm_k_weight, eps, bias):
    """Ulysses pre-attn baseline: local GEMM + norm + all_to_all."""
    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim

    local_q_nheads = q_nheads // num_ranks
    local_kv_nheads = kv_nheads // num_ranks
    local_q_n = local_q_nheads * head_dim
    local_kv_n = local_kv_nheads * head_dim
    local_n = local_q_n + 2 * local_kv_n

    # Local GEMM
    d_local = torch.matmul(a, b.t())  # [bs*local_seq, N_total]
    if bias is not None:
        d_local = d_local + bias
    d_local = d_local.view(bs, local_seq, n_total)

    # Split Q/K/V (local seq, full heads)
    q = d_local[:, :, :q_dim]                    # [bs, local_seq, q_dim]
    k = d_local[:, :, q_dim:q_dim + kv_dim]      # [bs, local_seq, kv_dim]
    v = d_local[:, :, q_dim + kv_dim:]            # [bs, local_seq, kv_dim]

    # RMSNorm on full dim (before A2A scatter)
    if norm_q_weight is not None:
        q = rmsnorm_ref(q, norm_q_weight, eps)
    if norm_k_weight is not None:
        k = rmsnorm_ref(k, norm_k_weight, eps)

    # Reshape for A2A: [num_ranks, bs, local_seq, local_n_per_segment]
    # Q: split q_nheads heads into num_ranks groups
    q_send = q.view(bs, local_seq, num_ranks, local_q_n).permute(2, 0, 1, 3).contiguous()
    k_send = k.view(bs, local_seq, num_ranks, local_kv_n).permute(2, 0, 1, 3).contiguous()
    v_send = v.view(bs, local_seq, num_ranks, local_kv_n).permute(2, 0, 1, 3).contiguous()

    # Pack Q/K/V into single tensor for combined A2A
    # send shape: [num_ranks, bs, local_seq, local_q_n + 2*local_kv_n]
    send = torch.cat([q_send, k_send, v_send], dim=-1)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)

    # recv[dst_rank] = src's seq shard for our head group
    # → out[bs, dst*local_seq + s_local, local_n]
    seq = local_seq * num_ranks
    out = recv.permute(1, 0, 2, 3).reshape(bs, seq, local_n).contiguous()

    torch.cuda.synchronize(local_rank)
    del d_local, q, k, v, q_send, k_send, v_send, send, recv
    return out


def run_single_shape_test(rank_idx, num_ranks, local_rank, group,
                          bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                          norm_enabled):
    """Returns (passed, max_diff, rel_error, max_diff_torch, rel_error_torch)."""
    seq = local_seq * num_ranks
    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim
    local_m = bs * local_seq
    eps = 1e-6

    # Full input X [bs, seq, K] and weights B [N_total, K], identical across ranks
    x_full = torch.randn((bs, seq, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(x_full, src=0)
    b = torch.randn((n_total, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(b, src=0)

    # Bias [N_total]
    bias = torch.randn((n_total,), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(bias, src=0)

    # Norm weights (fp32)
    if norm_enabled:
        norm_q_weight = torch.ones((q_dim,), dtype=torch.float32, device=f'cuda:{local_rank}')
        norm_q_weight.normal_(1.0, 0.1)
        norm_k_weight = torch.ones((kv_dim,), dtype=torch.float32, device=f'cuda:{local_rank}')
        norm_k_weight.normal_(1.0, 0.1)
        dist.broadcast(norm_q_weight, src=0)
        dist.broadcast(norm_k_weight, src=0)
    else:
        norm_q_weight = None
        norm_k_weight = None

    # This rank's local seq shard
    a = x_full[:, rank_idx * local_seq:(rank_idx + 1) * local_seq, :].reshape(local_m, k_dim).contiguous()

    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Ground truth
    ref = compute_ground_truth(
        x_full, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
        num_ranks, rank_idx, local_rank,
        norm_q_weight, norm_k_weight, eps, bias)
    dist.barrier()

    # Torch baseline
    out_torch = compute_torch_baseline(
        a, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
        num_ranks, local_rank, group,
        norm_q_weight, norm_k_weight, eps, bias)
    dist.barrier()

    # Compare torch baseline vs ground truth (should be very close, bf16 differences only)
    max_diff_ref_torch = (out_torch.float() - ref.float()).abs().max().item()

    # Fused API (single kernel: GEMM + x²sum + scatter + rms scatter)
    import deep_gemm
    sym_buffer = deep_gemm.get_symm_buffer_for_fused_qkv_norm_a2a(
        group, bs, seq, q_nheads, kv_nheads, head_dim)
    out_fused, rms_fused = deep_gemm.bf16_fused_qkv_norm_a2a_nt(
        a, b, sym_buffer, local_seq,
        q_nheads, kv_nheads, head_dim,
        eps=eps,
        norm_q_weight=norm_q_weight,
        norm_k_weight=norm_k_weight,
        bias=bias,
    )
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    if rank_idx == 0 and norm_enabled:
        print(f"  DEBUG rms[0,:3,0]={rms_fused[0,:3,0].tolist()} sum={rms_fused.abs().sum().item():.4f}")
        print(f"  DEBUG out[0,0,:3]={out_fused[0,0,:3].tolist()}")
        print(f"  DEBUG ref[0,0,:3]={ref[0,0,:3].tolist()}")

    max_diff = (out_fused.float() - ref.float()).abs().max().item()
    mean_diff = (out_fused.float() - ref.float()).abs().mean().item()
    ref_abs_mean = ref.float().abs().mean().item()
    rel_error = mean_diff / max(ref_abs_mean, 1e-8)

    max_diff_torch = (out_fused.float() - out_torch.float()).abs().max().item()
    mean_diff_torch = (out_fused.float() - out_torch.float()).abs().mean().item()
    torch_abs_mean = out_torch.float().abs().mean().item()
    rel_error_torch = mean_diff_torch / max(torch_abs_mean, 1e-8)

    sym_buffer.destroy()
    passed = (rel_error < 0.03 and rel_error_torch < 0.03 and max_diff_ref_torch < 1.0)

    return (passed, max_diff, rel_error, max_diff_ref_torch, max_diff_torch, rel_error_torch)


def run_test(local_rank: int, num_local_ranks: int, run_all: bool = False):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    test_cases = []

    # MHA shapes — norm enabled
    for shape in SHAPES_MHA:
        test_cases.append((*shape, True, 'MHA+norm'))
    # MHA shapes — norm disabled
    for shape in SHAPES_MHA:
        test_cases.append((*shape, False, 'MHA'))
    # GQA shapes — norm enabled
    for shape in SHAPES_GQA:
        test_cases.append((*shape, True, 'GQA+norm'))
    # GQA shapes — norm disabled
    for shape in SHAPES_GQA:
        test_cases.append((*shape, False, 'GQA'))

    if run_all:
        for shape in SHAPES_EXTENDED:
            test_cases.append((*shape, True, 'EXT+norm'))
            test_cases.append((*shape, False, 'EXT'))

    if rank_idx == 0:
        print(f"\n{'='*120}")
        print(f"  Fused QKV GEMM + RMSNorm(opt) + A2A-transpose Correctness Test: {num_ranks} GPUs")
        print(f"  Phase 1: Python reference verification (torch GEMM + norm + all_to_all)")
        print(f"  Testing {len(test_cases)} cases")
        print(f"{'='*120}\n")
        print(f"{'Shape (bs,lseq,qh,kvh,hd,K)':<32} | {'Mode':<10} | {'Max Diff':>9} {'Rel Err':>9} "
              f"{'Diff ref/torch':>14} | {'Status'}")
        print(f"{'-'*32} | {'-'*10} | {'-'*9} {'-'*9} {'-'*14} | {'-'*8}")

    all_passed = True
    results = []

    for bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim, norm_enabled, mode_label in test_cases:
        if q_nheads % num_ranks != 0 or kv_nheads % num_ranks != 0:
            if rank_idx == 0:
                print(f"  SKIP ({bs},{local_seq},{q_nheads},{kv_nheads},{head_dim},{k_dim}): "
                      f"head not divisible by {num_ranks}")
            dist.barrier()
            continue
        try:
            (passed, max_diff, rel_error, max_diff_ref_torch,
             _, _) = run_single_shape_test(
                rank_idx, num_ranks, local_rank, group,
                bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                norm_enabled)
        except Exception as e:
            passed = False
            max_diff = rel_error = max_diff_ref_torch = float('nan')
            if rank_idx == 0:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

        results.append((bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim, norm_enabled, passed))
        if rank_idx == 0:
            status = "PASS" if passed else "FAIL"
            shape_str = f"{bs},{local_seq},{q_nheads},{kv_nheads},{head_dim},{k_dim}"
            print(f"{shape_str:<32} | {mode_label:<10} | {max_diff:>9.6f} {rel_error:>9.6f} "
                  f"{max_diff_ref_torch:>14.6f} | {status}")
        if not passed:
            all_passed = False
        dist.barrier()

    if rank_idx == 0:
        num_passed = sum(1 for r in results if r[7])
        print(f"\n{'='*120}")
        print(f"  Summary: {num_passed}/{len(results)} cases passed")
        print(f"  {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
        print(f"{'='*120}\n")

    dist.barrier()
    dist.destroy_process_group()
    time.sleep(1)
    os._exit(0)


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_all = '--all' in sys.argv

    if os.getenv('MASTER_PORT') is None or os.getenv('MASTER_PORT') == '':
        os.environ['MASTER_PORT'] = str(_find_free_port())
    print(f"Launching Fused QKV+Norm+A2A correctness test with {num_gpus} GPUs "
          f"(port={os.environ['MASTER_PORT']})...")
    mp.spawn(run_test, args=(num_gpus, run_all), nprocs=num_gpus, join=True)
