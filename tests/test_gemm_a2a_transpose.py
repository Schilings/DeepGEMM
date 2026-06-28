"""
Correctness test for BF16 pre-attn fused GEMM + All2All-transpose (Ulysses SP).

This is the DUAL of post-attn `a2a_transpose_gemm`:
  - pre-attn  (this op): GEMM(QKV/Q proj) FIRST, then head-wise All2All-transpose (scatter heads,
                         gather seq) → BSHD [bs, seq, local_nheads, head_dim] for FlashAttention.
  - post-attn          : All2All-transpose FIRST (gather heads, scatter seq), then Wo GEMM.

Dataflow (per rank r, seq is sharded, hidden K is full):
  x_r  : [bs, local_seq, K]   (this rank's seq shard)
  W    : [N, K]               (projection weights, N = nheads*head_dim, shared)
  D_r = x_r @ W.t()           : [bs, local_seq, N]
  After A2A-transpose, rank d owns out_d[b, s, n_local] for the FULL seq, where
    out_d[b, s, n_local] = D_{s//local_seq}[b, s%local_seq, d*local_n + n_local]
  i.e. out_d = D_full[:, :, d*local_n:(d+1)*local_n], with D_full the seq-concat of all D_r.

Usage:
    python tests/test_gemm_a2a_transpose.py [num_gpus]        # default: 2
    python tests/test_gemm_a2a_transpose.py [num_gpus] --all  # run extended shapes
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


# ─── Test shapes: (bs, local_seq, nheads, head_dim, K) ───
# N = nheads * head_dim. Constraints: local_seq % 128 == 0, nheads % num_ranks == 0,
# local_n = (nheads/num_ranks)*head_dim % 128 == 0 (head_dim=128 → local_nheads>=1 suffices).
SHAPES_BASIC = [
    (1, 1024, 32, 128, 4096),   # THD (bs=1), N=4096
    (1, 2048, 64, 128, 8192),   # THD, N=8192
    (2, 512,  32, 128, 4096),   # BSHD bs=2
    (1, 4096, 32, 128, 7168),   # THD, long ctx, K=7168
    (2, 1024, 64, 128, 8192),   # BSHD bs=2, N=8192
]

SHAPES_EXTENDED = [
    (1, 8192, 64, 128, 8192),
    (4, 512,  64, 128, 4096),
    (1, 4096, 64, 128, 7168),
    (2, 2048, 32, 128, 4096),
]


def compute_reference(a, b, bs, local_seq, n, num_ranks, rank_idx, local_rank):
    """Exact ground truth: DeepGEMM bf16_gemm_nt local proj → all_gather → seq-concat → head slice."""
    local_m = bs * local_seq
    d_local = torch.empty((local_m, n), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_local)
    torch.cuda.synchronize(local_rank)

    all_d = [torch.empty_like(d_local) for _ in range(num_ranks)]
    dist.all_gather(all_d, d_local)
    torch.cuda.synchronize(local_rank)

    # D_full[b, src*local_seq + s_local, :] = all_d[src][b*local_seq + s_local, :]
    # → [bs, seq, n]
    seq = local_seq * num_ranks
    d_full = torch.stack(
        [all_d[src].view(bs, local_seq, n) for src in range(num_ranks)], dim=1
    )  # [bs, num_ranks, local_seq, n]
    d_full = d_full.reshape(bs, seq, n)

    local_n = n // num_ranks
    ref = d_full[:, :, rank_idx * local_n:(rank_idx + 1) * local_n].contiguous()  # [bs, seq, local_n]
    del all_d, d_local, d_full
    return ref


def compute_torch_native(a, b, bs, local_seq, n, num_ranks, local_rank, group):
    """Real-world Ulysses pre-attn baseline: torch.matmul local proj + all_to_all_single."""
    local_n = n // num_ranks
    # Local projection (full GEMM on this rank's seq shard)
    d_local = torch.matmul(a, b.t())                       # [bs*local_seq, N]
    d_local = d_local.view(bs, local_seq, num_ranks, local_n)
    # Reorder so the dst-rank dim is the all_to_all split dim (dim 0).
    send = d_local.permute(2, 0, 1, 3).contiguous()        # [num_ranks, bs, local_seq, local_n]
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)        # recv[s_rank] = src s_rank's seq shard
    # recv[s_rank, b, s_local, n_local] → out[b, s_rank*local_seq + s_local, n_local]
    seq = local_seq * num_ranks
    out = recv.permute(1, 0, 2, 3).reshape(bs, seq, local_n).contiguous()
    torch.cuda.synchronize(local_rank)
    del d_local, send, recv
    return out


def run_single_shape_test(rank_idx, num_ranks, local_rank, group,
                          bs, local_seq, nheads, head_dim, k_dim):
    """Returns (passed, max_diff, rel_error, consistency_diff, max_diff_torch, rel_error_torch)."""
    n = nheads * head_dim
    seq = local_seq * num_ranks
    local_n = n // num_ranks
    local_m = bs * local_seq

    # Full input X [bs, seq, K] and weights W [N, K], identical across ranks (broadcast).
    x_full = torch.randn((bs, seq, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(x_full, src=0)
    b = torch.randn((n, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(b, src=0)

    # This rank's local seq shard, as [bs*local_seq, K]
    a = x_full[:, rank_idx * local_seq:(rank_idx + 1) * local_seq, :].reshape(local_m, k_dim).contiguous()

    torch.cuda.synchronize(local_rank)
    dist.barrier()

    ref = compute_reference(a, b, bs, local_seq, n, num_ranks, rank_idx, local_rank)
    dist.barrier()
    out_torch = compute_torch_native(a, b, bs, local_seq, n, num_ranks, local_rank, group)
    dist.barrier()

    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_a2a_transpose(
        group, bs, seq, n, out_dtype=torch.bfloat16)

    out = deep_gemm.bf16_gemm_a2a_transpose_nt(a, b, sym_buffer, local_seq, compiled_dims='nk')
    out = out.clone()  # snapshot before the second run overwrites the symm buffer
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    out2 = deep_gemm.bf16_gemm_a2a_transpose_nt(a, b, sym_buffer, local_seq, compiled_dims='nk')
    out2 = out2.clone()
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    consistency_diff = (out.float() - out2.float()).abs().max().item()
    max_diff = (out.float() - ref.float()).abs().max().item()
    mean_diff = (out.float() - ref.float()).abs().mean().item()
    ref_abs_mean = ref.float().abs().mean().item()
    rel_error = mean_diff / max(ref_abs_mean, 1e-8)

    max_diff_torch = (out.float() - out_torch.float()).abs().max().item()
    mean_diff_torch = (out.float() - out_torch.float()).abs().mean().item()
    torch_abs_mean = out_torch.float().abs().mean().item()
    rel_error_torch = mean_diff_torch / max(torch_abs_mean, 1e-8)
    del out_torch

    # A2A is a pure permutation (no reduce), so it should match the reference essentially exactly;
    # the only error is the bf16 GEMM itself (identical math in ref and fused). Tight threshold.
    passed = (rel_error < 0.01 and rel_error_torch < 0.01 and consistency_diff < 1e-3)

    sym_buffer.destroy()
    dist.barrier()
    return (passed, max_diff, rel_error, consistency_diff, max_diff_torch, rel_error_torch)


def run_test(local_rank: int, num_local_ranks: int, run_all: bool = False):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    shapes = SHAPES_BASIC + (SHAPES_EXTENDED if run_all else [])

    if rank_idx == 0:
        print(f"\n{'='*100}")
        print(f"  BF16 pre-attn GEMM + A2A-transpose Correctness Test: {num_ranks} GPUs")
        print(f"  Testing {len(shapes)} shapes {'(full suite)' if run_all else '(basic suite)'}")
        print(f"{'='*100}\n")
        print(f"{'Shape (bs,lseq,h,hd,K)':<26} | {'Max Diff':>9} {'Rel Err':>9} {'Consist':>9} | "
              f"{'vs Torch Max':>12} {'vs Torch Rel':>12} | {'Status'}")
        print(f"{'-'*26} | {'-'*9} {'-'*9} {'-'*9} | {'-'*12} {'-'*12} | {'-'*8}")

    all_passed = True
    results = []

    for bs, local_seq, nheads, head_dim, k_dim in shapes:
        if nheads % num_ranks != 0:
            if rank_idx == 0:
                print(f"  SKIP ({bs},{local_seq},{nheads},{head_dim},{k_dim}): nheads % num_ranks != 0")
            dist.barrier()
            continue
        try:
            (passed, max_diff, rel_error, consistency_diff,
             max_diff_torch, rel_error_torch) = run_single_shape_test(
                rank_idx, num_ranks, local_rank, group,
                bs, local_seq, nheads, head_dim, k_dim)
        except Exception as e:
            passed = False
            max_diff = rel_error = consistency_diff = float('nan')
            max_diff_torch = rel_error_torch = float('nan')
            if rank_idx == 0:
                print(f"  ERROR for ({bs},{local_seq},{nheads},{head_dim},{k_dim}): {e}")

        results.append((bs, local_seq, nheads, head_dim, k_dim, passed))
        if rank_idx == 0:
            status = "PASS" if passed else "FAIL"
            shape_str = f"{bs},{local_seq},{nheads},{head_dim},{k_dim}"
            print(f"{shape_str:<26} | {max_diff:>9.6f} {rel_error:>9.6f} {consistency_diff:>9.6f} | "
                  f"{max_diff_torch:>12.6f} {rel_error_torch:>12.6f} | {status}")
        if not passed:
            all_passed = False
        dist.barrier()

    if rank_idx == 0:
        num_passed = sum(1 for r in results if r[5])
        print(f"\n{'='*100}")
        print(f"  Summary: {num_passed}/{len(results)} shapes passed")
        print(f"  {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
        print(f"{'='*100}\n")

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
    print(f"Launching GEMM+A2A-transpose correctness test with {num_gpus} GPUs "
          f"(port={os.environ['MASTER_PORT']})...")
    mp.spawn(run_test, args=(num_gpus, run_all), nprocs=num_gpus, join=True)
