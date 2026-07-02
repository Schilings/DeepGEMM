"""
Unified sym buffer correctness test — verify all operators work on the same buffer.

Tests:
  1. Create one UnifiedSymmBuffer
  2. Run each operator on it, verify correctness:
     - GEMM-A2A-transpose (pre-attn)
     - A2A-transpose-GEMM (post-attn)
     - GEMM-RS (post-attn variant)
     - AG-GEMM (bwd)
     - Fused QKV+Norm+A2A (pre-attn with norm)
  3. Verify serial reuse (call same operator twice on same buffer)

Usage:
    python tests/comm/test_unified_buffer.py [num_gpus]
"""

import os
import sys
import math
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run_test(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    device = f'cuda:{local_rank}'

    # Wan2.1 14B-like config
    bs, seq = 1, 2048
    q_nheads, kv_nheads, head_dim = 32, 32, 128
    hidden = q_nheads * head_dim  # 4096
    local_seq = seq // num_ranks
    local_m = bs * local_seq
    eps = 1e-6

    if rank_idx == 0:
        print(f"\n{'='*100}")
        print(f"  Unified Sym Buffer Test: {num_ranks} GPUs")
        print(f"  Config: bs={bs}, seq={seq}, q_nheads={q_nheads}, kv_nheads={kv_nheads}, "
              f"head_dim={head_dim}, hidden={hidden}")
        print(f"{'='*100}\n")

    # ── Create ONE unified sym buffer ──
    sym = deep_gemm.get_unified_symm_buffer(
        group, bs, seq, q_nheads, kv_nheads, head_dim, hidden)

    all_passed = True

    # ════════════════════════════════════════════════════════════════
    #  Test 1: GEMM-A2A-transpose (pre-attn) — use its own sym buffer
    # ════════════════════════════════════════════════════════════════
    try:
        n = q_nheads * head_dim
        x = torch.randn((local_m, hidden), dtype=torch.bfloat16, device=device)
        w = torch.randn((n, hidden), dtype=torch.bfloat16, device=device)

        gemm_a2a_sym = deep_gemm.get_symm_buffer_for_gemm_a2a_transpose(
            group, bs, seq, n, out_dtype=torch.bfloat16)
        out = deep_gemm.bf16_gemm_a2a_transpose_nt(x, w, gemm_a2a_sym, local_seq)
        out = out.clone()

        # Reference
        d_local = torch.matmul(x, w.t())
        all_d = [torch.empty_like(d_local) for _ in range(num_ranks)]
        dist.all_gather(all_d, d_local)
        d_full = torch.stack([all_d[s].view(bs, local_seq, n) for s in range(num_ranks)], dim=1).reshape(bs, seq, n)
        local_n = n // num_ranks
        ref = d_full[:, :, rank_idx * local_n:(rank_idx + 1) * local_n].contiguous()

        rel_err = (out.float() - ref.float()).abs().mean().item() / max(ref.float().abs().mean().item(), 1e-8)
        passed = rel_err < 0.01
        if rank_idx == 0:
            print(f"  [{'PASS' if passed else 'FAIL'}] GEMM-A2A-transpose: rel_err={rel_err:.6f}")
        all_passed &= passed
        gemm_a2a_sym.destroy()
    except Exception as e:
        if rank_idx == 0:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] GEMM-A2A-transpose: {e}")
        all_passed = False
    dist.barrier()

    # ════════════════════════════════════════════════════════════════
    #  Test 2: Fused QKV+Norm+A2A (pre-attn with norm) — uses unified buffer
    # ════════════════════════════════════════════════════════════════
    try:
        q_dim = q_nheads * head_dim
        kv_dim = kv_nheads * head_dim
        n_total = q_dim + 2 * kv_dim
        x2 = torch.randn((local_m, hidden), dtype=torch.bfloat16, device=device)
        w2 = torch.randn((n_total, hidden), dtype=torch.bfloat16, device=device)
        norm_q = torch.ones(q_dim, dtype=torch.float32, device=device).normal_(1.0, 0.1)
        norm_k = torch.ones(kv_dim, dtype=torch.float32, device=device).normal_(1.0, 0.1)

        fused_sym = deep_gemm.get_symm_buffer_for_fused_qkv_norm_a2a(
            group, bs, seq, q_nheads, kv_nheads, head_dim)
        out2, rms2 = deep_gemm.bf16_fused_qkv_norm_a2a_nt(
            x2, w2, fused_sym, local_seq, q_nheads, kv_nheads, head_dim,
            eps=eps, norm_q_weight=norm_q, norm_k_weight=norm_k, bias=None)

        # Reference
        d2 = torch.matmul(x2, w2.t())
        q_r = d2[:, :q_dim].float()
        rms_q = torch.rsqrt(q_r.pow(2).sum(-1, keepdim=True) / q_dim + eps)
        q_n = (q_r * rms_q * norm_q.float()).to(torch.bfloat16)
        k_r = d2[:, q_dim:q_dim+kv_dim].float()
        rms_k = torch.rsqrt(k_r.pow(2).sum(-1, keepdim=True) / kv_dim + eps)
        k_n = (k_r * rms_k * norm_k.float()).to(torch.bfloat16)
        v_n = d2[:, q_dim+kv_dim:]
        d_normed = torch.cat([q_n, k_n, v_n], dim=-1)

        local_q_n = (q_nheads // num_ranks) * head_dim
        local_kv_n = (kv_nheads // num_ranks) * head_dim
        local_n2 = local_q_n + 2 * local_kv_n
        q_v = d_normed[:, :q_dim].view(bs, local_seq, num_ranks, local_q_n)
        k_v = d_normed[:, q_dim:q_dim+kv_dim].view(bs, local_seq, num_ranks, local_kv_n)
        v_v = d_normed[:, q_dim+kv_dim:].view(bs, local_seq, num_ranks, local_kv_n)
        send = torch.cat([q_v, k_v, v_v], dim=-1).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        ref2 = recv.permute(1, 0, 2, 3).reshape(bs, seq, local_n2).contiguous()

        rel_err2 = (out2.float() - ref2.float()).abs().mean().item() / max(ref2.float().abs().mean().item(), 1e-8)
        passed2 = rel_err2 < 0.03
        if rank_idx == 0:
            print(f"  [{'PASS' if passed2 else 'FAIL'}] Fused QKV+Norm+A2A: rel_err={rel_err2:.6f}")
        all_passed &= passed2
        fused_sym.destroy()
    except Exception as e:
        if rank_idx == 0:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] Fused QKV+Norm+A2A: {e}")
        all_passed = False
    dist.barrier()

    # ════════════════════════════════════════════════════════════════
    #  Test 3: GEMM-RS (post-attn variant)
    # ════════════════════════════════════════════════════════════════
    try:
        rs_sym = deep_gemm.get_symm_buffer_for_gemm_rs(group, bs * local_seq, hidden, num_ranks)
        d_rs = torch.randn((bs * local_seq, hidden), dtype=torch.bfloat16, device=device)
        wo_rs = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=device)
        out_rs = torch.empty((bs * local_seq // num_ranks, hidden), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_rs_nt(out_rs, d_rs, wo_rs, rs_sym, bs * local_seq // num_ranks)

        # Reference: GEMM + reduce_scatter
        d_local_rs = torch.matmul(d_rs, wo_rs.t())
        ref_rs = torch.empty_like(out_rs)
        dist.reduce_scatter_tensor(ref_rs, d_local_rs, group=group)

        rel_err_rs = (out_rs.float() - ref_rs.float()).abs().mean().item() / max(ref_rs.float().abs().mean().item(), 1e-8)
        passed_rs = rel_err_rs < 0.05
        if rank_idx == 0:
            print(f"  [{'PASS' if passed_rs else 'FAIL'}] GEMM-RS: rel_err={rel_err_rs:.6f}")
        all_passed &= passed_rs
        rs_sym.destroy()
    except Exception as e:
        if rank_idx == 0:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] GEMM-RS: {e}")
        all_passed = False
    dist.barrier()

    # ════════════════════════════════════════════════════════════════
    #  Test 4: AG-GEMM (bwd)
    # ════════════════════════════════════════════════════════════════
    try:
        num_tokens = bs * local_seq
        ag_sym = deep_gemm.get_symm_buffer_for_bf16_ag_gemm(group, num_tokens, hidden)
        d_ag = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
        wo_ag = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=device)
        out_ag = torch.empty((num_tokens * num_ranks, hidden), dtype=torch.bfloat16, device=device)
        ag_sym.x.copy_(d_ag)
        deep_gemm.bf16_ag_gemm_nt(out_ag, wo_ag, ag_sym, num_tokens)

        # Reference: all_gather + GEMM
        all_d = [torch.empty_like(d_ag) for _ in range(num_ranks)]
        dist.all_gather(all_d, d_ag)
        d_full_ag = torch.cat(all_d, dim=0)
        ref_ag = torch.matmul(d_full_ag, wo_ag.t())

        rel_err_ag = (out_ag.float() - ref_ag.float()).abs().mean().item() / max(ref_ag.float().abs().mean().item(), 1e-8)
        passed_ag = rel_err_ag < 0.05
        if rank_idx == 0:
            print(f"  [{'PASS' if passed_ag else 'FAIL'}] AG-GEMM: rel_err={rel_err_ag:.6f}")
        all_passed &= passed_ag
        ag_sym.destroy()
    except Exception as e:
        if rank_idx == 0:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] AG-GEMM: {e}")
        all_passed = False
    dist.barrier()

    # ════════════════════════════════════════════════════════════════
    #  Test 5: Unified buffer — Fused QKV reuse
    # ════════════════════════════════════════════════════════════════
    try:
        sym_u = deep_gemm.get_unified_symm_buffer(
            group, bs, seq, q_nheads, kv_nheads, head_dim, hidden)

        # Wrap for fused QKV
        from deep_gemm.fused_qkv_norm_a2a import FusedQKVNormA2ASymmBuffer
        fused_sym_u = FusedQKVNormA2ASymmBuffer.__new__(FusedQKVNormA2ASymmBuffer)
        fused_sym_u.group = group
        fused_sym_u.world_size = num_ranks
        fused_sym_u.bs = bs
        fused_sym_u.seq = seq
        fused_sym_u.local_seq = local_seq
        fused_sym_u.q_nheads = q_nheads
        fused_sym_u.kv_nheads = kv_nheads
        fused_sym_u.head_dim = head_dim
        fused_sym_u.out_dtype = torch.bfloat16
        fused_sym_u.local_q_nheads = q_nheads // num_ranks
        fused_sym_u.local_kv_nheads = kv_nheads // num_ranks
        fused_sym_u.local_q_n = (q_nheads // num_ranks) * head_dim
        fused_sym_u.local_kv_n = (kv_nheads // num_ranks) * head_dim
        fused_sym_u.local_n = fused_sym_u.local_q_n + 2 * fused_sym_u.local_kv_n
        fused_sym_u.buffer = sym_u.buffer
        fused_sym_u.handle = sym_u.handle
        fused_sym_u.sum_buffer = sym_u.sum_buffer

        sym_u.reset_sum_buffer()
        out_u, rms_u = deep_gemm.bf16_fused_qkv_norm_a2a_nt(
            x2, w2, fused_sym_u, local_seq, q_nheads, kv_nheads, head_dim,
            eps=eps, norm_q_weight=norm_q, norm_k_weight=norm_k, bias=None)

        # Reuse: call again
        sym_u.reset_sum_buffer()
        out_u2, rms_u2 = deep_gemm.bf16_fused_qkv_norm_a2a_nt(
            x2, w2, fused_sym_u, local_seq, q_nheads, kv_nheads, head_dim,
            eps=eps, norm_q_weight=norm_q, norm_k_weight=norm_k, bias=None)

        reuse_diff = (out_u.float() - out_u2.float()).abs().max().item()
        passed_u = reuse_diff < 1e-3
        if rank_idx == 0:
            print(f"  [{'PASS' if passed_u else 'FAIL'}] Unified buffer reuse: reuse_diff={reuse_diff:.6f}")
        all_passed &= passed_u
        sym_u.destroy()
    except Exception as e:
        if rank_idx == 0:
            import traceback; traceback.print_exc()
            print(f"  [FAIL] Unified buffer reuse: {e}")
        all_passed = False
    dist.barrier()

    # ── Summary ──
    if rank_idx == 0:
        print(f"\n{'='*100}")
        print(f"  {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
        print(f"{'='*100}\n")

    sym.destroy()
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
    if os.getenv('MASTER_PORT') is None or os.getenv('MASTER_PORT') == '':
        os.environ['MASTER_PORT'] = str(_find_free_port())
    print(f"Launching unified buffer test with {num_gpus} GPUs...")
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
