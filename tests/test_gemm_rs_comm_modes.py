"""
Test BF16 GEMM-RS with different communication and reduce precision modes.

Modes tested:
  1. comm_dtype=bf16, reduce_in_fp32=True  (default: bandwidth-efficient, FP32 reduce)
  2. comm_dtype=bf16, reduce_in_fp32=False (NCCL-like: everything in BF16)
  3. comm_dtype=fp32, reduce_in_fp32=True  (full precision: FP32 comm + FP32 reduce)

Usage: python tests/test_gemm_rs_comm_modes.py [num_gpus]  (default: 2)
"""

import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run_test(local_rank, num_local_ranks):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    tokens_per_rank = 256
    k_dim = 1024
    n_dim = 512
    total_m = tokens_per_rank * num_ranks

    # Generate inputs (same across all ranks for deterministic reference)
    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(a, src=0)
    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(b, src=0)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Compute FP32 reference: all_gather + FP32 sum
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    torch.cuda.synchronize(local_rank)
    all_d_fulls = [torch.empty_like(d_full) for _ in range(num_ranks)]
    dist.all_gather(all_d_fulls, d_full)
    torch.cuda.synchronize(local_rank)

    start_row = rank_idx * tokens_per_rank
    end_row = start_row + tokens_per_rank
    ref_fp32 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    for r in range(num_ranks):
        ref_fp32 += all_d_fulls[r][start_row:end_row, :].float()
    ref_bf16 = ref_fp32.bfloat16()
    del all_d_fulls, d_full
    dist.barrier()

    # ═══════════════════════════════════════════════════════
    # Mode 1: comm_dtype=bf16, reduce_in_fp32=True (DEFAULT)
    # ═══════════════════════════════════════════════════════
    sym_buf_1 = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16, comm_dtype=torch.bfloat16)
    y1 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_rs_nt(y1, a, b, sym_buf_1, tokens_per_rank, reduce_in_fp32=True)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    diff1 = (y1.float() - ref_fp32).abs().max().item()
    sym_buf_1.destroy()
    dist.barrier()

    # ═══════════════════════════════════════════════════════
    # Mode 2: comm_dtype=bf16, reduce_in_fp32=False (BF16 reduce)
    # ═══════════════════════════════════════════════════════
    sym_buf_2 = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16, comm_dtype=torch.bfloat16)
    y2 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_rs_nt(y2, a, b, sym_buf_2, tokens_per_rank, reduce_in_fp32=False)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # For BF16 reduce, compare against BF16 accumulation reference
    # (expect some diff vs FP32 ref, but should be small)
    diff2_vs_fp32 = (y2.float() - ref_fp32).abs().max().item()
    sym_buf_2.destroy()
    dist.barrier()

    # ═══════════════════════════════════════════════════════
    # Mode 3: comm_dtype=fp32, reduce_in_fp32=True (full FP32 precision)
    # ═══════════════════════════════════════════════════════
    sym_buf_3 = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16, comm_dtype=torch.float32)
    y3 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_rs_nt(y3, a, b, sym_buf_3, tokens_per_rank, reduce_in_fp32=True)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    diff3 = (y3.float() - ref_fp32).abs().max().item()
    sym_buf_3.destroy()
    dist.barrier()

    # ═══════════════════════════════════════════════════════
    # Report results
    # ═══════════════════════════════════════════════════════
    if rank_idx == 0:
        print(f"\n{'='*70}")
        print(f"  GEMM-RS Communication Mode Test: {num_ranks} GPUs")
        print(f"  M_per_rank={tokens_per_rank}, K={k_dim}, N={n_dim}")
        print(f"{'='*70}")
        print()
        print(f"  Mode 1: comm=BF16, reduce=FP32 (default)")
        print(f"    max_diff vs FP32 ref: {diff1:.6f}  {'✅' if diff1 == 0.0 else '⚠️'}")
        print()
        print(f"  Mode 2: comm=BF16, reduce=BF16 (NCCL-like)")
        print(f"    max_diff vs FP32 ref: {diff2_vs_fp32:.6f}  (expected: small non-zero for {num_ranks} ranks)")
        # For BF16 reduce, diff should be bounded by num_ranks * bf16_epsilon * max_value
        bf16_reduce_tolerance = num_ranks * 1.0  # heuristic: up to ~1.0 per rank
        mode2_ok = diff2_vs_fp32 <= bf16_reduce_tolerance
        print(f"    tolerance ({bf16_reduce_tolerance:.1f}): {'✅ within bounds' if mode2_ok else '❌ too large!'}")
        print()
        print(f"  Mode 3: comm=FP32, reduce=FP32 (full precision)")
        print(f"    max_diff vs FP32 ref: {diff3:.6f}  {'✅' if diff3 == 0.0 else '⚠️'}")
        print()

        all_pass = (diff1 == 0.0) and mode2_ok and (diff3 == 0.0)
        if all_pass:
            print(f"  ✅ ALL MODES PASS")
        else:
            print(f"  ❌ SOME MODES FAILED")
        print(f"{'='*70}")

    dist.barrier()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Launching comm mode test with {num_gpus} GPUs...")
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
