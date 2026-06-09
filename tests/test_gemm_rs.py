"""
Validation test for BF16 GEMM + Reduce-Scatter (Pull-based, single-kernel fusion).
Compares bf16_gemm_rs_nt kernel with bf16_gemm + FP32 manual reduce-scatter.
Usage: python tests/test_gemm_rs.py [num_gpus]  (default: 2)
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run_test(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    # ── Config ──
    tokens_per_rank = 256
    k_dim = 1024
    n_dim = 512
    max_tokens_per_rank = tokens_per_rank
    total_m = tokens_per_rank * num_ranks

    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"BF16 GEMM-RS Test (Pull-based): {num_ranks} GPUs")
        print(f"  M_per_rank={tokens_per_rank}, K={k_dim}, N={n_dim}")
        print(f"{'='*60}\n")

    # ── Create test data ──
    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(a, src=0)

    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')

    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # ── Reference: bf16_gemm + FP32 manual reduce_scatter ──
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
    ref = ref_fp32.bfloat16()
    del all_d_fulls, d_full, ref_fp32
    dist.barrier()

    # ── Create symmetric buffer ──
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
    )

    # ── Warm-up (JIT compilation) ──
    y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    if rank_idx == 0:
        print(">>> Phase 1: Warm-up (JIT compilation)...")
    deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # ── Consistency check ──
    if rank_idx == 0:
        print(">>> Phase 2: Second run for consistency check...")
    y2 = torch.zeros_like(y)
    deep_gemm.bf16_gemm_rs_nt(y2, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    warmup_diff = (y.float() - y2.float()).abs().max().item()
    if rank_idx == 0:
        print(f"  Consistency check: max_diff={warmup_diff:.6f}")
        if warmup_diff < 0.01:
            print(f"  ✅ Kernel produces consistent results across runs")
        else:
            print(f"  ❌ Inconsistent results!")

    # ── Compare with reference ──
    if rank_idx == 0:
        print(">>> Phase 3: Comparing with reference (bf16_gemm + reduce_scatter)...")

    diff = (y.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Max abs diff:  {max_diff:.6f}")
        print(f"  Mean abs diff: {mean_diff:.6f}")
        if max_diff < 1.0:
            print(f"  ✅ PASS — BF16 GEMM-RS matches reference!")
        elif max_diff < 5.0:
            print(f"  ⚠️  Close but check numerical precision")
        else:
            print(f"  ❌ FAIL — Large difference")
            print(f"  y[0,0:4] = {y[0, 0:4].tolist()}")
            print(f"  ref[0,0:4] = {ref[0, 0:4].tolist()}")

    print(f"    [Rank {rank_idx}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    sym_buffer.destroy()
    dist.barrier()

    if rank_idx == 0:
        print(f"{'='*60}")
        print("Test complete.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Launching GEMM-RS test with {num_gpus} GPUs...")
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
