"""
Benchmark: GEMM-RS (Fused) vs Separate (GEMM + NCCL RS)

Compares end-to-end latency for:
  1. bf16_gemm_rs_nt (pull-based single kernel)
  2. bf16_gemm_nt + torch.distributed.reduce_scatter_tensor (separate)

Usage:
    python benchmarks/bench_gemm_rs.py [num_gpus] [num_iters]
    e.g.  python benchmarks/bench_gemm_rs.py 2 20
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple

import deep_gemm
from deep_gemm.utils.dist import init_dist


# ─── Test shapes ───
# (tokens_per_rank, N, K)
SHAPES = [
    # Small
    (256, 512, 1024),
    (256, 1024, 2048),
    # Medium
    (512, 2048, 4096),
    (1024, 2048, 4096),
    (2048, 2048, 4096),
    # Large
    (4096, 4096, 4096),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
]


def flush_l2():
    """Flush GPU L2 cache."""
    torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()


def bench_fn(fn, num_warmup=5, num_iters=20, barrier_group=None):
    """Benchmark a function, returns average time in ms."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    flush_l2()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters  # ms


def run_benchmark(local_rank: int, num_local_ranks: int, num_iters: int = 20):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    device = f'cuda:{local_rank}'

    if rank_idx == 0:
        print(f"\n{'='*80}")
        print(f"  GEMM-RS Benchmark: {num_ranks} GPUs, {num_iters} iterations")
        print(f"{'='*80}")
        print(f"{'Tokens/rank':<12} {'N':<8} {'K':<8} | {'Separate':>10} {'Fused':>10} | {'Speedup':>8}")
        print(f"{'-'*12} {'-'*8} {'-'*8} | {'-'*10} {'-'*10} | {'-'*8}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_ranks
        max_tokens_per_rank = tokens_per_rank

        # Create data
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_fused)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # Create symmetric buffer
        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )

        dist.barrier()

        # ── Benchmark: Separate (GEMM + NCCL RS) ──
        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        time_separate = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)

        # ── Benchmark: Fused (Pull-based) ──
        def run_fused():
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')

        time_fused = bench_fn(run_fused, num_iters=num_iters, barrier_group=group)

        # Compute speedup
        speedup = time_separate / time_fused if time_fused > 0 else float('inf')

        results.append((tokens_per_rank, n_dim, k_dim, time_separate, time_fused, speedup))

        if rank_idx == 0:
            print(f"{tokens_per_rank:<12} {n_dim:<8} {k_dim:<8} | "
                  f"{time_separate:>8.2f}ms {time_fused:>8.2f}ms | "
                  f"{speedup:>7.2f}x")

        sym_buffer.destroy()
        dist.barrier()

    # Summary
    if rank_idx == 0:
        print(f"\n{'-'*80}")
        geo_mean = 1.0
        for r in results:
            geo_mean *= r[5]
        geo_mean = geo_mean ** (1.0 / len(results))
        print(f"  Geo mean speedup (Fused / Separate): {geo_mean:.3f}x")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
