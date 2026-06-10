"""
GEMM-RS Performance Benchmark: Fused vs Separate (GEMM + NCCL RS)

Comprehensive comparison targeting large model training scenarios:
  1. bf16_gemm_rs_nt (pull-based single kernel) — fused
  2. bf16_gemm_nt + torch.distributed.reduce_scatter_tensor — separate (NCCL)

Focus on large hidden dimensions (7168) and long-context training scenarios
where the fused approach should show maximum benefit.

Usage:
    python benchmarks/bench_gemm_rs.py [num_gpus] [num_iters]
    python benchmarks/bench_gemm_rs.py 8 30                    # 8 GPU, 30 iters
    python benchmarks/bench_gemm_rs.py 2 20 --profile          # with nsys markers

Output:
    - Per-shape latency (μs), TFLOPS, and speedup
    - Geometric mean speedup across all shapes
    - Comm/Compute ratio analysis
"""

import os
import sys
import time
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple

import deep_gemm
from deep_gemm.utils.dist import init_dist


# ─── Test shapes ───
# (tokens_per_rank, N, K) — covering all typical LLM training regimes
#
# Design rationale:
#   - Small M + Small N/K: MoE inference (low batch)
#   - Large M + Large N/K: Dense training (long context, large hidden)
#   - N=7168: DeepSeek-V3 style models
#   - K=7168: attention projection dimensions
#
SHAPES_STANDARD = [
    # ── Small batch (MoE inference / short sequence) ──
    (128, 512, 1024),
    (256, 512, 1024),
    (256, 1024, 2048),

    # ── Medium (typical training, moderate sequence) ──
    (512, 2048, 4096),
    (1024, 2048, 4096),
    (2048, 2048, 4096),

    # ── Large hidden dimension (DeepSeek-V3 style, N=7168) ──
    (256, 7168, 2048),
    (512, 7168, 2048),
    (1024, 7168, 2048),
    (2048, 7168, 2048),
    (4096, 7168, 2048),

    # ── Large K dimension (attention projections) ──
    (256, 2048, 7168),
    (512, 2048, 7168),
    (1024, 2048, 7168),
    (2048, 2048, 7168),
    (4096, 2048, 7168),

    # ── Square-ish large (balanced compute/comm) ──
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 4096, 4096),

    # ── Extreme large (stress test) ──
    (4096, 7168, 7168),
    (8192, 7168, 2048),
    (8192, 2048, 7168),
]


def flush_l2():
    """Flush GPU L2 cache by allocating and zeroing a large tensor."""
    torch.empty(int(256e6 // 4), dtype=torch.int32, device='cuda').zero_()


def compute_tflops(m, n, k, time_ms):
    """Compute TFLOPS for a GEMM of shape M×N×K."""
    flops = 2.0 * m * n * k
    return flops / (time_ms * 1e-3) / 1e12


def compute_comm_bytes(tokens_per_rank, n_dim, num_ranks, dtype_bytes=2):
    """
    Compute communication volume for reduce-scatter.
    Bandwidth-optimal: each rank receives (N-1)/N × data_size from peers.
    """
    data_per_rank = tokens_per_rank * n_dim * dtype_bytes
    return data_per_rank * (num_ranks - 1) / num_ranks


def bench_fn(fn, num_warmup=5, num_iters=20, barrier_group=None):
    """
    Benchmark a function using CUDA events.
    Returns average time in milliseconds.
    """
    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    # Flush L2 cache
    flush_l2()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    # Timed iterations
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
        print(f"\n{'═'*100}")
        print(f"  GEMM-RS Performance Benchmark: {num_ranks} GPUs, {num_iters} iterations per measurement")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'═'*100}")
        print()
        print(f"  {'Shape':<22} │ {'Separate':>10} {'Fused':>10} │ "
              f"{'Sep TFLOPS':>10} {'Fus TFLOPS':>10} │ {'Speedup':>8} │ {'Comp/Comm':>10}")
        print(f"  {'(M/rank×N×K)':<22} │ {'(μs)':>10} {'(μs)':>10} │ "
              f"{'':>10} {'':>10} │ {'':>8} │ {'Ratio':>10}")
        print(f"  {'─'*22}─┼─{'─'*10}─{'─'*10}─┼─"
              f"{'─'*10}─{'─'*10}─┼─{'─'*8}─┼─{'─'*10}")

    results = []
    shape_groups = {}  # Group results by category

    for tokens_per_rank, n_dim, k_dim in SHAPES_STANDARD:
        total_m = tokens_per_rank * num_ranks
        max_tokens_per_rank = tokens_per_rank

        # Create data
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_fused)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # Create symmetric buffer
        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
                group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
            )
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<5}  │ {'SKIP (buffer alloc failed)':>50} │")
            dist.barrier()
            continue

        dist.barrier()

        # ── Benchmark: Separate (GEMM + NCCL RS) ──
        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        try:
            time_separate_ms = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<5}  │ {'SKIP (separate failed)':>50} │")
            sym_buffer.destroy()
            dist.barrier()
            continue

        # ── Benchmark: Fused (Pull-based) ──
        def run_fused():
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')

        try:
            time_fused_ms = bench_fn(run_fused, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<5}  │ {'SKIP (fused failed)':>50} │")
            sym_buffer.destroy()
            dist.barrier()
            continue

        # Compute metrics
        time_separate_us = time_separate_ms * 1000
        time_fused_us = time_fused_ms * 1000
        speedup = time_separate_us / time_fused_us if time_fused_us > 0 else float('inf')

        tflops_separate = compute_tflops(total_m, n_dim, k_dim, time_separate_ms)
        tflops_fused = compute_tflops(total_m, n_dim, k_dim, time_fused_ms)

        # Compute/comm ratio (rough estimate)
        compute_flops = 2.0 * total_m * n_dim * k_dim
        comm_bytes = compute_comm_bytes(tokens_per_rank, n_dim, num_ranks)
        # Assume ~900 GB/s NVLink bandwidth per direction, ~1000 TFLOPS peak BF16
        comp_time_ideal = compute_flops / (1000e12)  # seconds
        comm_time_ideal = comm_bytes / (900e9)  # seconds
        comp_comm_ratio = comp_time_ideal / max(comm_time_ideal, 1e-12)

        results.append({
            'tokens_per_rank': tokens_per_rank,
            'n_dim': n_dim,
            'k_dim': k_dim,
            'time_separate_us': time_separate_us,
            'time_fused_us': time_fused_us,
            'speedup': speedup,
            'tflops_separate': tflops_separate,
            'tflops_fused': tflops_fused,
            'comp_comm_ratio': comp_comm_ratio,
        })

        if rank_idx == 0:
            shape_str = f"{tokens_per_rank}×{n_dim}×{k_dim}"
            speedup_str = f"{speedup:.2f}x"
            if speedup >= 1.0:
                speedup_str = f"**{speedup:.2f}x**"

            print(f"  {shape_str:<22} │ {time_separate_us:>8.1f}μs {time_fused_us:>8.1f}μs │ "
                  f"{tflops_separate:>8.1f}T {tflops_fused:>8.1f}T │ "
                  f"{speedup:>7.2f}x │ {comp_comm_ratio:>8.2f}x")

        sym_buffer.destroy()
        dist.barrier()

    # ── Summary ──
    if rank_idx == 0 and results:
        print(f"\n{'═'*100}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'═'*100}")

        # Overall stats
        speedups = [r['speedup'] for r in results]
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        max_speedup = max(speedups)
        min_speedup = min(speedups)
        median_speedup = sorted(speedups)[len(speedups) // 2]

        print(f"\n  Overall Statistics:")
        print(f"    Geometric Mean Speedup: {geo_mean:.3f}x")
        print(f"    Median Speedup:         {median_speedup:.3f}x")
        print(f"    Best Speedup:           {max_speedup:.3f}x")
        print(f"    Worst Speedup:          {min_speedup:.3f}x")
        print(f"    Shapes where Fused wins (>1.0x): "
              f"{sum(1 for s in speedups if s > 1.0)}/{len(speedups)}")

        # Categorize by scenario
        print(f"\n  By Scenario:")

        # Large hidden dim (N=7168)
        large_n = [r for r in results if r['n_dim'] == 7168]
        if large_n:
            geo = math.exp(sum(math.log(r['speedup']) for r in large_n) / len(large_n))
            print(f"    N=7168 (large hidden):    geo_mean={geo:.3f}x  "
                  f"(best={max(r['speedup'] for r in large_n):.2f}x)")

        # Large K dim (K=7168)
        large_k = [r for r in results if r['k_dim'] == 7168]
        if large_k:
            geo = math.exp(sum(math.log(r['speedup']) for r in large_k) / len(large_k))
            print(f"    K=7168 (large input):     geo_mean={geo:.3f}x  "
                  f"(best={max(r['speedup'] for r in large_k):.2f}x)")

        # Small M (MoE inference)
        small_m = [r for r in results if r['tokens_per_rank'] <= 256]
        if small_m:
            geo = math.exp(sum(math.log(r['speedup']) for r in small_m) / len(small_m))
            print(f"    M/rank≤256 (MoE infer):   geo_mean={geo:.3f}x  "
                  f"(best={max(r['speedup'] for r in small_m):.2f}x)")

        # Large M (long context training)
        large_m = [r for r in results if r['tokens_per_rank'] >= 2048]
        if large_m:
            geo = math.exp(sum(math.log(r['speedup']) for r in large_m) / len(large_m))
            print(f"    M/rank≥2048 (long ctx):   geo_mean={geo:.3f}x  "
                  f"(best={max(r['speedup'] for r in large_m):.2f}x)")

        # Compute-bound vs comm-bound
        print(f"\n  By Compute/Comm Ratio:")
        compute_heavy = [r for r in results if r['comp_comm_ratio'] > 5.0]
        comm_heavy = [r for r in results if r['comp_comm_ratio'] <= 2.0]
        balanced = [r for r in results if 2.0 < r['comp_comm_ratio'] <= 5.0]

        if compute_heavy:
            geo = math.exp(sum(math.log(r['speedup']) for r in compute_heavy) / len(compute_heavy))
            print(f"    Compute-heavy (ratio>5):  geo_mean={geo:.3f}x ({len(compute_heavy)} shapes)")
        if balanced:
            geo = math.exp(sum(math.log(r['speedup']) for r in balanced) / len(balanced))
            print(f"    Balanced (2<ratio≤5):     geo_mean={geo:.3f}x ({len(balanced)} shapes)")
        if comm_heavy:
            geo = math.exp(sum(math.log(r['speedup']) for r in comm_heavy) / len(comm_heavy))
            print(f"    Comm-heavy (ratio≤2):     geo_mean={geo:.3f}x ({len(comm_heavy)} shapes)")

        # Best candidates for fusion
        print(f"\n  Top 5 Shapes for Fusion (highest speedup):")
        sorted_results = sorted(results, key=lambda r: r['speedup'], reverse=True)
        for i, r in enumerate(sorted_results[:5]):
            shape_str = f"{r['tokens_per_rank']}×{r['n_dim']}×{r['k_dim']}"
            print(f"    {i+1}. {shape_str:<18} {r['speedup']:.2f}x  "
                  f"(sep={r['time_separate_us']:.0f}μs, fused={r['time_fused_us']:.0f}μs)")

        print(f"\n{'═'*100}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
