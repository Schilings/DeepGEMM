"""
GEMM-RS V3 Dual-Kernel Benchmark: V3 vs Separate (GEMM + NCCL RS)

Only benchmarks V3 dual-kernel and Separate (NCCL) for stability.
Skips V1 single-kernel which has NVLink barrier timeout issues in continuous benchmark.

Usage:
    python benchmarks/bench_gemm_rs_v3.py [num_gpus] [num_iters]
    python benchmarks/bench_gemm_rs_v3.py 8 20
    python benchmarks/bench_gemm_rs_v3.py 2 20
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


# --- Test shapes: Focus on target scenarios ---
SHAPES_TARGET = [
    # -- Medium (typical training batch) --
    (1024, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (2048, 7168, 2048),

    # -- Large (long context) --
    (4096, 7168, 2048),
    (4096, 2048, 7168),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),

    # -- Very Large --
    (8192, 7168, 2048),
    (8192, 2048, 7168),
    (8192, 4096, 4096),
    (8192, 7168, 4096),

    # -- Extreme --
    (16384, 7168, 2048),
    (16384, 2048, 7168),
    (16384, 4096, 4096),
    (16384, 7168, 4096),

    # -- Stress test --
    (8192, 7168, 7168),
    (16384, 7168, 7168),
    (20480, 7168, 2048),
]


def flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda").zero_()


def compute_tflops(m, n, k, time_ms):
    return 2.0 * m * n * k / (time_ms * 1e-3) / 1e12


def bench_fn(fn, num_warmup=5, num_iters=20, barrier_group=None):
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

    device = f"cuda:{local_rank}"

    if rank_idx == 0:
        print(f"\n{'='*100}")
        print(f"  GEMM-RS V3 Dual-Kernel Benchmark: {num_ranks} GPUs, {num_iters} iters")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  V3=Dual-kernel(256T compute + 256T reduce), stream-level overlap")
        print(f"{'='*100}")
        print()
        print(f"  {'Shape':<22} | {'Separate':>10} {'V3 Dual':>10} | "
              f"{'Sep TFLOPS':>10} {'V3 TFLOPS':>10} | {'V3 Speedup':>10}")
        print(f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} | "
              f"{'':>10} {'':>10} | {'vs Sep':>10}")
        print(f"  {'-'*22}-+-{'-'*10}-{'-'*10}-+-"
              f"{'-'*10}-{'-'*10}-+-{'-'*10}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES_TARGET:
        total_m = tokens_per_rank * num_ranks
        max_tokens_per_rank = tokens_per_rank

        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_v3 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_v3)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
                group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
            )
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5}  | SKIP (buffer alloc failed)")
            dist.barrier()
            continue

        dist.barrier()

        # -- Benchmark: Separate (GEMM + NCCL RS) --
        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        try:
            time_separate_ms = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5}  | SKIP (separate failed: {e})")
            sym_buffer.destroy()
            dist.barrier()
            continue

        # -- Benchmark: V3 Dual-kernel overlap --
        def run_v3():
            deep_gemm.bf16_gemm_rs_nt_v3(y_v3, a, b, sym_buffer, tokens_per_rank, compiled_dims="nk")

        try:
            time_v3_ms = bench_fn(run_v3, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            time_v3_ms = float("inf")
            if rank_idx == 0:
                print(f"  V3 failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        # Compute metrics
        time_separate_us = time_separate_ms * 1000
        time_v3_us = time_v3_ms * 1000 if time_v3_ms != float("inf") else float("inf")

        speedup_v3 = time_separate_us / time_v3_us if time_v3_us > 0 and time_v3_us != float("inf") else 0
        tflops_separate = compute_tflops(total_m, n_dim, k_dim, time_separate_ms)
        tflops_v3 = compute_tflops(total_m, n_dim, k_dim, time_v3_ms) if time_v3_ms != float("inf") else 0

        results.append({
            "tokens_per_rank": tokens_per_rank,
            "n_dim": n_dim,
            "k_dim": k_dim,
            "time_separate_us": time_separate_us,
            "time_v3_us": time_v3_us,
            "speedup_v3": speedup_v3,
            "tflops_separate": tflops_separate,
            "tflops_v3": tflops_v3,
        })

        if rank_idx == 0:
            shape_str = f"{tokens_per_rank}x{n_dim}x{k_dim}"
            v3_str = f"**{speedup_v3:.2f}x**" if speedup_v3 >= 1.0 else f"{speedup_v3:.2f}x"
            time_v3_print = f"{time_v3_us:>8.1f}" if time_v3_us != float("inf") else "     FAIL"
            tflops_v3_print = f"{tflops_v3:>8.1f}T" if tflops_v3 > 0 else "     FAIL"

            print(f"  {shape_str:<22} | {time_separate_us:>8.1f}u {time_v3_print}u | "
                  f"{tflops_separate:>8.1f}T {tflops_v3_print} | "
                  f"{v3_str:>10}")

        sym_buffer.destroy()
        dist.barrier()

    # -- Summary --
    if rank_idx == 0 and results:
        print(f"\n{'='*100}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'='*100}")

        v3_speedups = [r["speedup_v3"] for r in results if r["speedup_v3"] > 0]

        if v3_speedups:
            geo_v3 = math.exp(sum(math.log(s) for s in v3_speedups) / len(v3_speedups))
            print(f"\n  V3 Dual-Kernel (256T compute + 256T reduce) vs Separate GEMM+NCCL RS:")
            print(f"    Geometric Mean Speedup: {geo_v3:.3f}x")
            print(f"    Best:  {max(v3_speedups):.2f}x")
            print(f"    Worst: {min(v3_speedups):.2f}x")
            print(f"    Shapes V3 wins (>1.0x): {sum(1 for s in v3_speedups if s > 1.0)}/{len(v3_speedups)}")

            # V1 comparison reference from PROGRESS.md
            print(f"\n  Historical Reference: V1 single-kernel (384T) geo_mean was 1.040x on 8 GPU")
            v3_vs_v1 = geo_v3 / 1.040 if 1.040 > 0 else 0
            print(f"    V3 vs V1: {v3_vs_v1:.3f}x ({'better' if v3_vs_v1 > 1.0 else 'worse'})")

        # By scenario
        print(f"\n  By Scenario:")
        for label, pred in [
            ("N=7168 (large hidden)", lambda r: r["n_dim"] == 7168),
            ("K=7168 (large input)", lambda r: r["k_dim"] == 7168),
            ("K=4096 (medium input)", lambda r: r["k_dim"] == 4096),
            ("M/rank>=2048 (long ctx)", lambda r: r["tokens_per_rank"] >= 2048),
            ("M/rank>=8192 (very long)", lambda r: r["tokens_per_rank"] >= 8192),
        ]:
            subset = [r for r in results if pred(r)]
            if subset:
                sv3 = [r["speedup_v3"] for r in subset if r["speedup_v3"] > 0]
                if sv3:
                    g3 = math.exp(sum(math.log(s) for s in sv3) / len(sv3))
                    wins = sum(1 for s in sv3 if s > 1.0)
                    print(f"    {label:<28} V3={g3:.3f}x  ({wins}/{len(sv3)} win)")

        # TFLOPS
        print(f"\n  Average TFLOPS:")
        avg_sep = sum(r["tflops_separate"] for r in results) / len(results)
        avg_v3 = sum(r["tflops_v3"] for r in results if r["tflops_v3"] > 0) / max(1, sum(1 for r in results if r["tflops_v3"] > 0))
        print(f"    Separate: {avg_sep:.1f} TFLOPS")
        print(f"    V3 Dual:  {avg_v3:.1f} TFLOPS")

        # Top shapes
        print(f"\n  Top 5 Shapes (highest speedup):")
        sorted_v3 = sorted(results, key=lambda r: r["speedup_v3"], reverse=True)
        for i, r in enumerate(sorted_v3[:5]):
            shape_str = f"{r['tokens_per_rank']}x{r['n_dim']}x{r['k_dim']}"
            print(f"    {i+1}. {shape_str:<18} V3={r['speedup_v3']:.2f}x  "
                  f"(sep={r['time_separate_us']:.0f}us, v3={r['time_v3_us']:.0f}us)")

        print(f"\n{'='*100}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    print(f"Launching GEMM-RS V3 benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
