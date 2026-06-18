"""
GEMM-RS Performance Benchmark: V3 Dual-Kernel vs V1 Fused vs Separate (GEMM + NCCL RS)

Comprehensive comparison targeting large model training scenarios:
  1. bf16_gemm_nt + torch.distributed.reduce_scatter_tensor -- separate (NCCL)
  2. bf16_gemm_rs_nt (pull-based single kernel, 384T) -- v1 fused
  3. bf16_gemm_rs_nt_v3 (dual-kernel overlap, 256T compute + 256T reduce) -- v3 fused

Focus on large hidden dimensions (7168) and long-context training scenarios.

Usage:
    python benchmarks/bench_gemm_rs.py [num_gpus] [num_iters]
    python benchmarks/bench_gemm_rs.py 8 30                    # 8 GPU, 30 iters
    python benchmarks/bench_gemm_rs.py 2 20                    # 2 GPU, 20 iters

Output:
    - Per-shape latency (us), TFLOPS, and speedup
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


# --- Test shapes ---
SHAPES_STANDARD = [
    # -- Medium (typical training batch, 8 GPU -> total 8k-16k tokens) --
    (1024, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (2048, 7168, 2048),

    # -- Large (long context, 8 GPU -> total 32k tokens) --
    (4096, 7168, 2048),
    (4096, 2048, 7168),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),

    # -- Very Large (long context, 8 GPU -> total 64k tokens) --
    (8192, 7168, 2048),
    (8192, 2048, 7168),
    (8192, 4096, 4096),
    (8192, 7168, 4096),

    # -- Extreme (batch size 16k/rank, total 128k tokens on 8 GPU) --
    (16384, 7168, 2048),
    (16384, 2048, 7168),
    (16384, 4096, 4096),
    (16384, 7168, 4096),

    # -- Stress test --
    (8192, 7168, 7168),
    (16384, 7168, 7168),
    (20480, 7168, 2048),
]

single_shape = os.getenv("DG_BENCH_SINGLE_SHAPE", "").strip()
if single_shape:
    m_str, n_str, k_str = [x.strip() for x in single_shape.split(",")]
    SHAPES_TO_RUN = [(int(m_str), int(n_str), int(k_str))]
else:
    max_shapes = int(os.getenv("DG_BENCH_MAX_SHAPES", "0"))
    SHAPES_TO_RUN = SHAPES_STANDARD[:max_shapes] if max_shapes > 0 else SHAPES_STANDARD


def flush_l2():
    """Flush GPU L2 cache by allocating and zeroing a large tensor."""
    torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda").zero_()


def compute_tflops(m, n, k, time_ms):
    """Compute TFLOPS for a GEMM of shape MxNxK."""
    flops = 2.0 * m * n * k
    return flops / (time_ms * 1e-3) / 1e12


def compute_comm_bytes(tokens_per_rank, n_dim, num_ranks, dtype_bytes=2):
    data_per_rank = tokens_per_rank * n_dim * dtype_bytes
    return data_per_rank * (num_ranks - 1) / num_ranks


def bench_fn(fn, num_warmup=5, num_iters=20, barrier_group=None):
    sync_each_iter = bool(int(os.getenv("DG_BENCH_SYNC_EACH_ITER", "0")))

    # Warmup
    for _ in range(num_warmup):
        fn()
        if sync_each_iter:
            torch.cuda.synchronize()
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
        if sync_each_iter:
            torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters  # ms


def run_benchmark(local_rank: int, num_local_ranks: int, num_iters: int = 20):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    device = f"cuda:{local_rank}"

    if rank_idx == 0:
        print(f"\n{'='*120}")
        print(f"  GEMM-RS Performance Benchmark: {num_ranks} GPUs, {num_iters} iterations per measurement")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  V1=Pull-based single-kernel(384T), V3=Dual-kernel overlap(256T compute + 256T reduce)")
        print(f"{'='*120}")
        print()
        print(f"  {'Shape':<22} | {'Separate':>10} {'V1 Fused':>10} {'V3 Dual':>10} | "
              f"{'Sep TFLOPS':>10} {'V1 TFLOPS':>10} {'V3 TFLOPS':>10} | "
              f"{'V1 Speedup':>10} {'V3 Speedup':>10}")
        print(f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} {'(us)':>10} | "
              f"{'':>10} {'':>10} {'':>10} | "
              f"{'vs Sep':>10} {'vs Sep':>10}")
        print(f"  {'-'*22}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-"
              f"{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES_TO_RUN:
        total_m = tokens_per_rank * num_ranks
        max_tokens_per_rank = tokens_per_rank

        # Create data
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_v3 = torch.zeros_like(y_fused)
        y_sep = torch.zeros_like(y_fused)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # Create symmetric buffer
        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
                group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
            )
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5}  | {'SKIP (buffer alloc failed)':>70} |")
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
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5}  | {'SKIP (separate failed)':>70} |")
            sym_buffer.destroy()
            dist.barrier()
            continue

        # -- Benchmark: V1 Fused (Pull-based single kernel, 384T) --
        def run_fused_v1():
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims="nk")

        try:
            time_v1_ms = bench_fn(run_fused_v1, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            time_v1_ms = float("inf")
            if rank_idx == 0:
                print(f"  V1 fused failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        # -- Benchmark: V3 Dual-kernel overlap (256T compute + 256T reduce) --
        def run_fused_v3():
            deep_gemm.bf16_gemm_rs_nt_v3(y_v3, a, b, sym_buffer, tokens_per_rank, compiled_dims="nk")

        try:
            time_v3_ms = bench_fn(run_fused_v3, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            time_v3_ms = float("inf")
            if rank_idx == 0:
                print(f"  V3 dual-kernel failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        # Compute metrics
        time_separate_us = time_separate_ms * 1000
        time_v1_us = time_v1_ms * 1000 if time_v1_ms != float("inf") else float("inf")
        time_v3_us = time_v3_ms * 1000 if time_v3_ms != float("inf") else float("inf")

        speedup_v1 = time_separate_us / time_v1_us if time_v1_us > 0 and time_v1_us != float("inf") else 0
        speedup_v3 = time_separate_us / time_v3_us if time_v3_us > 0 and time_v3_us != float("inf") else 0

        tflops_separate = compute_tflops(total_m, n_dim, k_dim, time_separate_ms)
        tflops_v1 = compute_tflops(total_m, n_dim, k_dim, time_v1_ms) if time_v1_ms != float("inf") else 0
        tflops_v3 = compute_tflops(total_m, n_dim, k_dim, time_v3_ms) if time_v3_ms != float("inf") else 0

        results.append({
            "tokens_per_rank": tokens_per_rank,
            "n_dim": n_dim,
            "k_dim": k_dim,
            "time_separate_us": time_separate_us,
            "time_v1_us": time_v1_us,
            "time_v3_us": time_v3_us,
            "speedup_v1": speedup_v1,
            "speedup_v3": speedup_v3,
            "tflops_separate": tflops_separate,
            "tflops_v1": tflops_v1,
            "tflops_v3": tflops_v3,
        })

        if rank_idx == 0:
            shape_str = f"{tokens_per_rank}x{n_dim}x{k_dim}"
            v1_str = f"{speedup_v1:.2f}x" if speedup_v1 > 0 else "FAIL"
            v3_str = f"{speedup_v3:.2f}x" if speedup_v3 > 0 else "FAIL"
            if speedup_v3 >= speedup_v1 and speedup_v3 > 0:
                v3_str = f"**{speedup_v3:.2f}x**"

            time_v1_str = f"{time_v1_us:>8.1f}" if time_v1_us != float("inf") else "     FAIL"
            time_v3_str = f"{time_v3_us:>8.1f}" if time_v3_us != float("inf") else "     FAIL"
            tflops_v1_str = f"{tflops_v1:>8.1f}T" if tflops_v1 > 0 else "     FAIL"
            tflops_v3_str = f"{tflops_v3:>8.1f}T" if tflops_v3 > 0 else "     FAIL"

            print(f"  {shape_str:<22} | {time_separate_us:>8.1f}u {time_v1_str}u {time_v3_str}u | "
                  f"{tflops_separate:>8.1f}T {tflops_v1_str} {tflops_v3_str} | "
                  f"{v1_str:>10} {v3_str:>10}")

        # Ensure async kernel failures are surfaced before symmetric memory teardown.
        try:
            torch.cuda.synchronize()
        except Exception as e:
            if rank_idx == 0:
                print(f"  CUDA sync failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        dist.barrier()
        try:
            sym_buffer.destroy()
        except Exception as e:
            if rank_idx == 0:
                print(f"  sym_buffer.destroy failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")
        dist.barrier()

    # -- Summary --
    if rank_idx == 0 and results:
        print(f"\n{'='*120}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'='*120}")

        # Overall stats
        v1_speedups = [r["speedup_v1"] for r in results if r["speedup_v1"] > 0]
        v3_speedups = [r["speedup_v3"] for r in results if r["speedup_v3"] > 0]

        geo_v1 = math.exp(sum(math.log(s) for s in v1_speedups) / len(v1_speedups)) if v1_speedups else 0
        geo_v3 = math.exp(sum(math.log(s) for s in v3_speedups) / len(v3_speedups)) if v3_speedups else 0

        print(f"\n  Overall Statistics (vs Separate GEMM+NCCL RS):")
        print(f"    V1 Fused (384T) geo_mean speedup: {geo_v1:.3f}x")
        print(f"    V3 Dual-Kernel (256T) geo_mean speedup: {geo_v3:.3f}x")
        if geo_v1 > 0 and geo_v3 > 0:
            v3_vs_v1 = geo_v3 / geo_v1
            print(f"    V3 vs V1 improvement ratio: {v3_vs_v1:.3f}x")

        # Best/worst
        if v1_speedups:
            print(f"\n  V1 Fused: best={max(v1_speedups):.2f}x, worst={min(v1_speedups):.2f}x")
        if v3_speedups:
            print(f"  V3 Dual:  best={max(v3_speedups):.2f}x, worst={min(v3_speedups):.2f}x")

        # By scenario
        print(f"\n  By Scenario:")
        for label, pred in [
            ("N=7168 (large hidden)", lambda r: r["n_dim"] == 7168),
            ("K=7168 (large input)", lambda r: r["k_dim"] == 7168),
            ("M/rank>=2048 (long ctx)", lambda r: r["tokens_per_rank"] >= 2048),
            ("M/rank>=8192 (very long)", lambda r: r["tokens_per_rank"] >= 8192),
        ]:
            subset = [r for r in results if pred(r)]
            if subset:
                sv1 = [r["speedup_v1"] for r in subset if r["speedup_v1"] > 0]
                sv3 = [r["speedup_v3"] for r in subset if r["speedup_v3"] > 0]
                g1 = math.exp(sum(math.log(s) for s in sv1) / len(sv1)) if sv1 else 0
                g3 = math.exp(sum(math.log(s) for s in sv3) / len(sv3)) if sv3 else 0
                print(f"    {label:<28} V1={g1:.3f}x  V3={g3:.3f}x  ({len(subset)} shapes)")

        # TFLOPS comparison
        print(f"\n  Average TFLOPS:")
        avg_sep = sum(r["tflops_separate"] for r in results) / len(results)
        avg_v1 = sum(r["tflops_v1"] for r in results if r["tflops_v1"] > 0) / max(1, sum(1 for r in results if r["tflops_v1"] > 0))
        avg_v3 = sum(r["tflops_v3"] for r in results if r["tflops_v3"] > 0) / max(1, sum(1 for r in results if r["tflops_v3"] > 0))
        print(f"    Separate: {avg_sep:.1f} TFLOPS")
        print(f"    V1 Fused: {avg_v1:.1f} TFLOPS")
        print(f"    V3 Dual:  {avg_v3:.1f} TFLOPS")

        # Top shapes
        print(f"\n  Top 5 Shapes for V3 (highest speedup over Separate):")
        sorted_v3 = sorted(results, key=lambda r: r["speedup_v3"], reverse=True)
        for i, r in enumerate(sorted_v3[:5]):
            shape_str = f"{r['tokens_per_rank']}x{r['n_dim']}x{r['k_dim']}"
            print(f"    {i+1}. {shape_str:<18} V3={r['speedup_v3']:.2f}x  V1={r['speedup_v1']:.2f}x  "
                  f"(sep={r['time_separate_us']:.0f}us, v3={r['time_v3_us']:.0f}us)")

        print(f"\n{'='*120}\n")

    dist.barrier()
    dist.destroy_process_group()

    # Avoid CUDA object destructor ordering issues at Python shutdown in mp workers.
    time.sleep(0.5)
    os._exit(0)


if __name__ == "__main__":
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
