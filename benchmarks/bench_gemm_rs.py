"""
GEMM-RS Performance Benchmark: Main Fused vs Separate (GEMM + NCCL RS)

Compares:
  1. bf16_gemm_nt + torch.distributed.reduce_scatter_tensor  -- separate baseline
  2. bf16_gemm_rs_nt (current production fused path)          -- main fused

Usage:
  python benchmarks/bench_gemm_rs.py [num_gpus] [num_iters]

Optional environment variables:
  DG_BENCH_SINGLE_SHAPE="M,N,K"   # run exactly one shape
  DG_BENCH_MAX_SHAPES=1            # run first N shapes from standard list
  DG_BENCH_SYNC_EACH_ITER=1        # synchronize each iteration for diagnostics
"""

import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


SHAPES_STANDARD = [
    # Medium
    (1024, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (2048, 7168, 2048),
    # Large
    (4096, 7168, 2048),
    (4096, 2048, 7168),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),
    # Very Large / Extreme
    (8192, 7168, 2048),
    (8192, 2048, 7168),
    (8192, 4096, 4096),
    (8192, 7168, 4096),
    (16384, 7168, 2048),
    (16384, 2048, 7168),
    (16384, 4096, 4096),
    (16384, 7168, 4096),
    (8192, 7168, 7168),
    (16384, 7168, 7168),
    (20480, 7168, 2048),
]


def get_shapes_to_run():
    single_shape = os.getenv("DG_BENCH_SINGLE_SHAPE", "").strip()
    if single_shape:
        m_str, n_str, k_str = [x.strip() for x in single_shape.split(",")]
        return [(int(m_str), int(n_str), int(k_str))]

    max_shapes = int(os.getenv("DG_BENCH_MAX_SHAPES", "0"))
    return SHAPES_STANDARD[:max_shapes] if max_shapes > 0 else SHAPES_STANDARD


def flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda").zero_()


def compute_tflops(m, n, k, time_ms):
    flops = 2.0 * m * n * k
    return flops / (time_ms * 1e-3) / 1e12


def bench_fn(fn, num_warmup=5, num_iters=20, barrier_group=None):
    sync_each_iter = bool(int(os.getenv("DG_BENCH_SYNC_EACH_ITER", "0")))

    for _ in range(num_warmup):
        fn()
        if sync_each_iter:
            torch.cuda.synchronize()
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
        if sync_each_iter:
            torch.cuda.synchronize()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


def run_benchmark(local_rank: int, num_local_ranks: int, num_iters: int = 20):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    device = f"cuda:{local_rank}"
    shapes_to_run = get_shapes_to_run()

    if rank_idx == 0:
        print(f"\n{'=' * 108}")
        print(f"  GEMM-RS Benchmark (Main Fused vs Separate), GPUs={num_ranks}, iters={num_iters}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'=' * 108}\n")
        print(f"  {'Shape':<22} | {'Separate':>10} {'Fused':>10} | {'Sep TFLOPS':>11} {'Fused TFLOPS':>12} | {'Speedup':>9}")
        print(f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} | {'':>11} {'':>12} | {'vs Sep':>9}")
        print(f"  {'-' * 22}-+-{'-' * 10}-{'-' * 10}-+-{'-' * 11}-{'-' * 12}-+-{'-' * 9}")

    results = []

    for tokens_per_rank, n_dim, k_dim in shapes_to_run:
        total_m = tokens_per_rank * num_ranks

        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_fused)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | SKIP (symm buffer alloc failed: {e})")
            dist.barrier(group)
            continue

        dist.barrier(group)

        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        def run_fused():
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims="nk")

        try:
            time_separate_ms = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | SKIP (separate failed: {e})")
            try:
                sym_buffer.destroy()
            finally:
                dist.barrier(group)
            continue

        try:
            time_fused_ms = bench_fn(run_fused, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            time_fused_ms = float("inf")
            if rank_idx == 0:
                print(f"  fused failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        time_separate_us = time_separate_ms * 1000.0
        time_fused_us = time_fused_ms * 1000.0 if time_fused_ms != float("inf") else float("inf")

        speedup = time_separate_us / time_fused_us if time_fused_us not in (0, float("inf")) else 0.0
        tflops_sep = compute_tflops(total_m, n_dim, k_dim, time_separate_ms)
        tflops_fused = compute_tflops(total_m, n_dim, k_dim, time_fused_ms) if time_fused_ms != float("inf") else 0.0

        results.append({
            "tokens_per_rank": tokens_per_rank,
            "n_dim": n_dim,
            "k_dim": k_dim,
            "time_separate_us": time_separate_us,
            "time_fused_us": time_fused_us,
            "speedup": speedup,
            "tflops_separate": tflops_sep,
            "tflops_fused": tflops_fused,
        })

        if rank_idx == 0:
            shape_str = f"{tokens_per_rank}x{n_dim}x{k_dim}"
            fused_time_str = f"{time_fused_us:>8.1f}" if time_fused_us != float("inf") else "    FAIL"
            fused_tflops_str = f"{tflops_fused:>8.1f}T" if tflops_fused > 0 else "    FAIL"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "FAIL"

            print(
                f"  {shape_str:<22} | {time_separate_us:>8.1f}u {fused_time_str}u | "
                f"{tflops_sep:>9.1f}T {fused_tflops_str:>12} | {speedup_str:>9}"
            )

        # Surface async failure before teardown
        try:
            torch.cuda.synchronize()
        except Exception as e:
            if rank_idx == 0:
                print(f"  CUDA sync failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        dist.barrier(group)
        try:
            sym_buffer.destroy()
        except Exception as e:
            if rank_idx == 0:
                print(f"  sym_buffer.destroy failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")
        dist.barrier(group)

    if rank_idx == 0 and results:
        print(f"\n{'=' * 108}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'=' * 108}")

        speedups = [r["speedup"] for r in results if r["speedup"] > 0]
        geo_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 0.0

        print("\n  Overall Statistics (vs Separate GEMM+NCCL RS):")
        print(f"    Main fused geo_mean speedup: {geo_speedup:.3f}x")
        if speedups:
            print(f"    Best speedup: {max(speedups):.2f}x")
            print(f"    Worst speedup: {min(speedups):.2f}x")

        print("\n  By Scenario:")
        scenarios = [
            ("N=7168 (large hidden)", lambda r: r["n_dim"] == 7168),
            ("K=7168 (large input)", lambda r: r["k_dim"] == 7168),
            ("M/rank>=2048 (long ctx)", lambda r: r["tokens_per_rank"] >= 2048),
            ("M/rank>=8192 (very long)", lambda r: r["tokens_per_rank"] >= 8192),
        ]
        for label, pred in scenarios:
            subset = [r for r in results if pred(r)]
            if not subset:
                continue
            subset_speedups = [r["speedup"] for r in subset if r["speedup"] > 0]
            subset_geo = math.exp(sum(math.log(s) for s in subset_speedups) / len(subset_speedups)) if subset_speedups else 0.0
            print(f"    {label:<28} speedup={subset_geo:.3f}x  ({len(subset)} shapes)")

        avg_sep = sum(r["tflops_separate"] for r in results) / len(results)
        fused_valid = [r["tflops_fused"] for r in results if r["tflops_fused"] > 0]
        avg_fused = sum(fused_valid) / len(fused_valid) if fused_valid else 0.0

        print("\n  Average TFLOPS:")
        print(f"    Separate:   {avg_sep:.1f} TFLOPS")
        print(f"    Main fused: {avg_fused:.1f} TFLOPS")

        print(f"\n{'=' * 108}\n")

    dist.barrier(group)
    dist.destroy_process_group()
    time.sleep(0.5)
    os._exit(0)


if __name__ == "__main__":
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
