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
    # User-specified Megatron SP oriented shape set
    (1024, 7168, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),
    (8192, 4096, 4096),
    (8192, 7168, 4096),
    (8192, 7168, 7168),
    (2048, 7168, 2048),
    (4096, 7168, 2048),
    (16384, 7168, 4096),
    (16384, 7168, 7168),
]

SHAPES_FOCUS = [
    # User-highlighted medium/large focus set
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),
    (8192, 4096, 4096),
    (8192, 7168, 4096),
]


def parse_shapes_list(shapes_str: str):
    shapes = []
    for item in shapes_str.split(";"):
        token = item.strip().lower().replace("x", ",")
        if not token:
            continue
        m_str, n_str, k_str = [x.strip() for x in token.split(",")]
        shapes.append((int(m_str), int(n_str), int(k_str)))
    return shapes


def get_shapes_to_run():
    explicit_shapes = os.getenv("DG_BENCH_SHAPES", "").strip()
    if explicit_shapes:
        parsed = parse_shapes_list(explicit_shapes)
        if parsed:
            return parsed

    focus_only = bool(int(os.getenv("DG_BENCH_FOCUS_ONLY", "0")))
    if focus_only:
        return SHAPES_FOCUS

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
        print(f"\n{'=' * 144}")
        print(f"  GEMM-RS Benchmark (Fused vs Torch-Native/DeepGEMM-Separate), GPUs={num_ranks}, iters={num_iters}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'=' * 144}\n")
        print(
            f"  {'Shape':<22} | {'Torch':>10} {'Separate':>10} {'Fused':>10} | "
            f"{'Torch TFLOPS':>12} {'Sep TFLOPS':>11} {'Fused TFLOPS':>12} | {'vs Torch':>9} {'vs Sep':>9}"
        )
        print(
            f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} {'(us)':>10} | "
            f"{'':>12} {'':>11} {'':>12} | {'speedup':>9} {'speedup':>9}"
        )
        print(f"  {'-' * 22}-+-{'-' * 10}-{'-' * 10}-{'-' * 10}-+-{'-' * 12}-{'-' * 11}-{'-' * 12}-+-{'-' * 9}-{'-' * 9}")

    results = []

    for tokens_per_rank, n_dim, k_dim in shapes_to_run:
        total_m = tokens_per_rank * num_ranks

        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros_like(y_fused)
        y_torch = torch.zeros_like(y_fused)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
        d_full_torch = torch.zeros_like(d_full)

        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | SKIP (symm buffer alloc failed: {e})")
            dist.barrier(group)
            continue

        dist.barrier(group)

        def run_torch_native():
            torch.matmul(a, b.t(), out=d_full_torch)
            dist.reduce_scatter_tensor(y_torch, d_full_torch, op=dist.ReduceOp.SUM, group=group)

        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        def run_fused():
            deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims="nk")

        try:
            time_torch_ms = bench_fn(run_torch_native, num_iters=num_iters, barrier_group=group)
            time_separate_ms = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | SKIP (baseline failed: {e})")
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

        time_torch_us = time_torch_ms * 1000.0
        time_separate_us = time_separate_ms * 1000.0
        time_fused_us = time_fused_ms * 1000.0 if time_fused_ms != float("inf") else float("inf")

        speedup_vs_torch = time_torch_us / time_fused_us if time_fused_us not in (0, float("inf")) else 0.0
        speedup_vs_sep = time_separate_us / time_fused_us if time_fused_us not in (0, float("inf")) else 0.0
        tflops_torch = compute_tflops(total_m, n_dim, k_dim, time_torch_ms)
        tflops_sep = compute_tflops(total_m, n_dim, k_dim, time_separate_ms)
        tflops_fused = compute_tflops(total_m, n_dim, k_dim, time_fused_ms) if time_fused_ms != float("inf") else 0.0

        results.append({
            "tokens_per_rank": tokens_per_rank,
            "n_dim": n_dim,
            "k_dim": k_dim,
            "time_torch_us": time_torch_us,
            "time_separate_us": time_separate_us,
            "time_fused_us": time_fused_us,
            "speedup_vs_torch": speedup_vs_torch,
            "speedup_vs_sep": speedup_vs_sep,
            "tflops_torch": tflops_torch,
            "tflops_separate": tflops_sep,
            "tflops_fused": tflops_fused,
        })

        if rank_idx == 0:
            shape_str = f"{tokens_per_rank}x{n_dim}x{k_dim}"
            fused_time_str = f"{time_fused_us:>8.1f}" if time_fused_us != float("inf") else "    FAIL"
            fused_tflops_str = f"{tflops_fused:>8.1f}T" if tflops_fused > 0 else "    FAIL"
            speedup_torch_str = f"{speedup_vs_torch:.2f}x" if speedup_vs_torch > 0 else "FAIL"
            speedup_sep_str = f"{speedup_vs_sep:.2f}x" if speedup_vs_sep > 0 else "FAIL"

            print(
                f"  {shape_str:<22} | {time_torch_us:>8.1f}u {time_separate_us:>8.1f}u {fused_time_str}u | "
                f"{tflops_torch:>10.1f}T {tflops_sep:>9.1f}T {fused_tflops_str:>12} | "
                f"{speedup_torch_str:>9} {speedup_sep_str:>9}"
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
        print(f"\n{'=' * 144}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'=' * 144}")

        speedups_vs_torch = [r["speedup_vs_torch"] for r in results if r["speedup_vs_torch"] > 0]
        speedups_vs_sep = [r["speedup_vs_sep"] for r in results if r["speedup_vs_sep"] > 0]
        geo_vs_torch = math.exp(sum(math.log(s) for s in speedups_vs_torch) / len(speedups_vs_torch)) if speedups_vs_torch else 0.0
        geo_vs_sep = math.exp(sum(math.log(s) for s in speedups_vs_sep) / len(speedups_vs_sep)) if speedups_vs_sep else 0.0

        print("\n  Overall Statistics:")
        print(f"    Main fused geo_mean speedup vs torch-native: {geo_vs_torch:.3f}x")
        print(f"    Main fused geo_mean speedup vs deepgemm-separate: {geo_vs_sep:.3f}x")
        if speedups_vs_torch:
            print(f"    Best speedup vs torch-native: {max(speedups_vs_torch):.2f}x")
            print(f"    Worst speedup vs torch-native: {min(speedups_vs_torch):.2f}x")
        if speedups_vs_sep:
            print(f"    Best speedup vs deepgemm-separate: {max(speedups_vs_sep):.2f}x")
            print(f"    Worst speedup vs deepgemm-separate: {min(speedups_vs_sep):.2f}x")

        print("\n  By Scenario:")
        focus_set = set(SHAPES_FOCUS)
        scenarios = [
            ("N=7168 (large hidden)", lambda r: r["n_dim"] == 7168),
            ("K=7168 (large input)", lambda r: r["k_dim"] == 7168),
            ("M/rank>=2048 (long ctx)", lambda r: r["tokens_per_rank"] >= 2048),
            ("M/rank>=8192 (very long)", lambda r: r["tokens_per_rank"] >= 8192),
            ("User focus medium/large", lambda r: (r["tokens_per_rank"], r["n_dim"], r["k_dim"]) in focus_set),
        ]
        for label, pred in scenarios:
            subset = [r for r in results if pred(r)]
            if not subset:
                continue
            subset_vs_torch = [r["speedup_vs_torch"] for r in subset if r["speedup_vs_torch"] > 0]
            subset_vs_sep = [r["speedup_vs_sep"] for r in subset if r["speedup_vs_sep"] > 0]
            subset_geo_vs_torch = math.exp(sum(math.log(s) for s in subset_vs_torch) / len(subset_vs_torch)) if subset_vs_torch else 0.0
            subset_geo_vs_sep = math.exp(sum(math.log(s) for s in subset_vs_sep) / len(subset_vs_sep)) if subset_vs_sep else 0.0
            print(
                f"    {label:<28} vs_torch={subset_geo_vs_torch:.3f}x  "
                f"vs_sep={subset_geo_vs_sep:.3f}x  ({len(subset)} shapes)"
            )

        avg_torch = sum(r["tflops_torch"] for r in results) / len(results)
        avg_sep = sum(r["tflops_separate"] for r in results) / len(results)
        fused_valid = [r["tflops_fused"] for r in results if r["tflops_fused"] > 0]
        avg_fused = sum(fused_valid) / len(fused_valid) if fused_valid else 0.0

        print("\n  Average TFLOPS:")
        print(f"    Torch native:       {avg_torch:.1f} TFLOPS")
        print(f"    DeepGEMM separate:  {avg_sep:.1f} TFLOPS")
        print(f"    Main fused:         {avg_fused:.1f} TFLOPS")

        print(f"\n{'=' * 144}\n")

    dist.barrier(group)
    dist.destroy_process_group()
    time.sleep(0.5)
    os._exit(0)


if __name__ == "__main__":
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20

    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
