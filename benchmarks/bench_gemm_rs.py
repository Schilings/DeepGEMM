"""
Benchmark: Fused GEMM-RS vs Separate GEMM + reduce_scatter

Compares end-to-end latency and TFLOPS for:
  1. BF16: bf16_gemm_rs_nt (fused) vs bf16_gemm_nt + reduce_scatter (separate)
  2. FP8:  fp8_gemm_rs_nt (fused) vs fp8_gemm_nt + reduce_scatter (separate)

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
from deep_gemm.utils.math import per_token_cast_to_fp8


# ─── Typical MoE / Dense shapes ───
# (tokens_per_rank, N, K) — N is output dim, K is hidden dim
SHAPES = [
    # Small
    (256, 512, 1024),
    (256, 1024, 2048),
    # Medium (typical MoE expert dims)
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
    """Benchmark a function, returns average time in seconds."""
    # Warmup
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    flush_l2()
    if barrier_group is not None:
        dist.barrier(barrier_group)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters / 1e3  # seconds


def run_benchmark(local_rank: int, num_local_ranks: int, num_iters: int = 20):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    device = f'cuda:{local_rank}'

    if rank_idx == 0:
        print(f"\n{'═'*80}")
        print(f"  GEMM-RS Benchmark: {num_ranks} GPUs, {num_iters} iterations per measurement")
        print(f"{'═'*80}")
        print()
        header = (f"{'Shape (M×N×K)':>20} │ {'Method':^30} │ "
                  f"{'Time (μs)':>10} │ {'TFLOPS':>8} │ {'Speedup':>8}")
        print(header)
        print("─" * len(header))

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_ranks
        flops = 2.0 * total_m * n_dim * k_dim  # total FLOPs across all ranks

        # ═══════════════════════════════════════════════════════════════════
        # BF16 Benchmark
        # ═══════════════════════════════════════════════════════════════════
        a_bf16 = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b_bf16 = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        dist.broadcast(a_bf16, src=0)

        # --- Separate: bf16_gemm_nt + reduce_scatter ---
        d_full_bf16 = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
        y_separate_bf16 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def bf16_separate():
            deep_gemm.bf16_gemm_nt(a_bf16, b_bf16, d_full_bf16)
            dist.reduce_scatter_tensor(y_separate_bf16, d_full_bf16, op=dist.ReduceOp.SUM, group=group)

        t_separate_bf16 = bench_fn(bf16_separate, num_iters=num_iters, barrier_group=group)

        # --- Fused: bf16_gemm_rs_nt ---
        sym_buffer_bf16 = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )
        y_fused_bf16 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def bf16_fused():
            deep_gemm.bf16_gemm_rs_nt(y_fused_bf16, a_bf16, b_bf16, sym_buffer_bf16,
                                       tokens_per_rank, compiled_dims='nk')

        t_fused_bf16 = bench_fn(bf16_fused, num_iters=num_iters, barrier_group=group)

        # --- Full Fused: bf16_gemm_rs_fused (single kernel, no reduce kernel) ---
        y_fused_full = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def bf16_fused_full():
            deep_gemm.bf16_gemm_rs_fused(y_fused_full, a_bf16, b_bf16, sym_buffer_bf16,
                                          tokens_per_rank, compiled_dims='nk')

        t_fused_full = bench_fn(bf16_fused_full, num_iters=num_iters, barrier_group=group)

        speedup_bf16 = t_separate_bf16 / t_fused_bf16 if t_fused_bf16 > 0 else float('inf')
        speedup_full = t_separate_bf16 / t_fused_full if t_fused_full > 0 else float('inf')

        if rank_idx == 0:
            shape_str = f"{total_m}×{n_dim}×{k_dim}"
            print(f"{shape_str:>20} │ {'BF16 separate (gemm+RS)':^30} │ "
                  f"{t_separate_bf16*1e6:10.1f} │ {flops/t_separate_bf16/1e12:8.1f} │ {'baseline':>8}")
            print(f"{'':>20} │ {'BF16 fused 2-kernel':^30} │ "
                  f"{t_fused_bf16*1e6:10.1f} │ {flops/t_fused_bf16/1e12:8.1f} │ {speedup_bf16:7.2f}x")
            print(f"{'':>20} │ {'BF16 fused FULL (方案A)':^30} │ "
                  f"{t_fused_full*1e6:10.1f} │ {flops/t_fused_full/1e12:8.1f} │ {speedup_full:7.2f}x")

        sym_buffer_bf16.destroy()
        del d_full_bf16, y_separate_bf16, y_fused_bf16, y_fused_full
        dist.barrier(group)

        # ═══════════════════════════════════════════════════════════════════
        # FP8 Benchmark
        # ═══════════════════════════════════════════════════════════════════
        gran_k = 128
        a_fp8, a_sf = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=gran_k)
        b_fp8, b_sf = per_token_cast_to_fp8(b_bf16, use_ue8m0=True, gran_k=gran_k)

        # --- Separate: fp8_gemm_nt + reduce_scatter ---
        d_full_fp8 = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
        y_separate_fp8 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def fp8_separate():
            deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d_full_fp8, recipe=(1, 1, gran_k))
            dist.reduce_scatter_tensor(y_separate_fp8, d_full_fp8, op=dist.ReduceOp.SUM, group=group)

        t_separate_fp8 = bench_fn(fp8_separate, num_iters=num_iters, barrier_group=group)

        # --- Fused: fp8_gemm_rs_nt (BF16 comm) ---
        sym_buffer_fp8 = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )
        y_fused_fp8 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def fp8_fused_bf16comm():
            deep_gemm.fp8_gemm_rs_nt(y_fused_fp8, (a_fp8, a_sf), (b_fp8, b_sf), sym_buffer_fp8,
                                      tokens_per_rank, recipe=(1, 1, gran_k),
                                      compiled_dims='nk')

        t_fused_fp8_bf16 = bench_fn(fp8_fused_bf16comm, num_iters=num_iters, barrier_group=group)

        # --- Fused: fp8_gemm_rs_nt (FP32 comm) ---
        sym_buffer_fp8_fp32 = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16,
            comm_dtype=torch.float32
        )
        y_fused_fp8_fp32 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)

        def fp8_fused_fp32comm():
            deep_gemm.fp8_gemm_rs_nt(y_fused_fp8_fp32, (a_fp8, a_sf), (b_fp8, b_sf), sym_buffer_fp8_fp32,
                                      tokens_per_rank, recipe=(1, 1, gran_k),
                                      compiled_dims='nk',
                                      comm_dtype='fp32', reduce_in_fp32=True)

        t_fused_fp8_fp32 = bench_fn(fp8_fused_fp32comm, num_iters=num_iters, barrier_group=group)

        speedup_fp8_bf16 = t_separate_fp8 / t_fused_fp8_bf16 if t_fused_fp8_bf16 > 0 else float('inf')
        speedup_fp8_fp32 = t_separate_fp8 / t_fused_fp8_fp32 if t_fused_fp8_fp32 > 0 else float('inf')

        if rank_idx == 0:
            print(f"{'':>20} │ {'FP8 separate (gemm+RS)':^30} │ "
                  f"{t_separate_fp8*1e6:10.1f} │ {flops/t_separate_fp8/1e12:8.1f} │ {'baseline':>8}")
            print(f"{'':>20} │ {'FP8 fused (BF16 comm)':^30} │ "
                  f"{t_fused_fp8_bf16*1e6:10.1f} │ {flops/t_fused_fp8_bf16/1e12:8.1f} │ {speedup_fp8_bf16:7.2f}x")
            print(f"{'':>20} │ {'FP8 fused (FP32 comm)':^30} │ "
                  f"{t_fused_fp8_fp32*1e6:10.1f} │ {flops/t_fused_fp8_fp32/1e12:8.1f} │ {speedup_fp8_fp32:7.2f}x")
            print("─" * 90)

        results.append({
            'shape': (total_m, n_dim, k_dim),
            'bf16_separate_us': t_separate_bf16 * 1e6,
            'bf16_fused_us': t_fused_bf16 * 1e6,
            'bf16_speedup': speedup_bf16,
            'fp8_separate_us': t_separate_fp8 * 1e6,
            'fp8_fused_bf16_us': t_fused_fp8_bf16 * 1e6,
            'fp8_fused_fp32_us': t_fused_fp8_fp32 * 1e6,
            'fp8_speedup_bf16': speedup_fp8_bf16,
            'fp8_speedup_fp32': speedup_fp8_fp32,
        })

        sym_buffer_fp8.destroy()
        sym_buffer_fp8_fp32.destroy()
        del d_full_fp8, y_separate_fp8, y_fused_fp8, y_fused_fp8_fp32, a_bf16, b_bf16
        dist.barrier(group)

    # ─── Summary ───
    if rank_idx == 0 and results:
        print(f"\n{'═'*80}")
        print("  SUMMARY")
        print(f"{'═'*80}")

        bf16_speedups = [r['bf16_speedup'] for r in results]
        fp8_bf16_speedups = [r['fp8_speedup_bf16'] for r in results]
        fp8_fp32_speedups = [r['fp8_speedup_fp32'] for r in results]

        import numpy as np
        print(f"\n  BF16 fused vs separate:")
        print(f"    Geometric mean speedup: {float(np.prod(bf16_speedups)) ** (1.0/len(bf16_speedups)):.3f}x")
        print(f"    Min / Max speedup:      {min(bf16_speedups):.3f}x / {max(bf16_speedups):.3f}x")

        print(f"\n  FP8 fused (BF16 comm) vs separate:")
        print(f"    Geometric mean speedup: {float(np.prod(fp8_bf16_speedups)) ** (1.0/len(fp8_bf16_speedups)):.3f}x")
        print(f"    Min / Max speedup:      {min(fp8_bf16_speedups):.3f}x / {max(fp8_bf16_speedups):.3f}x")

        print(f"\n  FP8 fused (FP32 comm) vs separate:")
        print(f"    Geometric mean speedup: {float(np.prod(fp8_fp32_speedups)) ** (1.0/len(fp8_fp32_speedups)):.3f}x")
        print(f"    Min / Max speedup:      {min(fp8_fp32_speedups):.3f}x / {max(fp8_fp32_speedups):.3f}x")

        print(f"\n{'═'*80}\n")

    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"Launching GEMM-RS benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
