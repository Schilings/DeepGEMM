"""
Quick benchmark: BF16 separate vs 2-kernel fused vs fully-fused (方案A)
Only large shapes (compute-bound) to focus on performance difference.

Usage: python benchmarks/bench_gemm_rs_quick.py [num_gpus] [num_iters]
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


SHAPES = [
    # (tokens_per_rank, N, K)
    (2048, 2048, 4096),
    (4096, 4096, 4096),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
]


def flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()


def bench_fn(fn, num_warmup=5, num_iters=10, barrier_group=None):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    if barrier_group is not None:
        dist.barrier(barrier_group)

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


def run_benchmark(local_rank, num_local_ranks, num_iters):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    if rank_idx == 0:
        print(f"\n{'='*90}")
        print(f"  BF16 GEMM-RS Quick Benchmark: {num_ranks} GPUs, {num_iters} iters")
        print(f"{'='*90}")
        print(f"{'Shape (MxNxK)':>20} | {'Separate':>10} | {'2-Kernel':>10} | {'Fused(A)':>10} | {'2K vs Sep':>10} | {'Fused vs Sep':>12}")
        print("-" * 90)

    for tpr, n, k in SHAPES:
        total_m = tpr * num_ranks
        flops = 2.0 * total_m * n * k

        a = torch.randn((total_m, k), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
        dist.broadcast(a, src=0)

        # --- Separate: gemm + reduce_scatter ---
        d_full = torch.zeros((total_m, n), dtype=torch.bfloat16, device='cuda')
        y_sep = torch.zeros((tpr, n), dtype=torch.bfloat16, device='cuda')

        def sep():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        t_sep = bench_fn(sep, num_iters=num_iters, barrier_group=group)

        # --- 2-kernel fused ---
        sym_buf = deep_gemm.get_symm_buffer_for_gemm_rs(group, tpr, n, out_dtype=torch.bfloat16)
        y_2k = torch.zeros((tpr, n), dtype=torch.bfloat16, device='cuda')

        def fused_2k():
            deep_gemm.bf16_gemm_rs_nt(y_2k, a, b, sym_buf, tpr, compiled_dims='nk')

        t_2k = bench_fn(fused_2k, num_iters=num_iters, barrier_group=group)

        # --- Fully fused (方案A) ---
        y_full = torch.zeros((tpr, n), dtype=torch.bfloat16, device='cuda')

        def fused_full():
            deep_gemm.bf16_gemm_rs_fused(y_full, a, b, sym_buf, tpr, compiled_dims='nk')

        t_full = bench_fn(fused_full, num_iters=num_iters, barrier_group=group)

        if rank_idx == 0:
            sp_2k = t_sep / t_2k if t_2k > 0 else 0
            sp_full = t_sep / t_full if t_full > 0 else 0
            shape_str = f"{total_m}x{n}x{k}"
            tflops_sep = flops / t_sep / 1e12
            tflops_2k = flops / t_2k / 1e12
            tflops_full = flops / t_full / 1e12
            print(f"{shape_str:>20} | {t_sep*1e6:>8.1f}us | {t_2k*1e6:>8.1f}us | {t_full*1e6:>8.1f}us | {sp_2k:>9.3f}x | {sp_full:>11.3f}x")

        sym_buf.destroy()
        del d_full, y_sep, y_2k, y_full, a, b
        torch.cuda.empty_cache()
        dist.barrier(group)

    if rank_idx == 0:
        print(f"{'='*90}\n")

    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"Launching quick benchmark: {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
