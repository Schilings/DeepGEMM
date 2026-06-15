"""
GEMM + A2A + PDL Benchmark: Compare three approaches

  1. Separate: bf16_gemm_nt (full M) + NCCL reduce_scatter_tensor
  2. V3 Dual-Kernel: bf16_gemm_rs_nt_v3 (256T compute + 256T reduce, stream overlap)
  3. A2A PDL: bf16_gemm_nt (my_rows only) × num_ranks  — no communication!

Usage:
    python benchmarks/bench_gemm_a2a_pdl.py [num_gpus] [num_iters]
"""

import os
import sys
import math
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SHAPES = [
    (1024, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (2048, 7168, 2048),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
    (4096, 4096, 7168),
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
]


def compute_tflops(m, n, k, time_ms):
    return 2.0 * m * n * k / (time_ms * 1e-3) / 1e12


def flush_l2():
    torch.empty(int(256e6 // 4), dtype=torch.int32, device='cuda').zero_()


def bench_fn(fn, num_warmup=5, num_iters=20, group=None):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()
    if group is not None:
        dist.barrier(group)

    flush_l2()
    if group is not None:
        dist.barrier(group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters


def run_benchmark(local_rank, num_local_ranks, num_iters=20):
    from deep_gemm.utils.dist import init_dist
    import deep_gemm

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    if rank_idx == 0:
        print(f"\n{'='*130}")
        print(f"  GEMM-RS Benchmark: {num_ranks} GPUs, {num_iters} iters")
        print(f"  Separate=GEMM(full M)+NCCL RS, V3=Dual-kernel overlap, A2A PDL=GEMM(my_rows)*num_ranks")
        print(f"{'='*130}")
        print(f"\n  {'Shape':<22} | {'Separate':>10} {'V3 Dual':>10} {'A2A PDL':>10} | "
              f"{'Sep TFLOPS':>10} {'V3 TFLOPS':>10} {'A2A TFLOPS':>10} | "
              f"{'V3 Speedup':>10} {'A2A Speedup':>10}")
        print(f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} {'(us)':>10} | "
              f"{'':>10} {'':>10} {'':>10} | "
              f"{'vs Sep':>10} {'vs Sep':>10}")
        print(f"  {'-'*22}-+-{'-'*10}-{'-'*10}-{'-'*10}-+-"
              f"{'-'*10}-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_ranks
        start_row = rank_idx * tokens_per_rank
        end_row = start_row + tokens_per_rank

        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_v3 = torch.zeros_like(y_sep)
        y_a2a = torch.zeros_like(y_sep)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # ── Sym buffer for V3 ──
        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
                group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16)
        except Exception:
            sym_buffer = None

        dist.barrier()

        # ── Separate: GEMM (full M) + NCCL RS ──
        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        time_sep_ms = bench_fn(run_separate, num_iters=num_iters, group=group)

        # ── V3 Dual-kernel ──
        time_v3_ms = float('inf')
        if sym_buffer is not None:
            def run_v3():
                deep_gemm.bf16_gemm_rs_nt_v3(y_v3, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
            try:
                time_v3_ms = bench_fn(run_v3, num_iters=num_iters, group=group)
            except Exception as e:
                time_v3_ms = float('inf')
                if rank_idx == 0:
                    print(f"  V3 failed for {tokens_per_rank}x{n_dim}x{k_dim}: {e}")

        # ── A2A PDL: GEMM (my rows only) × num_ranks ──
        def run_a2a_pdl():
            deep_gemm.bf16_gemm_nt(a[start_row:end_row, :], b, y_a2a)
            y_a2a.mul_(num_ranks)

        time_a2a_ms = bench_fn(run_a2a_pdl, num_iters=num_iters, group=group)

        # Compute metrics
        sep_us = time_sep_ms * 1000
        v3_us = time_v3_ms * 1000 if time_v3_ms != float('inf') else float('inf')
        a2a_us = time_a2a_ms * 1000

        speedup_v3 = sep_us / v3_us if v3_us > 0 and v3_us != float('inf') else 0
        speedup_a2a = sep_us / a2a_us if a2a_us > 0 else 0

        # TFLOPS: all use total_m for fair comparison
        tf_sep = compute_tflops(total_m, n_dim, k_dim, time_sep_ms)
        tf_v3 = compute_tflops(total_m, n_dim, k_dim, time_v3_ms) if time_v3_ms != float('inf') else 0
        # A2A PDL computes M_per_rank * N * K * 2 per rank, total = total_m * N * K * 2
        tf_a2a = compute_tflops(total_m, n_dim, k_dim, time_a2a_ms)

        results.append({
            'tokens_per_rank': tokens_per_rank, 'n': n_dim, 'k': k_dim,
            'sep_us': sep_us, 'v3_us': v3_us, 'a2a_us': a2a_us,
            'speedup_v3': speedup_v3, 'speedup_a2a': speedup_a2a,
            'tf_sep': tf_sep, 'tf_v3': tf_v3, 'tf_a2a': tf_a2a,
        })

        if rank_idx == 0:
            v3_s = f"{speedup_v3:.2f}x" if speedup_v3 > 0 else "FAIL"
            a2a_s = f"**{speedup_a2a:.2f}x**" if speedup_a2a >= speedup_v3 else f"{speedup_a2a:.2f}x"
            print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | {sep_us:>8.1f}u "
                  f"{v3_us:>8.1f}u {a2a_us:>8.1f}u | "
                  f"{tf_sep:>8.1f}T {tf_v3:>8.1f}T {tf_a2a:>8.1f}T | "
                  f"{v3_s:>10} {a2a_s:>10}")

        if sym_buffer is not None:
            sym_buffer.destroy()
        dist.barrier()

    # ── Summary ──
    if rank_idx == 0 and results:
        print(f"\n{'='*130}")
        v3_speeds = [r['speedup_v3'] for r in results if r['speedup_v3'] > 0]
        a2a_speeds = [r['speedup_a2a'] for r in results if r['speedup_a2a'] > 0]
        geo_v3 = math.exp(sum(math.log(s) for s in v3_speeds) / len(v3_speeds)) if v3_speeds else 0
        geo_a2a = math.exp(sum(math.log(s) for s in a2a_speeds) / len(a2a_speeds)) if a2a_speeds else 0
        print(f"  Geometric Mean Speedup (vs Separate GEMM+NCCL RS):")
        print(f"    V3 Dual-Kernel: {geo_v3:.3f}x")
        print(f"    A2A PDL:        {geo_a2a:.3f}x")
        if geo_v3 > 0 and geo_a2a > 0:
            print(f"    A2A PDL vs V3:  {geo_a2a/geo_v3:.3f}x")

        avg_tf_sep = sum(r['tf_sep'] for r in results) / len(results)
        avg_tf_v3 = sum(r['tf_v3'] for r in results if r['tf_v3'] > 0) / max(1, sum(1 for r in results if r['tf_v3'] > 0))
        avg_tf_a2a = sum(r['tf_a2a'] for r in results if r['tf_a2a'] > 0) / max(1, sum(1 for r in results if r['tf_a2a'] > 0))
        print(f"\n  Average TFLOPS:")
        print(f"    Separate: {avg_tf_sep:.1f} TFLOPS")
        print(f"    V3 Dual:  {avg_tf_v3:.1f} TFLOPS")
        print(f"    A2A PDL:  {avg_tf_a2a:.1f} TFLOPS")
        print(f"{'='*130}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
