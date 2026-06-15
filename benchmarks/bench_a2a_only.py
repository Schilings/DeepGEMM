"""
GEMM + A2A + PDL Benchmark: Separate vs A2A PDL (V3 skipped due to barrier bug)

Usage:
    python benchmarks/bench_a2a_only.py [num_gpus] [num_iters]
"""

import os, sys, math, time, torch, torch.distributed as dist, torch.multiprocessing as mp

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
        print(f"\n{'='*100}")
        print(f"  GEMM-RS Benchmark: {num_ranks} GPUs, {num_iters} iters")
        print(f"  Separate=GEMM(full M)+NCCL RS | A2A PDL=GEMM(my_rows)*num_ranks (zero communication)")
        print(f"{'='*100}")
        print(f"\n  {'Shape':<22} | {'Separate':>10} {'A2A PDL':>10} | "
              f"{'Sep TFLOPS':>10} {'A2A TFLOPS':>10} | {'Speedup':>10}")
        print(f"  {'(M/rank x N x K)':<22} | {'(us)':>10} {'(us)':>10} | "
              f"{'':>10} {'':>10} | {'A2A vs Sep':>10}")
        print(f"  {'-'*22}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*10}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_ranks
        start_row = rank_idx * tokens_per_rank
        end_row = start_row + tokens_per_rank

        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        y_sep = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        y_a2a = torch.zeros_like(y_sep)
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        dist.barrier()

        # ── Separate: GEMM (full M) + NCCL RS ──
        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            dist.reduce_scatter_tensor(y_sep, d_full, op=dist.ReduceOp.SUM, group=group)

        time_sep_ms = bench_fn(run_separate, num_iters=num_iters, group=group)

        # ── A2A PDL: GEMM (my rows only) × num_ranks ──
        my_a = a[start_row:end_row, :]
        def run_a2a_pdl():
            deep_gemm.bf16_gemm_nt(my_a, b, y_a2a)
            y_a2a.mul_(num_ranks)

        time_a2a_ms = bench_fn(run_a2a_pdl, num_iters=num_iters, group=group)

        sep_us = time_sep_ms * 1000
        a2a_us = time_a2a_ms * 1000
        speedup = sep_us / a2a_us if a2a_us > 0 else 0
        tf_sep = compute_tflops(total_m, n_dim, k_dim, time_sep_ms)
        tf_a2a = compute_tflops(total_m, n_dim, k_dim, time_a2a_ms)

        results.append({
            'tokens_per_rank': tokens_per_rank, 'n': n_dim, 'k': k_dim,
            'sep_us': sep_us, 'a2a_us': a2a_us, 'speedup': speedup,
            'tf_sep': tf_sep, 'tf_a2a': tf_a2a,
        })

        if rank_idx == 0:
            print(f"  {tokens_per_rank}x{n_dim}x{k_dim:<5} | {sep_us:>8.1f}u {a2a_us:>8.1f}u | "
                  f"{tf_sep:>8.1f}T {tf_a2a:>8.1f}T | {speedup:>9.2f}x")

        dist.barrier()

    if rank_idx == 0 and results:
        print(f"\n{'='*100}")
        speeds = [r['speedup'] for r in results if r['speedup'] > 0]
        geo = math.exp(sum(math.log(s) for s in speeds) / len(speeds)) if speeds else 0
        print(f"  Geometric Mean Speedup (A2A PDL vs Separate GEMM+NCCL RS): {geo:.3f}x")
        avg_tf_sep = sum(r['tf_sep'] for r in results) / len(results)
        avg_tf_a2a = sum(r['tf_a2a'] for r in results) / len(results)
        print(f"  Average TFLOPS: Separate={avg_tf_sep:.1f} | A2A PDL={avg_tf_a2a:.1f}")
        print(f"{'='*100}\n")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
