"""
Performance benchmark for BF16 All2All + GEMM fusion (Ulysses SP scenario).

Compares: Fused A2A+GEMM vs Separate (all_to_all + bf16_gemm_nt)

Usage:
    python benchmarks/bench_a2a_gemm.py <num_gpus> [num_iters]
    python benchmarks/bench_a2a_gemm.py 8 20
"""

import os, sys, math, socket, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.a2a_gemm import bf16_a2a_gemm_nt, get_symm_buffer_for_a2a_gemm


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# Ulysses SP shapes: (M_per_rank, N, K)
# M_per_rank = sequence length per rank (after A2A: each rank gets all tokens)
# K = heads_per_tp * head_dim
# N = output hidden dim
SHAPES = [
    (1024, 4096, 4096),
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


def run_benchmark(rank, num_gpus, num_iters, port):
    os.environ.update({
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
        'RANK': str(rank),
        'WORLD_SIZE': str(num_gpus),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)
    group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*100}")
        print(f"  BF16 A2A+GEMM Benchmark: {num_gpus} GPUs, {num_iters} iterations")
        print(f"{'='*100}")
        print(f"  {'Shape':<22} | {'Separate':>10} {'Fused':>10} | {'Sep TFLOPS':>10} {'Fus TFLOPS':>10} | {'Speedup':>8}")
        print(f"  {'(M/rank×N×K)':<22} | {'(μs)':>10} {'(μs)':>10} |{' ':>10} {' ':>10} |")
        print(f"  {'─'*22}┼{'─'*22}┼{'─'*22}┼{'─'*9}")

    results = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_gpus
        x_full = torch.randn((num_gpus, tokens_per_rank, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)

        # === Separate: all_to_all + GEMM ===
        send_list = list(x_full.unbind(0))
        recv_list = [torch.empty_like(send_list[0]) for _ in range(num_gpus)]
        d_sep = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            dist.all_to_all(recv_list, send_list, group=group)
            a_gathered = torch.cat(recv_list, dim=0)
            deep_gemm.bf16_gemm_nt(a_gathered, b, d_sep)
        torch.cuda.synchronize()
        dist.barrier()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(num_iters):
            dist.all_to_all(recv_list, send_list, group=group)
            a_gathered = torch.cat(recv_list, dim=0)
            deep_gemm.bf16_gemm_nt(a_gathered, b, d_sep)
        e.record()
        torch.cuda.synchronize()
        sep_us = s.elapsed_time(e) / num_iters * 1000

        # === Fused: A2A+GEMM ===
        try:
            sym_buffer = get_symm_buffer_for_a2a_gemm(group, tokens_per_rank, k_dim)
        except Exception as ex:
            if rank == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<6} | SKIP ({ex})")
            dist.barrier()
            continue

        d_fused = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            sym_buffer.x[:num_gpus, :tokens_per_rank, :k_dim].copy_(x_full)
            bf16_a2a_gemm_nt(d_fused, b, sym_buffer, tokens_per_rank)
        torch.cuda.synchronize()
        dist.barrier()

        s.record()
        for _ in range(num_iters):
            sym_buffer.x[:num_gpus, :tokens_per_rank, :k_dim].copy_(x_full)
            bf16_a2a_gemm_nt(d_fused, b, sym_buffer, tokens_per_rank)
        e.record()
        torch.cuda.synchronize()
        fused_us = s.elapsed_time(e) / num_iters * 1000

        flops = 2.0 * total_m * n_dim * k_dim
        sep_tflops = flops / sep_us / 1e6
        fused_tflops = flops / fused_us / 1e6
        speedup = sep_us / fused_us

        if rank == 0:
            print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<6} | {sep_us:>9.1f}μs {fused_us:>9.1f}μs | {sep_tflops:>9.0f}T {fused_tflops:>9.0f}T | {speedup:>7.2f}x")
            results.append({
                'tokens_per_rank': tokens_per_rank,
                'n_dim': n_dim, 'k_dim': k_dim,
                'time_separate_us': sep_us, 'time_fused_us': fused_us,
                'speedup': speedup,
            })

        sym_buffer.destroy()
        dist.barrier()

    if rank == 0 and results:
        speedups = [r['speedup'] for r in results]
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\n{'='*100}")
        print(f"    Geometric Mean Speedup: {geo_mean:.3f}x")
        print(f"    Shapes where Fused wins (>1.0x): {sum(1 for s in speedups if s > 1.0)}/{len(speedups)}")
        sorted_r = sorted(results, key=lambda r: r['speedup'], reverse=True)
        print(f"\n  Top 5 Shapes:")
        for i, r in enumerate(sorted_r[:5]):
            print(f"    {i+1}. {r['tokens_per_rank']}×{r['n_dim']}×{r['k_dim']}  {r['speedup']:.2f}x")
        print(f"\n{'='*100}\n")

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    port = find_free_port()
    print(f"Launching A2A+GEMM benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters, port), nprocs=num_gpus, join=True)
