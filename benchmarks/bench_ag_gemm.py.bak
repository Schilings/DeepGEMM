"""
Performance benchmark for BF16 All-Gather + GEMM fusion.

Compares:
  1. Separate: torch.distributed.all_gather + deep_gemm.bf16_gemm_nt
  2. Fused:    deep_gemm.bf16_ag_gemm_nt

Usage:
    python benchmarks/bench_ag_gemm.py [num_gpus] [num_iters]
    python benchmarks/bench_ag_gemm.py 2 10
"""

import math
import os
import sys
import time
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.ag_gemm import bf16_ag_gemm_nt, get_symm_buffer_for_bf16_ag_gemm


SHAPES = [
    # Medium sequence length (4K~8K per rank, N=4096/7168, K=4096/7168)
    (4096,  4096, 4096),
    (4096,  7168, 4096),
    (4096,  7168, 7168),
    (6144,  4096, 4096),
    (6144,  7168, 4096),
    (6144,  7168, 7168),
    (8192,  4096, 4096),
    (8192,  7168, 4096),
    (8192,  7168, 7168),
    # Large shapes (10K~20K per rank)
    (10240, 7168, 4096),
    (10240, 7168, 7168),
    (12288, 7168, 4096),
    (12288, 7168, 7168),
    (16384, 7168, 4096),
    (16384, 7168, 7168),
    (20480, 7168, 4096),
    (20480, 7168, 7168),
]


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


def _bench_ms(fn, num_iters: int, group) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    dist.barrier(group)
    return start.elapsed_time(end) / num_iters


def _worker(rank: int, num_gpus: int, num_iters: int, port: int):
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

    torch.manual_seed(20260612 + rank)
    torch.cuda.manual_seed(20260612 + rank)

    if rank == 0:
        print(f"\n{'=' * 100}")
        print(f"  BF16 AG+GEMM Benchmark: {num_gpus} GPUs, {num_iters} iterations")
        print(f"{'=' * 100}")
        print(f"  {'Shape':<22} | {'Separate':>10} {'Fused':>10} | {'Sep TFLOPS':>10} {'Fus TFLOPS':>10} | {'Speedup':>8}")
        print(f"  {'(M/rank×N×K)':<22} | {'(μs)':>10} {'(μs)':>10} | {'':>10} {'':>10} |")
        print(f"  {'─' * 22}┼{'─' * 22}┼{'─' * 22}┼{'─' * 9}")

    results = []
    for tokens_per_rank, n_dim, k_dim in SHAPES:
        total_m = tokens_per_rank * num_gpus
        x_local = torch.randn((tokens_per_rank, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        dist.broadcast(b, src=0, group=group)

        recv_list = [torch.empty_like(x_local) for _ in range(num_gpus)]
        d_sep = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        def run_separate():
            dist.all_gather(recv_list, x_local, group=group)
            x_full = torch.cat(recv_list, dim=0)
            deep_gemm.bf16_gemm_nt(x_full, b, d_sep)

        try:
            sym_buffer = get_symm_buffer_for_bf16_ag_gemm(group, tokens_per_rank, k_dim)
        except Exception as exc:
            if rank == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<6} | SKIP (buffer alloc failed: {exc})")
            dist.barrier(group)
            continue

        d_fused = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)

        def run_fused():
            sym_buffer.x[:tokens_per_rank, :k_dim].copy_(x_local)
            bf16_ag_gemm_nt(d_fused, b, sym_buffer, tokens_per_rank)

        try:
            separate_ms = _bench_ms(run_separate, num_iters, group)
            fused_ms = _bench_ms(run_fused, num_iters, group)
        except Exception as exc:
            if rank == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<6} | SKIP ({type(exc).__name__}: {exc})")
            sym_buffer.destroy()
            dist.barrier(group)
            continue

        separate_us = separate_ms * 1000.0
        fused_us = fused_ms * 1000.0
        flops = 2.0 * total_m * n_dim * k_dim
        sep_tflops = flops / separate_us / 1e6
        fused_tflops = flops / fused_us / 1e6
        speedup = separate_us / fused_us

        if rank == 0:
            print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<6} | {separate_us:>9.1f}μs {fused_us:>9.1f}μs | {sep_tflops:>9.0f}T {fused_tflops:>9.0f}T | {speedup:>7.2f}x")
            results.append(speedup)

        sym_buffer.destroy()
        dist.barrier(group)

    if rank == 0 and results:
        geo_mean = math.exp(sum(math.log(v) for v in results) / len(results))
        print(f"\n{'=' * 100}")
        print(f"    Geometric Mean Speedup: {geo_mean:.3f}x")
        print(f"    Shapes where Fused wins (>1.0x): {sum(1 for v in results if v > 1.0)}/{len(results)}")
        print(f"{'=' * 100}\n")

    dist.destroy_process_group()
    time.sleep(1)
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    port = int(os.getenv('MASTER_PORT', '0')) or _find_free_port()
    os.environ['MASTER_PORT'] = str(port)
    print(f"Launching AG+GEMM benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(_worker, args=(num_gpus, num_iters, port), nprocs=num_gpus, join=True)
