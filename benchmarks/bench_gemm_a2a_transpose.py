"""
pre-attn GEMM + A2A-transpose Benchmark: Fused vs Torch-Native vs DeepGEMM-Separate (Ulysses SP).

Compares three paths (all produce the same [bs, seq, local_n] head-scattered / seq-gathered output):
  1. torch.matmul(x, W.t())  + all_to_all_single   -- torch-native baseline
  2. deep_gemm.bf16_gemm_nt  + all_to_all_single   -- deepgemm-separate baseline
  3. bf16_gemm_a2a_transpose_nt (fused single kernel)

Shape format: (bs, local_seq, nheads, head_dim, K). N = nheads*head_dim, seq = local_seq*num_ranks.

Usage:
  python benchmarks/bench_gemm_a2a_transpose.py [num_gpus] [num_iters]

Optional environment variables:
  DG_BENCH_FOCUS_ONLY=1           # run the focus subset only
  DG_BENCH_SYNC_EACH_ITER=1       # synchronize each iteration for diagnostics
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


# (bs, local_seq, nheads, head_dim, K)
SHAPES_STANDARD = [
    (1, 1024, 32, 128, 4096),
    (1, 2048, 64, 128, 8192),
    (1, 4096, 32, 128, 7168),
    (1, 4096, 64, 128, 8192),
    (1, 8192, 64, 128, 8192),
    (2, 1024, 64, 128, 8192),
    (2, 2048, 32, 128, 4096),
    (2, 2048, 64, 128, 8192),
    (4, 1024, 64, 128, 4096),
]

SHAPES_FOCUS = [
    (1, 4096, 64, 128, 8192),
    (1, 8192, 64, 128, 8192),
    (2, 2048, 64, 128, 8192),
]


def get_shapes_to_run():
    if bool(int(os.getenv("DG_BENCH_FOCUS_ONLY", "0"))):
        return SHAPES_FOCUS
    return SHAPES_STANDARD


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
        print(f"\n{'=' * 140}")
        print(f"  GEMM+A2A-transpose Benchmark (Fused vs Torch-Native/DeepGEMM-Separate), GPUs={num_ranks}, iters={num_iters}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'=' * 140}\n")
        print(f"  {'Shape (bs,lseq,h,hd,K)':<26} | {'Torch':>10} {'Separate':>10} {'Fused':>10} | "
              f"{'Fused TFLOPS':>12} | {'vs Torch':>9} {'vs Sep':>9}")
        print(f"  {'-' * 26}-+-{'-' * 10}-{'-' * 10}-{'-' * 10}-+-{'-' * 12}-+-{'-' * 9}-{'-' * 9}")

    results = []

    for bs, local_seq, nheads, head_dim, k_dim in shapes_to_run:
        if nheads % num_ranks != 0:
            if rank_idx == 0:
                print(f"  {bs},{local_seq},{nheads},{head_dim},{k_dim}: SKIP (nheads % num_ranks != 0)")
            dist.barrier(group)
            continue

        n = nheads * head_dim
        seq = local_seq * num_ranks
        local_n = n // num_ranks
        local_m = bs * local_seq

        a = torch.randn((local_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n, k_dim), dtype=torch.bfloat16, device=device)
        d_full = torch.empty((local_m, n), dtype=torch.bfloat16, device=device)

        try:
            sym_buffer = deep_gemm.get_symm_buffer_for_gemm_a2a_transpose(
                group, bs, seq, n, out_dtype=torch.bfloat16)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {bs},{local_seq},{nheads},{head_dim},{k_dim}: SKIP (symm alloc failed: {e})")
            dist.barrier(group)
            continue

        send = torch.empty((num_ranks, bs, local_seq, local_n), dtype=torch.bfloat16, device=device)
        recv = torch.empty_like(send)

        dist.barrier(group)

        def run_torch_native():
            d = torch.matmul(a, b.t()).view(bs, local_seq, num_ranks, local_n)
            send.copy_(d.permute(2, 0, 1, 3))
            dist.all_to_all_single(recv, send, group=group)

        def run_separate():
            deep_gemm.bf16_gemm_nt(a, b, d_full)
            d = d_full.view(bs, local_seq, num_ranks, local_n)
            send.copy_(d.permute(2, 0, 1, 3))
            dist.all_to_all_single(recv, send, group=group)

        def run_fused():
            deep_gemm.bf16_gemm_a2a_transpose_nt(a, b, sym_buffer, local_seq, compiled_dims="nk")

        try:
            t_torch = bench_fn(run_torch_native, num_iters=num_iters, barrier_group=group)
            t_sep = bench_fn(run_separate, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            if rank_idx == 0:
                print(f"  {bs},{local_seq},{nheads},{head_dim},{k_dim}: SKIP (baseline failed: {e})")
            try:
                sym_buffer.destroy()
            finally:
                dist.barrier(group)
            continue

        try:
            t_fused = bench_fn(run_fused, num_iters=num_iters, barrier_group=group)
        except Exception as e:
            t_fused = float("inf")
            if rank_idx == 0:
                print(f"  fused failed for {bs},{local_seq},{nheads},{head_dim},{k_dim}: {e}")

        us = lambda t: t * 1000.0
        t_torch_us, t_sep_us = us(t_torch), us(t_sep)
        t_fused_us = us(t_fused) if t_fused != float("inf") else float("inf")
        sp_torch = t_torch_us / t_fused_us if t_fused_us not in (0, float("inf")) else 0.0
        sp_sep = t_sep_us / t_fused_us if t_fused_us not in (0, float("inf")) else 0.0
        tflops_fused = compute_tflops(local_m, n, k_dim, t_fused) if t_fused != float("inf") else 0.0

        results.append({
            "shape": (bs, local_seq, nheads, head_dim, k_dim),
            "n": n, "k": k_dim, "local_seq": local_seq,
            "sp_torch": sp_torch, "sp_sep": sp_sep, "tflops_fused": tflops_fused,
        })

        if rank_idx == 0:
            shape_str = f"{bs},{local_seq},{nheads},{head_dim},{k_dim}"
            fused_str = f"{t_fused_us:>8.1f}" if t_fused_us != float("inf") else "    FAIL"
            ft_str = f"{tflops_fused:>8.1f}T" if tflops_fused > 0 else "    FAIL"
            sp_t_str = f"{sp_torch:.2f}x" if sp_torch > 0 else "FAIL"
            sp_s_str = f"{sp_sep:.2f}x" if sp_sep > 0 else "FAIL"
            print(f"  {shape_str:<26} | {t_torch_us:>8.1f}u {t_sep_us:>8.1f}u {fused_str}u | "
                  f"{ft_str:>12} | {sp_t_str:>9} {sp_s_str:>9}")

        try:
            torch.cuda.synchronize()
        except Exception as e:
            if rank_idx == 0:
                print(f"  CUDA sync failed for {bs},{local_seq},{nheads},{head_dim},{k_dim}: {e}")
        dist.barrier(group)
        try:
            sym_buffer.destroy()
        except Exception:
            pass
        dist.barrier(group)

    if rank_idx == 0 and results:
        print(f"\n{'=' * 140}")
        print(f"  Summary ({num_ranks} GPUs)")
        print(f"{'=' * 140}")
        sp_torch_list = [r["sp_torch"] for r in results if r["sp_torch"] > 0]
        sp_sep_list = [r["sp_sep"] for r in results if r["sp_sep"] > 0]
        geo = lambda xs: math.exp(sum(math.log(s) for s in xs) / len(xs)) if xs else 0.0
        print(f"    Fused geo_mean speedup vs torch-native:       {geo(sp_torch_list):.3f}x")
        print(f"    Fused geo_mean speedup vs deepgemm-separate:  {geo(sp_sep_list):.3f}x")
        if sp_torch_list:
            print(f"    Best/Worst vs torch-native: {max(sp_torch_list):.2f}x / {min(sp_torch_list):.2f}x")
        if sp_sep_list:
            print(f"    Best/Worst vs deepgemm-separate: {max(sp_sep_list):.2f}x / {min(sp_sep_list):.2f}x")
        fused_valid = [r["tflops_fused"] for r in results if r["tflops_fused"] > 0]
        print(f"    Fused avg TFLOPS: {sum(fused_valid)/len(fused_valid):.1f}" if fused_valid else "    Fused avg TFLOPS: N/A")
        print(f"\n{'=' * 140}\n")

    dist.barrier(group)
    dist.destroy_process_group()
    time.sleep(0.5)
    os._exit(0)


if __name__ == "__main__":
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 20
    print(f"Launching GEMM+A2A-transpose benchmark with {num_gpus} GPUs, {num_iters} iterations...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters), nprocs=num_gpus, join=True)
