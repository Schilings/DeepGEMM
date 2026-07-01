"""
Benchmark for Fused QKV GEMM + RMSNorm + A2A-transpose (Ulysses SP pre-attn).

Compares:
  1. serial: separate torch.matmul + RMSNorm + NCCL all_to_all (Q/K/V separate)
  2. fused: bf16_fused_qkv_norm_a2a_transpose_nt (Phase 2 v1: Python-orchestrated)

Usage:
    python benchmarks/bench_fused_qkv_norm_a2a.py [num_gpus] [iters]
    python benchmarks/bench_fused_qkv_norm_a2a.py 8 30
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


# Wan2.1 14B + Megatron SP shapes: (bs, local_seq, q_nheads, kv_nheads, head_dim, K)
SHAPES_FOCUS = [
    # Wan2.1 14B: dim=5120, q=40, kv=8, hd=128
    (1, 1024, 40, 8, 128, 5120),    # 1x8K seq
    (1, 2048, 40, 8, 128, 5120),    # 1x16K seq
    (1, 4096, 40, 8, 128, 5120),    # 1x32K seq
    (1, 8192, 40, 8, 128, 5120),    # 1x64K seq
    # MHA variants
    (1, 1024, 32, 32, 128, 4096),
    (1, 2048, 64, 64, 128, 8192),
    (1, 4096, 32, 32, 128, 5120),
    (2, 1024, 40, 8, 128, 5120),    # BSHD bs=2
]


def rmsnorm_ref(x, weight, eps):
    xf = x.float()
    norm = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * norm).to(x.dtype) * weight.to(x.dtype)


def serial_pre_attn(a, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                    num_ranks, group, norm_q_weight, norm_k_weight, eps, bias):
    """Serial baseline: separate GEMM + norm + NCCL A2A."""
    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim
    seq = local_seq * num_ranks

    local_q_n = (q_nheads // num_ranks) * head_dim
    local_kv_n = (kv_nheads // num_ranks) * head_dim
    local_n = local_q_n + 2 * local_kv_n

    # GEMM
    d = torch.matmul(a, b.t())
    if bias is not None:
        d = d + bias
    d = d.view(bs, local_seq, n_total)

    # Split + norm
    q = d[:, :, :q_dim]
    k = d[:, :, q_dim:q_dim + kv_dim]
    v = d[:, :, q_dim + kv_dim:]

    if norm_q_weight is not None:
        q = rmsnorm_ref(q, norm_q_weight, eps)
    if norm_k_weight is not None:
        k = rmsnorm_ref(k, norm_k_weight, eps)

    # A2A per Q/K/V
    q_send = q.view(bs, local_seq, num_ranks, local_q_n).permute(2, 0, 1, 3).contiguous()
    k_send = k.view(bs, local_seq, num_ranks, local_kv_n).permute(2, 0, 1, 3).contiguous()
    v_send = v.view(bs, local_seq, num_ranks, local_kv_n).permute(2, 0, 1, 3).contiguous()

    send = torch.cat([q_send, k_send, v_send], dim=-1)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    out = recv.permute(1, 0, 2, 3).reshape(bs, seq, local_n).contiguous()
    return out


def time_call(fn, iters=30):
    """Time a function with CUDA synchronization."""
    torch.cuda.synchronize()
    # Warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters * 1e6  # us


def run_bench(local_rank, num_local_ranks, iters=30):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'
    eps = 1e-6

    if rank_idx == 0:
        print(f"\n{'='*130}")
        print(f"  Fused QKV GEMM + RMSNorm + A2A-transpose Benchmark: {num_ranks} GPUs")
        print(f"  Phase 2 v1: Python-orchestrated (serial vs fused)")
        print(f"{'='*130}\n")
        print(f"{'Shape (bs,lseq,qh,kvh,hd,K)':<32} | {'Mode':<8} | "
              f"{'Serial (us)':>12} {'Fused (us)':>12} {'Speedup':>8} | {'Status'}")
        print(f"{'-'*32} | {'-'*8} | {'-'*12} {'-'*12} {'-'*8} | {'-'*8}")

    for bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim in SHAPES_FOCUS:
        if q_nheads % num_ranks != 0 or kv_nheads % num_ranks != 0:
            if rank_idx == 0:
                print(f"  SKIP: head not divisible by {num_ranks}")
            dist.barrier()
            continue

        seq = local_seq * num_ranks
        q_dim = q_nheads * head_dim
        kv_dim = kv_nheads * head_dim
        n_total = q_dim + 2 * kv_dim
        local_m = bs * local_seq

        if local_seq % 128 != 0:
            if rank_idx == 0:
                print(f"  SKIP: local_seq {local_seq} not 128-aligned")
            dist.barrier()
            continue

        # Data
        x = torch.randn((local_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_total, k_dim), dtype=torch.bfloat16, device=device)
        bias = torch.randn((n_total,), dtype=torch.bfloat16, device=device)
        norm_q = torch.randn((q_dim,), dtype=torch.float32, device=device).normal_(1.0, 0.1)
        norm_k = torch.randn((kv_dim,), dtype=torch.float32, device=device).normal_(1.0, 0.1)

        dist.barrier()

        # Serial baseline
        def serial_fn():
            serial_pre_attn(x, b, bs, local_seq, q_nheads, kv_nheads, head_dim, k_dim,
                           num_ranks, group, norm_q, norm_k, eps, bias)

        t_serial = time_call(serial_fn, iters)

        # Fused
        sym_buffer = deep_gemm.get_symm_buffer_for_fused_qkv_norm_a2a(
            group, bs, seq, q_nheads, kv_nheads, head_dim)

        def fused_fn():
            deep_gemm.bf16_fused_qkv_norm_a2a_transpose_nt(
                x, b, sym_buffer, local_seq,
                q_nheads, kv_nheads, head_dim,
                eps=eps, norm_q_weight=norm_q, norm_k_weight=norm_k, bias=bias)

        t_fused = time_call(fused_fn, iters)
        sym_buffer.destroy()

        speedup = t_serial / t_fused if t_fused > 0 else 0

        if rank_idx == 0:
            shape_str = f"{bs},{local_seq},{q_nheads},{kv_nheads},{head_dim},{k_dim}"
            print(f"{shape_str:<32} | {'norm' if norm_q is not None else 'noNorm':<8} | "
                  f"{t_serial:>12.1f} {t_fused:>12.1f} {speedup:>7.2f}x | "
                  f"{'fused' if speedup > 1 else 'serial wins'}")

        dist.barrier()

    if rank_idx == 0:
        print(f"\n{'='*130}\n")

    dist.barrier()
    dist.destroy_process_group()
    time.sleep(1)
    os._exit(0)


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if os.getenv('MASTER_PORT') is None or os.getenv('MASTER_PORT') == '':
        os.environ['MASTER_PORT'] = str(_find_free_port())
    print(f"Launching benchmark with {num_gpus} GPUs, {iters} iters...")
    mp.spawn(run_bench, args=(num_gpus, iters), nprocs=num_gpus, join=True)
