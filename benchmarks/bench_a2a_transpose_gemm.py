"""
Benchmark: Ulysses SP post-attention A2A-transpose + Wo GEMM.

Compares the fused op (comm overlapped with the Wo GEMM via per-M-tile barrier) against a
separate baseline (transpose-scatter comm, then a standard bf16 GEMM). Reports per-call GPU time.

Usage: python benchmarks/bench_a2a_transpose_gemm.py <num_gpus> [num_iters]
"""

import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm import _C
from deep_gemm.a2a_transpose_gemm import (
    get_symm_buffer_for_a2a_transpose_gemm, bf16_a2a_transpose_gemm_nt)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# (bs, nheads, seq, head_dim, N) — Ulysses post-attn; hidden = nheads*head_dim
SHAPES = [
    (1, 32, 2048, 128, 4096),
    (1, 56, 2048, 128, 7168),
    # larger-M (more M-tiles -> more pipelining headroom)
    (8, 32, 2048, 128, 4096),    # M = 8*256 = 2048
    (4, 32, 8192, 128, 4096),    # M = 4*1024 = 4096
    (1, 32, 16384, 128, 4096),   # seq=16384 -> local_seq=2048, M=2048
    (8, 56, 4096, 128, 7168),    # M = 8*512 = 4096, hidden=7168
]


def sp_sync(group):
    torch.cuda.synchronize()
    dist.barrier(group)


def run_benchmark(rank, num_gpus, num_iters, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(num_gpus)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)
    group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')
    sp = num_gpus

    if rank == 0:
        print(f"\n{'='*96}\n  A2A-transpose + Wo GEMM bench: {num_gpus} GPUs, {num_iters} iters\n{'='*96}")
        # comm/gemm are measured at FULL SMs (the realistic non-overlap deployment); serial = their
        # sum = the fair baseline. fused uses the SM carveout (DG_A2AT_COMM_SMS) + 1024-thread comm.
        print(f"{'(bs,nh,seq,hd,N)':<24} | {'comm(us)':>9} {'gemm(us)':>9} {'serial':>9} {'fused(us)':>10} | {'fus/ser':>8}")

    def time_call(fn, resets):
        for _ in range(3):
            for r in resets: r()
            sp_sync(group)
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        total = 0.0
        for _ in range(num_iters):
            for r in resets: r()
            sp_sync(group)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            total += s.elapsed_time(e)
        return total / num_iters * 1000.0  # us

    for (bs, nheads, seq, head_dim, N) in SHAPES:
        if nheads % sp or seq % sp or (seq // sp) % 128:
            if rank == 0:
                print(f"  ({bs},{nheads},{seq},{head_dim},{N}) SKIP")
            dist.barrier(); continue
        local_seq = seq // sp
        hidden = nheads * head_dim
        g = torch.Generator(device=device).manual_seed(1234)
        Wo = torch.randn((N, hidden), dtype=torch.bfloat16, device=device, generator=g)
        x_r = torch.randn((bs, nheads // sp, seq, head_dim), dtype=torch.bfloat16, device=device)

        sym = get_symm_buffer_for_a2a_transpose_gemm(group, bs, nheads, seq, head_dim)
        sym.x.copy_(x_r)
        d = torch.zeros((bs * local_seq, N), dtype=torch.bfloat16, device=device)

        rank_i = group.rank()
        ptrs = sym.handle.buffer_ptrs
        # comm-only (no host barrier): pure transpose-scatter kernel time
        t_comm = time_call(lambda: _C.bf16_a2a_transpose_comm(
            sym.buffer, ptrs, rank_i, bs, nheads, seq, head_dim), resets=[])
        # gemm-only: standard GEMM on the gathered buffer
        t_gemm = time_call(lambda: deep_gemm.bf16_gemm_nt(sym.gathered, Wo, d), resets=[])
        # fused (M1): comm overlapped with GEMM (per-M-tile barrier); reset barriers each iter
        t_fused = time_call(lambda: bf16_a2a_transpose_gemm_nt(d, Wo, sym),
                            resets=[sym.reset_barriers])

        if rank == 0:
            t_sum = t_comm + t_gemm
            sp_ratio = t_sum / t_fused if t_fused > 0 else 0.0
            # flux convention: exposed comm = fused_total - gemm_only. If overlap worked, exposed
            # should be << the real comm (t_comm). hidden = how much comm got hidden vs full comm.
            exposed = t_fused - t_gemm
            hidden_pct = (t_comm - exposed) / t_comm * 100.0 if t_comm > 0 else 0.0
            print(f"  ({bs},{nheads},{seq},{head_dim},{N})".ljust(24) +
                  f" | {t_comm:>9.1f} {t_gemm:>9.1f} {t_sum:>9.1f} {t_fused:>10.1f} | {sp_ratio:>6.2f}x"
                  f" | exposed={exposed:>6.1f} hidden={hidden_pct:>5.0f}%")
        sym.destroy()
        dist.barrier()

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    port = find_free_port()
    print(f"Launching A2A-transpose bench with {num_gpus} GPUs, {num_iters} iters...")
    mp.spawn(run_benchmark, args=(num_gpus, num_iters, port), nprocs=num_gpus, join=True)
