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
    get_symm_buffer_for_a2a_transpose_gemm,
    bf16_a2a_transpose_gemm_nt, bf16_a2a_transpose_gemm_nt_fused)


def torch_a2a_transpose(x_r, sp, group):
    """flux-style baseline comm from BHSD (torch-SDPA-native) attention output: permute+contiguous
    + NCCL all_to_all_single + permute/reshape. x_r: [bs, local_nh, seq, hd] -> [bs*local_seq, hidden]."""
    bs, local_nh, seq, hd = x_r.shape
    local_seq = seq // sp
    send = x_r.view(bs, local_nh, sp, local_seq, hd).permute(2, 0, 3, 1, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.permute(1, 2, 0, 3, 4).reshape(bs * local_seq, local_nh * sp * hd).contiguous()


def torch_a2a_transpose_bshd(x_bshd, sp, group):
    """FAIR baseline from BSHD (FlashAttention-native) attention output. Mirrors torch_a2a_transpose
    but the input is seq-major [bs, seq, local_nh, hd], so the baseline ALSO gets whatever transpose
    savings FA's layout provides (no gratuitous permute charged to one side only)."""
    bs, seq, local_nh, hd = x_bshd.shape
    local_seq = seq // sp
    send = x_bshd.view(bs, sp, local_seq, local_nh, hd).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.permute(1, 2, 0, 3, 4).reshape(bs * local_seq, sp * local_nh * hd).contiguous()


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
        print(f"{'(bs,nh,seq,hd,N)':<24} | {'torch comm/tot':>13} | {'ours comm/gemm M0 fused':>30} | {'M0/torch':>8}")

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
        # our comm-only (rotated, all SMs): pure transpose-scatter kernel time
        t_comm = time_call(lambda: _C.bf16_a2a_transpose_comm(
            sym.buffer, ptrs, rank_i, bs, nheads, seq, head_dim), resets=[])
        # gemm-only: standard GEMM on the gathered buffer
        t_gemm = time_call(lambda: deep_gemm.bf16_gemm_nt(sym.gathered, Wo, d), resets=[])
        # M0 (default): our comm + GEMM serial == t_comm + t_gemm (our strong baseline)
        t_m0 = t_comm + t_gemm
        Wo_t = Wo.t().contiguous()

        # --- FAIR FlashAttention(BSHD) world: BOTH our op and the torch baseline consume FA's native
        # BSHD output (no gratuitous permute charged to one side). FA does the seq<->head transpose
        # INSIDE its kernel (BSHD in/out), exactly as our comm absorbs it in-kernel. So a well-matched
        # pipeline pays NO external permute on either side; seq_major just gives our op the BSHD-matching
        # variant. This measures whether FA's BSHD layout changes the our-op-vs-baseline ratio at all.
        x_bshd = torch.randn((bs, seq, nheads // sp, head_dim), dtype=torch.bfloat16, device=device)
        t_comm_bshd = time_call(lambda: _C.bf16_a2a_transpose_comm(
            sym.buffer, ptrs, rank_i, bs, nheads, seq, head_dim, True), resets=[])  # seq_major=True
        t_m0_bshd = t_comm_bshd + t_gemm
        t_torch_bshd = time_call(lambda: torch.matmul(torch_a2a_transpose_bshd(x_bshd, sp, group), Wo_t),
                                 resets=[])
        # M1 fused (opt-in): comm overlapped with GEMM (per-M-tile barrier); reset barriers each iter
        t_fused = time_call(lambda: bf16_a2a_transpose_gemm_nt_fused(d, Wo, sym),
                            resets=[sym.reset_barriers])
        # flux-style baseline (BHSD / torch-SDPA world): torch transpose-a2a + torch.matmul, serial
        t_torch_comm = time_call(lambda: torch_a2a_transpose(x_r, sp, group), resets=[])
        def torch_total():
            a = torch_a2a_transpose(x_r, sp, group)
            return torch.matmul(a, Wo_t)
        t_torch = time_call(torch_total, resets=[])

        if rank == 0:
            vs_torch = t_torch / t_m0 if t_m0 > 0 else 0.0               # BHSD world: M0 vs torch
            vs_torch_bshd = t_torch_bshd / t_m0_bshd if t_m0_bshd > 0 else 0.0  # BSHD/FA world: M0 vs torch
            comm_x = t_torch_comm / t_comm if t_comm > 0 else 0.0
            print(f"  ({bs},{nheads},{seq},{head_dim},{N})".ljust(24) +
                  f" | torch:{t_torch_comm:>6.0f}/{t_torch:>6.0f} | ours:{t_comm:>6.1f}/{t_gemm:>6.1f}"
                  f" M0={t_m0:>6.1f} fused={t_fused:>6.1f} | M0/torch={vs_torch:>4.2f}x comm{comm_x:>4.1f}x")
            print(f"  {'  └─ FA(BSHD) world:':<24}"
                  f" torch={t_torch_bshd:>6.1f} our-M0(seq_major)={t_m0_bshd:>6.1f} | M0/torch={vs_torch_bshd:>4.2f}x"
                  f"   [both consume BSHD, no gratuitous permute]")
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
