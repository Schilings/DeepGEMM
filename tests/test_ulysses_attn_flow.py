"""
Full Ulysses SP attention flow, with correctness end-to-end and profiling of ONLY the post-attn
A2A-transpose + Wo GEMM (our op).

Flow per rank r (sp_size = world_size, sequence-parallel input):
  X_local[bs, local_seq, hidden]                       # rank r owns seq shard [r*local_seq:...]
  --QKV proj (local)-->        q,k,v [bs, local_seq, nheads, hd]
  --pre-attn A2A (transpose)-> q,k,v [bs, seq, local_nheads, hd]   # gather seq, scatter heads
  --attention (SDPA, per local head, full seq)--> attn[bs, local_nheads, seq, hd]  == our op's x_r
  --post-attn A2A-transpose + Wo GEMM (OUR OP)--> y[bs*local_seq, N]

Correctness: each rank also computes the WHOLE thing from the all-gathered global X (full QKV +
full attention + Wo) and slices its own seq shard; compare to the distributed result.
Only the post-attn A2A-transpose+Wo GEMM (M0 default and fused) is timed/profiled.

Usage: python tests/test_ulysses_attn_flow.py <num_gpus> [iters]
"""

import os, sys, socket, math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


# (bs, nheads, seq, head_dim, N)
SHAPES = [
    (1, 32, 2048, 128, 4096),
    (1, 56, 2048, 128, 7168),
    (8, 56, 4096, 128, 7168),
]


def pre_attn_a2a(t, sp, group):
    """[bs, local_seq, nheads, hd] -> [bs, seq, local_nheads, hd] (gather seq, scatter heads)."""
    bs, local_seq, nheads, hd = t.shape
    local_nh = nheads // sp
    send = [t[:, :, d * local_nh:(d + 1) * local_nh, :].contiguous() for d in range(sp)]
    recv = [torch.empty_like(send[0]) for _ in range(sp)]
    dist.all_to_all(recv, send, group=group)      # recv[s]: src s's seq shard, OUR heads
    return torch.cat(recv, dim=1)                  # [bs, seq, local_nh, hd]


def sdpa(q, k, v, hd):
    """q,k,v [bs, H, seq, hd] -> [bs, H, seq, hd]."""
    return F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hd))


def run(rank, ng, port, iters):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}'); sp = ng

    import deep_gemm
    from deep_gemm.a2a_transpose_gemm import (
        get_symm_buffer_for_a2a_transpose_gemm,
        bf16_a2a_transpose_gemm_nt, bf16_a2a_transpose_gemm_nt_fused)

    if rank == 0:
        print(f"\n{'='*94}\n  Full Ulysses SP attn flow: {ng} GPUs (correctness end-to-end + post-attn-only timing)\n{'='*94}")
        print(f"{'(bs,nh,seq,hd,N)':<24} | {'e2e status':>12} | {'post M0(us)':>11} {'post fused(us)':>14}")

    def time_call(fn, resets, it):
        for _ in range(3):
            for r in resets: r()
            torch.cuda.synchronize(); dist.barrier(group); fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        tot = 0.0
        for _ in range(it):
            for r in resets: r()
            torch.cuda.synchronize(); dist.barrier(group)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            tot += s.elapsed_time(e)
        return tot / it * 1000.0

    num_pass = 0; fails = []
    for (bs, nheads, seq, head_dim, N) in SHAPES:
        if nheads % sp or seq % sp or (seq // sp) % 128:
            if rank == 0: print(f"  ({bs},{nheads},{seq},{head_dim},{N}) SKIP")
            dist.barrier(); continue
        local_nh = nheads // sp; local_seq = seq // sp; hidden = nheads * head_dim
        g = torch.Generator(device=dev).manual_seed(42)
        # shared weights across ranks (so the global reference is well-defined)
        Wq = (torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden))
        Wk = (torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden))
        Wv = (torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden))
        Wo = (torch.randn((N, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden))
        # this rank's seq shard of the global input
        X_local = torch.randn((bs, local_seq, hidden), dtype=torch.bfloat16, device=dev)

        # ---- distributed Ulysses flow ----
        def qkv(x):  # x [bs, S, hidden] -> [bs, S, nheads, hd]
            return ((x @ Wq).view(bs, -1, nheads, head_dim),
                    (x @ Wk).view(bs, -1, nheads, head_dim),
                    (x @ Wv).view(bs, -1, nheads, head_dim))
        q, k, v = qkv(X_local)                                  # [bs, local_seq, nheads, hd]
        qf = pre_attn_a2a(q, sp, group)                         # [bs, seq, local_nh, hd]
        kf = pre_attn_a2a(k, sp, group)
        vf = pre_attn_a2a(v, sp, group)
        attn = sdpa(qf.permute(0, 2, 1, 3), kf.permute(0, 2, 1, 3),
                    vf.permute(0, 2, 1, 3), head_dim)           # [bs, local_nh, seq, hd] == x_r
        attn = attn.contiguous()

        # ---- post-attn A2A-transpose + Wo GEMM (OUR OP) ----
        sym = get_symm_buffer_for_a2a_transpose_gemm(group, bs, nheads, seq, head_dim)
        sym.x.copy_(attn)
        y = torch.zeros((bs * local_seq, N), dtype=torch.bfloat16, device=dev)
        bf16_a2a_transpose_gemm_nt(y, Wo, sym)                  # default M0
        torch.cuda.synchronize()

        # ---- single-process reference (each rank recomputes the whole flow globally) ----
        xs = [torch.empty_like(X_local) for _ in range(sp)]
        dist.all_gather(xs, X_local, group=group)
        Xg = torch.cat(xs, dim=1)                               # [bs, seq, hidden]
        qg = (Xg @ Wq).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        kg = (Xg @ Wk).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        vg = (Xg @ Wv).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        ag = sdpa(qg, kg, vg, head_dim)                         # [bs, nheads, seq, hd]
        ag = ag.permute(0, 2, 1, 3).reshape(bs, seq, hidden)    # [bs, seq, hidden]
        Yg = (ag.float() @ Wo.float().t())                      # [bs, seq, N]
        y_ref = Yg[:, rank * local_seq:(rank + 1) * local_seq, :].reshape(bs * local_seq, N)

        rel = (y.float() - y_ref).abs().mean().item() / (y_ref.abs().mean().item() + 1e-8)
        passed = rel < 0.03                                     # full bf16 attn+proj+gemm pipeline

        # ---- profile ONLY the post-attn a2a-gemm (M0 default + fused) ----
        t_m0 = time_call(lambda: bf16_a2a_transpose_gemm_nt(y, Wo, sym), resets=[], it=iters)
        t_fused = time_call(lambda: bf16_a2a_transpose_gemm_nt_fused(y, Wo, sym),
                            resets=[sym.reset_barriers], it=iters)
        if rank == 0:
            print(f"  ({bs},{nheads},{seq},{head_dim},{N})".ljust(24) +
                  f" | {('PASS' if passed else 'FAIL'):>12} | {t_m0:>11.1f} {t_fused:>14.1f}"
                  f"   (rel={rel:.2e})")
            num_pass += int(passed)
            if not passed: fails.append((bs, nheads, seq, head_dim, N))
        sym.destroy(); dist.barrier()

    if rank == 0:
        print(f"\n  e2e correctness: {num_pass}/{len(SHAPES)} passed" +
              ("  ALL PASS" if not fails else f"  FAILED: {fails}"))
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    port = find_free_port()
    print(f"Launching full Ulysses attn flow with {ng} GPUs...")
    mp.spawn(run, args=(ng, port, iters), nprocs=ng, join=True)
