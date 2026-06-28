"""
FULL Ulysses SP attention-chain benchmark: speedup of the two fused ops end-to-end vs a
torch-native (NCCL all_to_all + torch.matmul) baseline, across shapes, for BSHD and THD input.

Chain per rank r (sp = world_size, sequence-parallel input):
  X_local[bs, local_seq, hidden]
    --[PRE]  fused QKV-proj GEMM + A2A-transpose --> q,k,v [bs, seq, local_nh, hd]
    --[ATTN] attention (FlashAttention-4) -------->  attn  [bs, local_nh, seq, hd]
    --[POST] A2A-transpose + Wo GEMM (overlapped)->  y     [bs*local_seq, N]

For each shape and layout we compare THREE chains:
  fused chain = PRE(ours)                 + ATTN + POST(ours, comm/GEMM overlapped)
  torch chain = PRE(matmul + a2a, serial) + ATTN + POST(a2a + matmul, serial)
  async chain = PRE(split-QKV multi-stream overlap) + ATTN + POST(token-chunk multi-stream overlap)
ATTN is identical on all paths (not optimized here), so we report speedups vs BOTH baselines:
  * e2e speedup        = (torch | async)-chain / fused-chain          (honest full-chain numbers)
  * comm+GEMM speedup  = (PRE+POST torch | async)/(PRE+POST ours)     (where the fused ops actually help)

ASYNC ULYSSES baseline (stronger, hand-rolled overlap): instead of the serial torch path (one big QKV
GEMM -> one full all_to_all), it splits the work and pipelines compute vs comm on >=2 CUDA streams:
  * PRE : split into Q/K/V -> 3 separate GEMM + 3 separate A2A; A2A(Q) overlaps the K GEMM, etc.
  * POST: split tokens into chunks -> per-chunk (transpose-scatter + A2A); chunk A2As overlap the
          Wo GEMM of already-arrived chunks.
This is the fair "manual multi-stream overlap" reference; our fused ops instead overlap inside a single
kernel (epilogue scatter), so the gap async->ours isolates the extra gain over hand-rolled overlap.

EQUIVALENCE (BSHD vs THD): a shape (bs, nheads, seq, hd) is run as
  * BSHD: `bs` sequences of length `seq`, batched   -> tokens = bs*seq.
  * THD : the SAME `bs` sequences packed into one stream T = bs*seq (bs'=1, seq'=T).
The fused comm/GEMM ops process the exact same bs*seq tokens either way and are called identically
(only the symm-buffer (bs, seq) descriptor differs), so the speedup must match -> demonstrates the
ops handle BSHD and THD equivalently. Attention FLOPs are identical for uniform lengths, so ATTN is
timed once per shape and shared by both layouts.

NORMAL TRAINING SCENARIO: weights are SQUARE. hidden = nheads*head_dim and N (Wo width) = hidden.
  * Wq / Wk / Wv : [hidden, hidden]      (square)
  * fused Wqkv   : [3*hidden, hidden]    (3 square blocks stacked, rank-major)
  * Wo           : [hidden, hidden]      (square, N = hidden)

Usage: python benchmarks/bench_ulysses_full_attn_flow.py <num_gpus> [iters]
"""

import os, sys, socket, math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


# (bs, nheads, seq, head_dim)  -- SQUARE weights: hidden = N = nheads*head_dim
SHAPES = [
    (1, 32, 4096, 128),   # hidden = 4096
    (1, 56, 4096, 128),   # hidden = 7168
    (2, 32, 4096, 128),   # hidden = 4096   (THD T = 8192)
    (2, 56, 2048, 128),   # hidden = 7168   (THD T = 4096)
    (1, 64, 8192, 128),   # hidden = 8192
]


def build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, hd):
    """Rank-major [Q,K,V] head-group blocks: rows[d*local_n:(d+1)*local_n] = [Q(d),K(d),V(d)]."""
    rows = local_nh * hd
    blocks = []
    for d in range(sp):
        sl = slice(d * rows, (d + 1) * rows)
        blocks += [Wq[sl], Wk[sl], Wv[sl]]
    return torch.cat(blocks, dim=0).contiguous()


def run(rank, ng, port, iters):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}'); sp = ng

    import deep_gemm
    from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose, bf16_gemm_a2a_transpose_nt
    from deep_gemm.a2a_transpose_gemm import (
        get_symm_buffer_for_a2a_transpose_gemm,
        bf16_a2a_transpose_gemm_nt, bf16_a2a_transpose_gemm_nt_fused)
    from flash_attn.cute import flash_attn_func as fa4_func   # FlashAttention-4 (see docs/INSTALL_FA4.md)

    def time_call(fn, it, resets=()):
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
        return tot / it * 1000.0  # us

    if rank == 0:
        print(f"\n{'='*132}")
        print(f"  Ulysses FULL attn-chain speedup (fused 2 ops vs torch-native vs async-Ulysses): {ng} GPUs, iters={iters}")
        print(f"  GPU: {torch.cuda.get_device_name(rank)}   "
              f"(SQUARE weights: Wq/Wk/Wv/Wo=[hidden,hidden], Wqkv=[3*hidden,hidden]; hidden=N=nh*hd)")
        print(f"  times in us = ours/torch/async ; speedups = vs_torch/vs_async (x)")
        print(f"{'='*132}")
        print(f"{'(bs,nh,seq,hd) hid':<22} {'lay':>4} | "
              f"{'pre o/t/a':>17} {'attn':>6} {'post o/t/a':>17} | "
              f"{'e2e o/t/a':>17} | {'e2e t/a':>9} | {'c+g t/a':>9}")
        print('-' * 132)

    results = []
    for (bs, nheads, seq, head_dim) in SHAPES:
        hidden = nheads * head_dim
        N = hidden                               # SQUARE: Wo is [hidden, hidden]
        if nheads % sp or seq % sp or (seq // sp) % 128:
            if rank == 0: print(f"  ({bs},{nheads},{seq},{head_dim}) SKIP (divisibility)")
            dist.barrier(); continue
        local_nh = nheads // sp
        scale = 1.0 / math.sqrt(head_dim)

        g = torch.Generator(device=dev).manual_seed(42)
        Wq = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wk = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wv = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wo = torch.randn((N, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, head_dim)   # [3*hidden, hidden]
        Wqkv_t = Wqkv.t().contiguous()
        Wo_t = Wo.t().contiguous()
        Wq_t = Wq.t().contiguous(); Wk_t = Wk.t().contiguous(); Wv_t = Wv.t().contiguous()  # async-PRE split GEMMs
        n_qkv = 3 * hidden
        local_nqkv = n_qkv // sp                 # = 3*local_nh*hd

        # ---- attention timed ONCE with FlashAttention-4 (uniform-length THD == BSHD); local_nh heads ----
        # FA4 native dense layout is [B, S, H, D]; bs uniform-length sequences == both layouts' attn FLOPs.
        qb = torch.randn((bs, seq, local_nh, head_dim), dtype=torch.bfloat16, device=dev)
        kb = torch.randn_like(qb); vb = torch.randn_like(qb)

        def attn_fa4():
            o = fa4_func(qb, kb, vb, softmax_scale=scale, causal=False)
            return o[0] if isinstance(o, tuple) else o
        t_attn = time_call(attn_fa4, iters)

        for layout in ('BSHD', 'THD'):
            lbs, lseq = (bs, seq) if layout == 'BSHD' else (1, bs * seq)
            llocal_seq = lseq // sp
            if llocal_seq % 128:
                if rank == 0: print(f"  ({bs},{nheads},{seq},{head_dim}) {layout} SKIP (local_seq%128)")
                dist.barrier(); continue
            local_m = lbs * llocal_seq

            X_local = torch.randn((local_m, hidden), dtype=torch.bfloat16, device=dev)
            sym_pre = get_symm_buffer_for_gemm_a2a_transpose(group, lbs, lseq, n_qkv)
            sym_post = get_symm_buffer_for_a2a_transpose_gemm(group, lbs, nheads, lseq, head_dim)
            sym_post.x.copy_(torch.randn_like(sym_post.x))      # BHSD attn bytes (values irrelevant for timing)
            y = torch.zeros((local_m, N), dtype=torch.bfloat16, device=dev)

            # torch-native PRE: single fused-Wqkv matmul + transpose-scatter all_to_all
            send_pre = torch.empty((sp, lbs, llocal_seq, local_nqkv), dtype=torch.bfloat16, device=dev)
            recv_pre = torch.empty_like(send_pre)

            def pre_fused():
                bf16_gemm_a2a_transpose_nt(X_local, Wqkv, sym_pre, llocal_seq)

            def pre_torch():
                d = torch.matmul(X_local, Wqkv_t).view(lbs, llocal_seq, sp, local_nqkv)
                send_pre.copy_(d.permute(2, 0, 1, 3))
                dist.all_to_all_single(recv_pre, send_pre, group=group)

            # torch-native POST: transpose-scatter all_to_all (BHSD attn) + Wo matmul
            x_bhsd = torch.randn((lbs, local_nh, lseq, head_dim), dtype=torch.bfloat16, device=dev)
            send_po = x_bhsd.view(lbs, local_nh, sp, llocal_seq, head_dim).permute(2, 0, 3, 1, 4).contiguous()
            recv_po = torch.empty_like(send_po)

            def post_fused():
                bf16_a2a_transpose_gemm_nt_fused(y, Wo, sym_post)

            def post_torch():
                send_po.copy_(x_bhsd.view(lbs, local_nh, sp, llocal_seq, head_dim).permute(2, 0, 3, 1, 4))
                dist.all_to_all_single(recv_po, send_po, group=group)
                gathered = recv_po.permute(1, 2, 0, 3, 4).reshape(local_m, sp * local_nh * head_dim)
                torch.matmul(gathered, Wo_t)

            # ---- async-Ulysses PRE: split Q/K/V -> 3 GEMM + 3 A2A, multi-stream compute/comm overlap ----
            qkv_feat = local_nh * head_dim                 # per-(Q|K|V) local feature width; hidden = sp*qkv_feat
            Wt_qkv = (Wq_t, Wk_t, Wv_t)
            send_qkv = [torch.empty((sp, lbs, llocal_seq, qkv_feat), dtype=torch.bfloat16, device=dev) for _ in range(3)]
            recv_qkv = [torch.empty_like(s) for s in send_qkv]
            comm_stream = torch.cuda.Stream()

            def pre_async():
                comp = torch.cuda.current_stream()
                done = []
                for i in range(3):                          # Q, K, V
                    d = torch.matmul(X_local, Wt_qkv[i]).view(lbs, llocal_seq, sp, qkv_feat)
                    send_qkv[i].copy_(d.permute(2, 0, 1, 3))
                    ev = torch.cuda.Event(); ev.record(comp)
                    with torch.cuda.stream(comm_stream):    # A2A(i) overlaps GEMM(i+1) on comp stream
                        comm_stream.wait_event(ev)
                        dist.all_to_all_single(recv_qkv[i], send_qkv[i], group=group)
                        de = torch.cuda.Event(); de.record(comm_stream); done.append(de)
                for de in done:
                    comp.wait_event(de)

            # ---- async-Ulysses POST: split tokens into chunks -> per-chunk (scatter+A2A) overlaps Wo GEMM ----
            nseg = 4
            while llocal_seq % nseg:
                nseg //= 2                                  # llocal_seq is a multiple of 128 -> ends at nseg>=1
            seg = llocal_seq // nseg
            send_seg = [torch.empty((sp, lbs, seg, local_nh, head_dim), dtype=torch.bfloat16, device=dev) for _ in range(nseg)]
            recv_seg = [torch.empty_like(s) for s in send_seg]
            comm_stream_po = torch.cuda.Stream()

            def post_async():
                comp = torch.cuda.current_stream()
                src = x_bhsd.view(lbs, local_nh, sp, llocal_seq, head_dim).permute(2, 0, 3, 1, 4)  # non-contig view
                base = torch.cuda.Event(); base.record(comp)
                done = []
                with torch.cuda.stream(comm_stream_po):     # pipeline all chunk scatter+A2A on comm stream
                    comm_stream_po.wait_event(base)
                    for c in range(nseg):
                        send_seg[c].copy_(src[:, :, c * seg:(c + 1) * seg])
                        dist.all_to_all_single(recv_seg[c], send_seg[c], group=group)
                        ev = torch.cuda.Event(); ev.record(comm_stream_po); done.append(ev)
                for c in range(nseg):                        # GEMM(c) waits A2A(c); overlaps later A2As
                    comp.wait_event(done[c])
                    gc = recv_seg[c].permute(1, 2, 0, 3, 4).reshape(lbs * seg, sp * local_nh * head_dim)
                    torch.matmul(gc, Wo_t)

            t_pre_o = time_call(pre_fused, iters)
            t_pre_t = time_call(pre_torch, iters)
            t_pre_a = time_call(pre_async, iters)
            t_post_o = time_call(post_fused, iters, resets=[sym_post.reset_barriers])
            t_post_t = time_call(post_torch, iters)
            t_post_a = time_call(post_async, iters)

            e2e_o = t_pre_o + t_attn + t_post_o
            e2e_t = t_pre_t + t_attn + t_post_t
            e2e_a = t_pre_a + t_attn + t_post_a
            cg_o = t_pre_o + t_post_o
            sp_e2e_t = e2e_t / e2e_o if e2e_o > 0 else 0.0
            sp_e2e_a = e2e_a / e2e_o if e2e_o > 0 else 0.0
            sp_cg_t = (t_pre_t + t_post_t) / cg_o if cg_o > 0 else 0.0
            sp_cg_a = (t_pre_a + t_post_a) / cg_o if cg_o > 0 else 0.0

            if rank == 0:
                tag = f"({bs},{nheads},{seq},{head_dim}) {hidden}"
                print(f"{tag:<22} {layout:>4} | "
                      f"{t_pre_o:>5.0f}/{t_pre_t:>5.0f}/{t_pre_a:<5.0f} {t_attn:>6.0f} "
                      f"{t_post_o:>5.0f}/{t_post_t:>5.0f}/{t_post_a:<5.0f} | "
                      f"{e2e_o:>5.0f}/{e2e_t:>5.0f}/{e2e_a:<5.0f} | "
                      f"{sp_e2e_t:>4.2f}/{sp_e2e_a:<4.2f} | {sp_cg_t:>4.2f}/{sp_cg_a:<4.2f}")
                results.append({'layout': layout, 'sp_e2e_t': sp_e2e_t, 'sp_e2e_a': sp_e2e_a,
                                'sp_cg_t': sp_cg_t, 'sp_cg_a': sp_cg_a})
            sym_pre.destroy(); sym_post.destroy(); dist.barrier()

    if rank == 0 and results:
        geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0
        col = lambda lay, k: [r[k] for r in results if r['layout'] == lay and r[k] > 0]
        print('-' * 132)
        for lay in ('BSHD', 'THD'):
            print(f"  {lay}: geo_mean e2e speedup     vs_torch = {geo(col(lay, 'sp_e2e_t')):.3f}x"
                  f"   vs_async = {geo(col(lay, 'sp_e2e_a')):.3f}x")
            print(f"        geo_mean comm+GEMM speedup vs_torch = {geo(col(lay, 'sp_cg_t')):.3f}x"
                  f"   vs_async = {geo(col(lay, 'sp_cg_a')):.3f}x")
        print('=' * 132 + '\n')
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    port = find_free_port()
    print(f"Launching Ulysses FULL attn-chain benchmark with {ng} GPUs, {iters} iters...")
    mp.spawn(run, args=(ng, port, iters), nprocs=ng, join=True)
