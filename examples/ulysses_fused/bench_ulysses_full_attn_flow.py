"""Standard Ulysses forward benchmark: torch baseline vs DeepGEMM fused.

Both paths implement the same standard Ulysses dataflow and retain replicated
full Q/K/V/Wo weights.  Exactly two benchmark arms are implemented.

Per-rank sequence-parallel flow:
  X_local[bs, local_seq, hidden]
    --[PRE]  QKV projection + A2A transpose --> q,k,v [bs, seq, local_nh, hd]
    --[ATTN] FlashAttention-4 -------------> attention output
    --[POST] A2A transpose + full Wo GEMM -> y[bs*local_seq, hidden]

Compared paths:
  baseline = torch.matmul + synchronous NCCL all_to_all_single
  fused    = DeepGEMM GEMM+A2A PRE + A2A+GEMM POST

Attention is identical and timed once.  The reported "chain" time is the sum of
independently timed PRE, attention and POST stages; it is not an autograd
training benchmark or a tensor-connected end-to-end execution.

Usage:
  python3 examples/ulysses_fused/bench_ulysses_full_attn_flow.py <num_gpus> [iters]
"""

import os, sys, socket, math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


# (bs, nheads, seq, head_dim)  -- SQUARE weights: hidden = N = nheads*head_dim
# LONG-SEQUENCE regime: Ulysses SP only makes sense for long sequences. With sp=8 each rank
# holds local_seq = (bs*seq)/8 tokens; all below give local_seq that is a multiple of 128.
SHAPES = [
    (1, 32, 32768,  128),   # hidden=4096  | 1 sequence of  32K tokens  (per-rank  4K)
    (1, 64, 32768,  128),   # hidden=8192  | 1 sequence of  32K tokens  (per-rank  4K)
    (1, 64, 65536,  128),   # hidden=8192  | 1 sequence of  64K tokens  (per-rank  8K)
    (1, 32, 131072, 128),   # hidden=4096  | 1 sequence of 128K tokens  (per-rank 16K)
    (2, 32, 32768,  128),   # hidden=4096  | BSHD 2x32K ; THD packs to 1x64K (BSHD==THD demo)
    # 🌟 Wan2.1 attention shapes (vae_stride=(4,8,8), patch=(1,2,2); 81帧→T_latent=21)
    #   seq = (T/4)×(H/16)×(W/16); 480p 832×480: 21×30×52=32760→pad 32768; 720p 1280×720: 21×45×80=75600→pad 75776
    (1, 40, 32768,  128),   # 🌟 Wan2.1 14B 480p 81帧 | hidden=5120 nh=40 | per-rank 4K
    (1, 40, 75776,  128),   # 🌟 Wan2.1 14B 720p 81帧 | hidden=5120 nh=40 | per-rank 9472
    (1, 16, 32768,  128),   # 🌟 Wan2.1 1.3B 480p 81帧| hidden=2048 nh=16 | per-rank 4K
    (1, 16, 75776,  128),   # 🌟 Wan2.1 1.3B 720p 81帧| hidden=2048 nh=16 | per-rank 9472
]


def kfmt(n):
    """Human-readable token count: 32768 -> '32K', 131072 -> '128K', non-multiples kept as-is."""
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)


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
        bf16_a2a_transpose_gemm_nt_fused,
    )
    from flash_attn.cute import flash_attn_func as fa4_func   # FlashAttention-4 (see docs/INSTALL_FA4.md)

    def time_call(fn, it, resets=()):
        for _ in range(3):
            for reset in resets:
                reset()
            torch.cuda.synchronize()
            dist.barrier(group)
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_us = 0.0
        for _ in range(it):
            for reset in resets:
                reset()
            torch.cuda.synchronize()
            dist.barrier(group)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            elapsed = torch.tensor(start.elapsed_time(end), dtype=torch.float64, device=dev)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
            total_us += elapsed.item() * 1000.0
        return total_us / it

    if rank == 0:
        print(f"\n{'='*118}")
        print(f"  Standard Ulysses forward: baseline vs DeepGEMM fused — {ng} GPUs, iters={iters}")
        print(f"  GPU: {torch.cuda.get_device_name(rank)}")
        print("  BF16 square weights; FA4 attention is identical; all times are rank-max microseconds")
        print("  chain = independently timed PRE + ATTN + POST")
        print(f"{'='*118}")
        print(f"{'h / nh / lbs x lseq / L':<28} {'lay':>4} | "
              f"{'PRE f/base':>15} {'ATTN':>8} {'POST f/base':>15} | "
              f"{'chain f/base':>17} {'e2e':>7} {'c+g':>7}")
        print('-' * 118)

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

            t_pre_fused = time_call(pre_fused, iters)
            t_pre_baseline = time_call(pre_torch, iters)
            t_post_fused = time_call(post_fused, iters, resets=[sym_post.reset_barriers])
            t_post_baseline = time_call(post_torch, iters)

            chain_fused = t_pre_fused + t_attn + t_post_fused
            chain_baseline = t_pre_baseline + t_attn + t_post_baseline
            comm_gemm_fused = t_pre_fused + t_post_fused
            e2e_speedup = chain_baseline / chain_fused
            comm_gemm_speedup = (t_pre_baseline + t_post_baseline) / comm_gemm_fused

            if rank == 0:
                tag = f"h{hidden} nh{nheads} {lbs}x{kfmt(lseq)} L{kfmt(llocal_seq)}"
                print(f"{tag:<28} {layout:>4} | "
                      f"{t_pre_fused:>7.0f}/{t_pre_baseline:<7.0f} {t_attn:>8.0f} "
                      f"{t_post_fused:>7.0f}/{t_post_baseline:<7.0f} | "
                      f"{chain_fused:>8.0f}/{chain_baseline:<8.0f} "
                      f"{e2e_speedup:>6.2f}x {comm_gemm_speedup:>6.2f}x")
                results.append({
                    'layout': layout,
                    'e2e_speedup': e2e_speedup,
                    'comm_gemm_speedup': comm_gemm_speedup,
                })
            sym_pre.destroy()
            sym_post.destroy()
            dist.barrier()

    if rank == 0 and results:
        def geo(values):
            return math.exp(sum(math.log(value) for value in values) / len(values))

        print('-' * 118)
        for layout in ('BSHD', 'THD'):
            rows = [result for result in results if result['layout'] == layout]
            print(
                f"  {layout}: geo_mean chain speedup = "
                f"{geo([row['e2e_speedup'] for row in rows]):.3f}x; "
                f"PRE+POST speedup = "
                f"{geo([row['comm_gemm_speedup'] for row in rows]):.3f}x"
            )
        print('=' * 118 + '\n')
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    port = find_free_port()
    print(f"Launching Ulysses FULL attn-chain benchmark with {ng} GPUs, {iters} iters...")
    mp.spawn(run, args=(ng, port, iters), nprocs=ng, join=True)
