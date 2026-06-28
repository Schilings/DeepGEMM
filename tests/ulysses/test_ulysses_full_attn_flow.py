"""
FULL Ulysses SP attention flow using BOTH fused ops end-to-end, with correctness + profiling.

Flow per rank r (sp = world_size, sequence-parallel input):
  X_local[bs, local_seq, hidden]                       # rank r owns seq shard [r*local_seq:...]
  --[PRE-ATTN OP] fused QKV proj + A2A-transpose--> q,k,v [bs, seq, local_nheads, hd]   (OUR op #1)
  --attention (FlashAttention-4, per local head, full seq)--> attn [bs, local_nheads, seq, hd]
  --[POST-ATTN OP] A2A-transpose + Wo GEMM-->      y     [bs*local_seq, N]              (OUR op #2)

PRE-ATTN op:  bf16_gemm_a2a_transpose_nt  (QKV is ONE fused linear Wqkv[3*nheads*hd, hidden],
              rank-major rows so the contiguous-N scatter delivers each rank its [Q,K,V] head group).
POST-ATTN op: bf16_a2a_transpose_gemm_nt  (Wo[N, hidden]).

Correctness: each rank recomputes the WHOLE flow from the all-gathered global X (global fused QKV +
global attention + Wo) and slices its own seq shard; compare to the distributed (both-fused) result.
Both fused ops are timed/profiled separately and as a sum.

Usage: python tests/test_ulysses_full_attn_flow.py <num_gpus> [iters]
"""

import os, sys, socket, math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fa4_attn import fa4_attn_bhsd


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


# (bs, nheads, seq, head_dim, N)   -- hidden = nheads*head_dim, N = Wo output width
SHAPES = [
    (1, 32, 2048, 128, 4096),
    (1, 56, 2048, 128, 7168),
    (8, 56, 4096, 128, 7168),
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
        get_symm_buffer_for_a2a_transpose_gemm, bf16_a2a_transpose_gemm_nt)

    if rank == 0:
        print(f"\n{'='*108}\n  FULL Ulysses SP attn flow (BOTH fused ops): {ng} GPUs (correctness end-to-end + per-op timing)\n{'='*108}")
        print(f"{'(bs,nh,seq,hd,N)':<24} | {'e2e status':>12} | {'pre(us)':>9} {'post(us)':>9} {'sum(us)':>9}")

    def time_call(fn, it):
        for _ in range(3):
            torch.cuda.synchronize(); dist.barrier(group); fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        tot = 0.0
        for _ in range(it):
            torch.cuda.synchronize(); dist.barrier(group)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            tot += s.elapsed_time(e)
        return tot / it * 1000.0

    num_pass = 0; fails = []
    for (bs, nheads, seq, head_dim, N) in SHAPES:
        local_seq = seq // sp
        if nheads % sp or seq % sp or local_seq % 128:
            if rank == 0: print(f"  ({bs},{nheads},{seq},{head_dim},{N}) SKIP")
            dist.barrier(); continue
        local_nh = nheads // sp; hidden = nheads * head_dim; loc = local_nh * head_dim
        g = torch.Generator(device=dev).manual_seed(42)
        Wq = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wk = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wv = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wo = torch.randn((N, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, head_dim)   # [3*hidden, hidden]
        X_local = torch.randn((bs, local_seq, hidden), dtype=torch.bfloat16, device=dev)

        # ---- PRE-ATTN OP: fused QKV proj + A2A-transpose ----
        sym_pre = get_symm_buffer_for_gemm_a2a_transpose(group, bs, seq, 3 * hidden)
        a = X_local.reshape(bs * local_seq, hidden)
        out = bf16_gemm_a2a_transpose_nt(a, Wqkv, sym_pre, local_seq)     # [bs, seq, 3*loc]
        qf = out[..., 0:loc].reshape(bs, seq, local_nh, head_dim)
        kf = out[..., loc:2 * loc].reshape(bs, seq, local_nh, head_dim)
        vf = out[..., 2 * loc:3 * loc].reshape(bs, seq, local_nh, head_dim)

        # ---- attention (FlashAttention-4, per local head, full seq) ----
        attn = fa4_attn_bhsd(qf.permute(0, 2, 1, 3), kf.permute(0, 2, 1, 3),
                             vf.permute(0, 2, 1, 3), head_dim).contiguous()  # [bs, local_nh, seq, hd]

        # ---- POST-ATTN OP: A2A-transpose + Wo GEMM ----
        sym_post = get_symm_buffer_for_a2a_transpose_gemm(group, bs, nheads, seq, head_dim)
        sym_post.x.copy_(attn)
        y = torch.zeros((bs * local_seq, N), dtype=torch.bfloat16, device=dev)
        bf16_a2a_transpose_gemm_nt(y, Wo, sym_post)                      # default M0
        torch.cuda.synchronize()

        # ---- single-process reference (each rank recomputes the whole flow globally) ----
        xs = [torch.empty_like(X_local) for _ in range(sp)]
        dist.all_gather(xs, X_local, group=group)
        Xg = torch.cat(xs, dim=1)                                        # [bs, seq, hidden]
        qg = (Xg @ Wq.t()).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        kg = (Xg @ Wk.t()).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        vg = (Xg @ Wv.t()).view(bs, seq, nheads, head_dim).permute(0, 2, 1, 3)
        # Attention MUST be computed per rank head-group (each rank runs FA4 on its own local_nh
        # heads). A single all-head attention instead differs by ~1e-3 in bf16 purely from a
        # different head-count tiling / reduction order -- a reference artifact, not real error.
        # Keeping the final Wo projection in FP32 so rel reflects the genuine bf16-output-GEMM floor.
        ag_groups = [fa4_attn_bhsd(qg[:, d * local_nh:(d + 1) * local_nh],
                                   kg[:, d * local_nh:(d + 1) * local_nh],
                                   vg[:, d * local_nh:(d + 1) * local_nh], head_dim) for d in range(sp)]
        ag = torch.cat(ag_groups, dim=1).permute(0, 2, 1, 3).reshape(bs, seq, hidden)
        Yg = (ag.float() @ Wo.float().t())                              # [bs, seq, N]
        y_ref = Yg[:, rank * local_seq:(rank + 1) * local_seq, :].reshape(bs * local_seq, N)

        rel = (y.float() - y_ref).abs().mean().item() / (y_ref.abs().mean().item() + 1e-8)
        passed = rel < 0.03

        # ---- profile each fused op ----
        t_pre = time_call(lambda: bf16_gemm_a2a_transpose_nt(a, Wqkv, sym_pre, local_seq), it=iters)
        t_post = time_call(lambda: bf16_a2a_transpose_gemm_nt(y, Wo, sym_post), it=iters)
        if rank == 0:
            print(f"  ({bs},{nheads},{seq},{head_dim},{N})".ljust(24) +
                  f" | {('PASS' if passed else 'FAIL'):>12} | {t_pre:>9.1f} {t_post:>9.1f} {t_pre + t_post:>9.1f}"
                  f"   (rel={rel:.2e})")
            num_pass += int(passed)
            if not passed: fails.append((bs, nheads, seq, head_dim, N))
        sym_pre.destroy(); sym_post.destroy(); dist.barrier()

    if rank == 0:
        print(f"\n  e2e correctness: {num_pass}/{len(SHAPES)} passed" +
              ("  ALL PASS" if not fails else f"  FAILED: {fails}"))
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    port = find_free_port()
    print(f"Launching FULL Ulysses attn flow (both fused ops) with {ng} GPUs...")
    mp.spawn(run, args=(ng, port, iters), nprocs=ng, join=True)
