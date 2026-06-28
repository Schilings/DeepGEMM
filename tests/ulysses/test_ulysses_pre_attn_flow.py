"""
Ulysses SP **PRE-ATTN** flow, with correctness end-to-end and profiling of ONLY the pre-attn
fused QKV-proj GEMM + A2A-transpose (OUR op `bf16_gemm_a2a_transpose_nt`).

Flow per rank r (sp = world_size, sequence-parallel input):
  X_local[bs, local_seq, hidden]                       # rank r owns seq shard [r*local_seq:...]
  --fused QKV proj + A2A-transpose (OUR OP)--> q,k,v [bs, seq, local_nheads, hd]  # gather seq, scatter heads
  --attention (FlashAttention-4, per local head, full seq)--> attn[bs, local_nheads, seq, hd]

QKV is a SINGLE fused linear Wqkv[3*nheads*hd, hidden] (NT layout). Its rows are laid out
rank-major so the op's contiguous-N scatter lands each rank's [Q, K, V] head group together:
  rows[d*local_n : (d+1)*local_n] = [Q heads(d), K heads(d), V heads(d)],  local_n = 3*local_nh*hd.
After the op, rank r's output [bs, seq, local_n] splits on the last dim into q|k|v, each
[bs, seq, local_nheads, hd] (BSHD, FlashAttention-native).

Correctness: each rank recomputes the WHOLE thing from the all-gathered global X (global fused QKV
+ global attention) and slices its own head group / full seq; compare q,k,v and attn to the
distributed result. Only the pre-attn fused GEMM+A2A op is timed/profiled.

Usage: python tests/test_ulysses_pre_attn_flow.py <num_gpus> [iters]
"""

import os, sys, socket, math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fa4_attn import fa4_attn_bhsd


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


# (bs, nheads, seq, head_dim)  -- hidden = K = nheads*head_dim
SHAPES = [
    (1, 32, 2048, 128),
    (1, 56, 2048, 128),
    (8, 56, 4096, 128),
]


def build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, hd):
    """Wq/Wk/Wv: [nheads*hd, hidden] (NT, row = output feature).
    Return Wqkv [3*nheads*hd, hidden] with rank-major [Q,K,V] head-group blocks so the op's
    contiguous-N scatter delivers each rank its own Q/K/V heads."""
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

    if rank == 0:
        print(f"\n{'='*94}\n  Ulysses SP PRE-attn flow: {ng} GPUs (correctness end-to-end + pre-attn-only timing)\n{'='*94}")
        print(f"{'(bs,nh,seq,hd)':<22} | {'e2e status':>12} | {'pre fused(us)':>14}")

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
    for (bs, nheads, seq, head_dim) in SHAPES:
        local_seq = seq // sp
        if nheads % sp or seq % sp or local_seq % 128:
            if rank == 0: print(f"  ({bs},{nheads},{seq},{head_dim}) SKIP")
            dist.barrier(); continue
        local_nh = nheads // sp; hidden = nheads * head_dim; loc = local_nh * head_dim
        g = torch.Generator(device=dev).manual_seed(42)
        # shared weights across ranks (so the global reference is well-defined)
        Wq = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wk = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wv = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
        Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, head_dim)   # [3*hidden, hidden]
        # this rank's seq shard of the global input
        X_local = torch.randn((bs, local_seq, hidden), dtype=torch.bfloat16, device=dev)

        # ---- distributed pre-attn: fused QKV proj + A2A-transpose (OUR OP) ----
        sym = get_symm_buffer_for_gemm_a2a_transpose(group, bs, seq, 3 * hidden)
        a = X_local.reshape(bs * local_seq, hidden)
        out = bf16_gemm_a2a_transpose_nt(a, Wqkv, sym, local_seq)         # [bs, seq, 3*loc]
        qf = out[..., 0:loc].reshape(bs, seq, local_nh, head_dim)
        kf = out[..., loc:2 * loc].reshape(bs, seq, local_nh, head_dim)
        vf = out[..., 2 * loc:3 * loc].reshape(bs, seq, local_nh, head_dim)
        attn = fa4_attn_bhsd(qf.permute(0, 2, 1, 3), kf.permute(0, 2, 1, 3),
                             vf.permute(0, 2, 1, 3), head_dim).contiguous()  # [bs, local_nh, seq, hd]
        torch.cuda.synchronize()

        # ---- single-process reference (each rank recomputes the whole flow globally) ----
        xs = [torch.empty_like(X_local) for _ in range(sp)]
        dist.all_gather(xs, X_local, group=group)
        Xg = torch.cat(xs, dim=1)                                        # [bs, seq, hidden]
        hs = slice(rank * local_nh, (rank + 1) * local_nh)               # this rank's head group
        qg = (Xg @ Wq.t()).view(bs, seq, nheads, head_dim)[:, :, hs, :]
        kg = (Xg @ Wk.t()).view(bs, seq, nheads, head_dim)[:, :, hs, :]
        vg = (Xg @ Wv.t()).view(bs, seq, nheads, head_dim)[:, :, hs, :]
        attn_ref = fa4_attn_bhsd(qg.permute(0, 2, 1, 3), kg.permute(0, 2, 1, 3),
                                 vg.permute(0, 2, 1, 3), head_dim).contiguous()  # [bs, local_nh, seq, hd]

        def rel_err(x, y):
            return (x.float() - y.float()).abs().mean().item() / (y.float().abs().mean().item() + 1e-8)
        rel_q = rel_err(qf, qg); rel_k = rel_err(kf, kg); rel_v = rel_err(vf, vg)
        rel_a = rel_err(attn, attn_ref)
        passed = max(rel_q, rel_k, rel_v, rel_a) < 0.03

        # ---- profile ONLY the pre-attn fused GEMM+A2A op ----
        t_pre = time_call(lambda: bf16_gemm_a2a_transpose_nt(a, Wqkv, sym, local_seq), it=iters)
        if rank == 0:
            print(f"  ({bs},{nheads},{seq},{head_dim})".ljust(22) +
                  f" | {('PASS' if passed else 'FAIL'):>12} | {t_pre:>14.1f}"
                  f"   (rel q/k/v/attn={rel_q:.1e}/{rel_k:.1e}/{rel_v:.1e}/{rel_a:.1e})")
            num_pass += int(passed)
            if not passed: fails.append((bs, nheads, seq, head_dim))
        sym.destroy(); dist.barrier()

    if rank == 0:
        print(f"\n  e2e correctness: {num_pass}/{len(SHAPES)} passed" +
              ("  ALL PASS" if not fails else f"  FAILED: {fails}"))
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    port = find_free_port()
    print(f"Launching Ulysses PRE-attn flow with {ng} GPUs...")
    mp.spawn(run, args=(ng, port, iters), nprocs=ng, join=True)
