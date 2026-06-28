"""
**PRE-ATTN** varlen (THD) verification. Mirror of test_ulysses_post_attn_varlen_thd.py for the
pre-attn fused op. Verify the claim: with THD (packed varlen) Ulysses SP, the sequence dimension
can be split UNIFORMLY across the whole packed total_tokens (not per-sequence). If so, the pre-attn
fused QKV-proj + A2A-transpose is a plain uniform split == our existing kernel (bs=1, seq=T,
local_seq=T//sp), and NO cu_seqlens-aware "dyn-seq" comm kernel is needed (cu_seqlens only matters
inside attention). Op under test: `bf16_gemm_a2a_transpose_nt` (fused QKV proj + gather-seq/scatter-heads).

Flow per rank r (sp = world_size), packed THD, uniform token split:
  X_local[T_local, hidden]                                     # rank r owns packed tokens [r*T_local:...]
  --fused QKV proj + A2A-transpose (OUR OP, bs=1, seq=T)-->     out[1, T, 3*loc]
      split last dim into q|k|v -> each [T, local_nh, hd]      # gather packed tokens, this rank's head group
  --varlen attention (per-seq, cu_seqlens)-->                  attn[T, local_nh, hd]   # THD output

QKV is a SINGLE fused linear Wqkv[3*nheads*hd, hidden] laid out rank-major so the op's contiguous-N
scatter lands each rank's [Q, K, V] head group together (same contract as test_ulysses_pre_attn_flow).

Reference: single process recomputes the whole thing globally (global fused QKV + per-seq varlen
attention) and slices rank r's head group. PASS => uniform token split is correct for the pre-attn
op AND it handles packed THD directly (bs=1, seq=total_tokens).

Usage: python tests/ulysses/test_ulysses_pre_attn_varlen_thd.py <num_gpus>
"""

import os, sys, socket, math
from itertools import accumulate
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fa4_attn import fa4_attn_varlen_thd


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


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


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}'); sp = ng

    import deep_gemm
    from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose, bf16_gemm_a2a_transpose_nt

    nheads, head_dim = 16, 128
    hidden = nheads * head_dim
    local_nh = nheads // sp
    loc = local_nh * head_dim
    seqlens = [512, 768, 256, 512]            # variable-length sequences, packed
    T = sum(seqlens)                          # 2048
    assert T % sp == 0, "packed total_tokens must be divisible by sp"
    T_local = T // sp                         # uniform packed-token shard
    assert T_local % 128 == 0, "local_seq (T//sp) must be a multiple of 128 for the kernel"
    cu = torch.tensor([0] + list(accumulate(seqlens)), dtype=torch.int32, device=dev)
    max_seq = max(seqlens)

    # shared weights + global packed input (identical on all ranks via fixed seed)
    g = torch.Generator(device=dev).manual_seed(0)
    Wq = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wk = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wv = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Xg = torch.randn((T, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, head_dim)        # [3*hidden, hidden]

    # this rank's UNIFORM packed-token shard
    X_local = Xg[rank * T_local:(rank + 1) * T_local].contiguous()         # [T_local, hidden]

    # ---- distributed pre-attn: fused QKV proj + A2A-transpose (OUR OP; THD == bs=1, seq=T) ----
    sym = get_symm_buffer_for_gemm_a2a_transpose(group, 1, T, 3 * hidden)
    out = bf16_gemm_a2a_transpose_nt(X_local, Wqkv, sym, T_local)          # [1, T, 3*loc]
    qf = out[..., 0:loc].reshape(T, local_nh, head_dim)
    kf = out[..., loc:2 * loc].reshape(T, local_nh, head_dim)
    vf = out[..., 2 * loc:3 * loc].reshape(T, local_nh, head_dim)

    attn = fa4_attn_varlen_thd(qf, kf, vf, cu, max_seq, head_dim)           # [T, local_nh, hd] THD
    torch.cuda.synchronize()

    # ---- single-process reference: global fused QKV + per-seq varlen attention, slice head group ----
    hs = slice(rank * local_nh, (rank + 1) * local_nh)                     # this rank's head group
    qg = (Xg @ Wq.t()).view(T, nheads, head_dim)[:, hs, :].contiguous()    # [T, local_nh, hd]
    kg = (Xg @ Wk.t()).view(T, nheads, head_dim)[:, hs, :].contiguous()
    vg = (Xg @ Wv.t()).view(T, nheads, head_dim)[:, hs, :].contiguous()
    attn_ref = fa4_attn_varlen_thd(qg, kg, vg, cu, max_seq, head_dim)      # [T, local_nh, hd]

    def rel_err(x, y):
        return (x.float() - y.float()).abs().mean().item() / (y.float().abs().mean().item() + 1e-8)
    rel_q, rel_k, rel_v = rel_err(qf, qg), rel_err(kf, kg), rel_err(vf, vg)
    rel_a = rel_err(attn, attn_ref)
    passed = max(rel_q, rel_k, rel_v, rel_a) < 0.03

    flags = torch.tensor([1.0 if passed else 0.0], device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        print(f"  attn=flash_attn_4_varlen  seqlens={seqlens} T={T} sp={sp} T_local={T_local} local_nh={local_nh}")
        print(f"  rel q/k/v/attn={rel_q:.1e}/{rel_k:.1e}/{rel_v:.1e}/{rel_a:.1e}  ->  "
              f"{'PASS' if flags.item() > 0.5 else 'FAIL'}"
              f"   (uniform packed-token split + our fused pre-attn THD op)")
    sym.destroy()
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    port = find_free_port()
    print(f"Launching PRE-attn varlen-THD Ulysses verification with {ng} GPUs...")
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
