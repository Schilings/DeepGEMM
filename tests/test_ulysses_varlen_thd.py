"""
Verify the claim: with THD (packed varlen) Ulysses SP, the sequence dimension can be split
UNIFORMLY across the whole packed total_tokens (not per-sequence). If so, the post-attn
A2A-transpose is a plain uniform split == our existing seq_major kernel (bs=1, seq=total_tokens),
and NO cu_seqlens-aware "dyn-seq" comm kernel is needed (cu_seqlens only matters inside attention).

Flow per rank r (sp = world_size), packed THD, uniform token split:
  X_local[T_local, hidden]                              # rank r owns packed tokens [r*T_local:...]
  --QKV proj (local)-->        q,k,v [T_local, nheads, hd]
  --pre-attn A2A (scatter heads, gather tokens)--> q,k,v [T, local_nh, hd]   # uniform gather == packed order
  --varlen attention (per-seq, cu_seqlens) -->     attn [T, local_nh, hd]    # THD output
  --post-attn A2A-transpose + Wo GEMM (OUR seq_major op, bs=1, seq=T)--> y[T_local, N]

Reference: single process recomputes the whole thing globally (full QKV + per-seq attention + Wo)
and slices rank r's uniform token shard. PASS => uniform split is correct AND our seq_major kernel
handles THD directly.

Usage: python tests/test_ulysses_varlen_thd.py <num_gpus>
"""

import os, sys, socket, math
from itertools import accumulate
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)); return s.getsockname()[1]


def varlen_sdpa_thd(q, k, v, cu, scale):
    """Per-sequence SDPA on THD tensors q,k,v [T, H, hd] with cu_seqlens cu. -> [T, H, hd]."""
    out = torch.empty_like(q)
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        qi = q[s:e].transpose(0, 1)            # [H, L, hd]
        ki = k[s:e].transpose(0, 1)
        vi = v[s:e].transpose(0, 1)
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=scale)
        out[s:e] = oi.transpose(0, 1)
    return out


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}'); sp = ng

    import deep_gemm
    from deep_gemm.a2a_transpose_gemm import (
        get_symm_buffer_for_a2a_transpose_gemm, bf16_a2a_transpose_gemm_nt)

    have_fa = False
    try:
        from flash_attn import flash_attn_varlen_func
        have_fa = True
    except Exception:
        pass

    nheads, head_dim, N = 16, 128, 2048
    hidden = nheads * head_dim
    local_nh = nheads // sp
    seqlens = [512, 768, 256, 512]            # variable-length sequences, packed
    T = sum(seqlens)                          # 2048
    assert T % sp == 0, "padded packed total_tokens must be divisible by sp"
    T_local = T // sp
    scale = 1.0 / math.sqrt(head_dim)
    cu = torch.tensor([0] + list(accumulate(seqlens)), dtype=torch.int32, device=dev)
    max_seq = max(seqlens)

    # shared weights + global packed input (identical on all ranks via fixed seed)
    g = torch.Generator(device=dev).manual_seed(0)
    Wq = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wk = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wv = torch.randn((hidden, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Wo = torch.randn((N, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)
    Xg = torch.randn((T, hidden), dtype=torch.bfloat16, device=dev, generator=g) / math.sqrt(hidden)

    # this rank's UNIFORM packed-token shard
    X_local = Xg[rank * T_local:(rank + 1) * T_local].contiguous()     # [T_local, hidden]

    # ---- distributed Ulysses (uniform token split) ----
    def proj(W):
        return (X_local @ W).view(T_local, nheads, head_dim)           # [T_local, H, hd]
    q, k, v = proj(Wq), proj(Wk), proj(Wv)

    def pre_a2a(t):  # [T_local, H, hd] -> [T, local_nh, hd]  (scatter heads, gather tokens)
        send = [t[:, d * local_nh:(d + 1) * local_nh, :].contiguous() for d in range(sp)]
        recv = [torch.empty_like(send[0]) for _ in range(sp)]
        dist.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=0)                                  # [sp*T_local=T, local_nh, hd] = packed order
    qf, kf, vf = pre_a2a(q), pre_a2a(k), pre_a2a(v)                     # [T, local_nh, hd]

    if have_fa:
        attn = flash_attn_varlen_func(qf, kf, vf, cu, cu, max_seq, max_seq,
                                      dropout_p=0.0, softmax_scale=scale, causal=False)
    else:
        attn = varlen_sdpa_thd(qf, kf, vf, cu, scale)                  # [T, local_nh, hd] THD

    # ---- post-attn A2A-transpose + Wo GEMM (OUR seq_major op; THD == bs=1, seq=T) ----
    sym = get_symm_buffer_for_a2a_transpose_gemm(group, 1, nheads, T, head_dim)
    sym.x.view(-1).copy_(attn.reshape(-1).to(torch.bfloat16))          # BSHD/THD bytes [1, T, local_nh, hd]
    y = torch.zeros((T_local, N), dtype=torch.bfloat16, device=dev)
    bf16_a2a_transpose_gemm_nt(y, Wo, sym, seq_major=True)
    torch.cuda.synchronize()

    # ---- single-process reference: global per-seq attention + Wo, slice uniform shard ----
    qg = (Xg @ Wq).view(T, nheads, head_dim)
    kg = (Xg @ Wk).view(T, nheads, head_dim)
    vg = (Xg @ Wv).view(T, nheads, head_dim)
    ag = varlen_sdpa_thd(qg, kg, vg, cu, scale).reshape(T, hidden)     # [T, hidden]
    Yg = (ag.float() @ Wo.float().t())
    y_ref = Yg[rank * T_local:(rank + 1) * T_local]                    # [T_local, N]

    rel = (y.float() - y_ref).abs().mean().item() / (y_ref.abs().mean().item() + 1e-8)
    passed = rel < 0.03
    flags = torch.tensor([1.0 if passed else 0.0], device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        attn_impl = "flash_attn_varlen" if have_fa else "manual per-seq SDPA"
        print(f"  attn={attn_impl}  seqlens={seqlens} T={T} sp={sp} T_local={T_local} local_nh={local_nh}")
        print(f"  rel={rel:.3e}  ->  {'PASS' if flags.item() > 0.5 else 'FAIL'}"
              f"   (uniform packed-token split + our seq_major THD op)")
    dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    port = find_free_port()
    print(f"Launching varlen-THD Ulysses verification with {ng} GPUs...")
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
