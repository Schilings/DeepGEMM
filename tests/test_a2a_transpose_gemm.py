"""
Semantics-lock test for Ulysses SP post-attention A2A-transpose + Wo GEMM.

This test pins down the CORRECT dataflow (gather along hidden/K with a seq<->head
transpose), independent of any kernel, so the upcoming CUDA implementation has a
non-circular ground truth.

Dataflow (sp_size = world_size here):
  Attention output, per rank r:  x_r[bs, local_nheads, seq, head_dim]
      where rank r owns heads [r*local_nheads : (r+1)*local_nheads], for the FULL seq.
  A2A-transpose -> per rank r:    xt_r[bs, local_seq, nheads, head_dim] = [bs, local_seq, hidden]
      xt_r[b, s, h, d] = X_global[b, h, r*local_seq + s, d]
      i.e. rank r keeps its own seq shard but gathers ALL heads (full hidden) from every rank.
  Wo GEMM -> y_r[bs*local_seq, N] = xt_r.reshape(bs*local_seq, hidden) @ Wo.t()

Two independent computations are compared:
  - ground_truth: reconstruct X_global via all_gather of inputs, then slice/permute (no A2A).
  - candidate:    perform the transpose A2A via dist.all_to_all + assembly (the real comm pattern).
If they match, the transpose semantics + reference are correct.

Usage: python tests/test_a2a_transpose_gemm.py <num_gpus>
"""

import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


# (bs, nheads, seq, head_dim, N) — Ulysses post-attn shapes; hidden = nheads*head_dim
SHAPES = [
    (1, 32, 2048, 128, 4096),   # hidden=4096
    (1, 16, 4096, 128, 2048),   # hidden=2048
    (2, 32, 1024, 128, 4096),   # bs=2, hidden=4096
    (1, 56, 2048, 128, 7168),   # hidden=7168
]


def a2a_transpose_candidate(x_r, sp_size, rank, group):
    """Real transpose-A2A via dist.all_to_all.
    x_r: [bs, local_nheads, seq, head_dim] (this rank's heads, full seq).
    returns xt_r: [bs, local_seq, nheads, head_dim].
    """
    bs, local_nheads, seq, head_dim = x_r.shape
    local_seq = seq // sp_size
    # send_list[dst] = our heads for dst's seq shard: [bs, local_nheads, local_seq, head_dim]
    send_list = [x_r[:, :, d * local_seq:(d + 1) * local_seq, :].contiguous() for d in range(sp_size)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(sp_size)]
    dist.all_to_all(recv_list, send_list, group=group)
    # recv_list[src] = src's heads for OUR seq shard: [bs, local_nheads, local_seq, head_dim]
    # place src s into head slice [s*local_nheads:(s+1)*local_nheads]; layout [bs, local_seq, nheads, head_dim]
    parts = [recv_list[s].permute(0, 2, 1, 3).contiguous() for s in range(sp_size)]  # [bs, local_seq, local_nheads, head_dim]
    xt = torch.cat(parts, dim=2)  # [bs, local_seq, nheads, head_dim]
    return xt


def run_test(rank, num_gpus, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(num_gpus)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)
    group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')
    sp = num_gpus

    if rank == 0:
        print(f"\n{'='*84}\n  A2A-transpose + Wo GEMM semantics-lock: {num_gpus} GPUs, {len(SHAPES)} shapes\n{'='*84}")
        print(f"{'(bs,nh,seq,hd,N)':<26} | {'Max Diff':>10} {'Rel Err':>10} | Status")

    num_passed = 0
    fails = []
    for (bs, nheads, seq, head_dim, N) in SHAPES:
        if nheads % sp or seq % sp:
            if rank == 0:
                print(f"  ({bs},{nheads},{seq},{head_dim},{N}) SKIP (not divisible by sp={sp})")
            dist.barrier(); continue
        local_nheads = nheads // sp
        local_seq = seq // sp
        hidden = nheads * head_dim

        # Wo identical across ranks
        g = torch.Generator(device=device).manual_seed(1234)
        Wo = torch.randn((N, hidden), dtype=torch.bfloat16, device=device, generator=g)

        # this rank's attention output (its head slice, full seq)
        x_r = torch.randn((bs, local_nheads, seq, head_dim), dtype=torch.bfloat16, device=device)

        # --- candidate: real transpose A2A + GEMM ---
        xt = a2a_transpose_candidate(x_r, sp, rank, group)          # [bs, local_seq, nheads, head_dim]
        a_cand = xt.reshape(bs * local_seq, hidden)
        d_cand = (a_cand.float() @ Wo.float().t()).to(torch.bfloat16)

        # --- ground truth: reconstruct X_global via all_gather, slice our seq shard ---
        xg = [torch.empty_like(x_r) for _ in range(sp)]
        dist.all_gather(xg, x_r, group=group)                       # xg[s] = src s's heads, full seq
        X_global = torch.cat(xg, dim=1)                             # [bs, nheads, seq, head_dim]
        gt = X_global[:, :, rank * local_seq:(rank + 1) * local_seq, :]  # [bs, nheads, local_seq, head_dim]
        a_gt = gt.permute(0, 2, 1, 3).reshape(bs * local_seq, hidden)
        d_gt = (a_gt.float() @ Wo.float().t()).to(torch.bfloat16)

        # candidate(torch) sanity vs ground truth (semantics lock)
        rel_cand = (d_cand.float() - d_gt.float()).abs().mean().item() / (d_gt.float().abs().mean().item() + 1e-8)

        # --- kernels under test: BOTH default M0 and the opt-in fused M1 ---
        import deep_gemm
        from deep_gemm.a2a_transpose_gemm import (
            get_symm_buffer_for_a2a_transpose_gemm,
            bf16_a2a_transpose_gemm_nt, bf16_a2a_transpose_gemm_nt_fused)

        def run_and_diff(fn):
            sym = get_symm_buffer_for_a2a_transpose_gemm(group, bs, nheads, seq, head_dim)
            sym.x.copy_(x_r)
            d_k = torch.zeros((bs * local_seq, N), dtype=torch.bfloat16, device=device)
            fn(d_k, Wo, sym)
            torch.cuda.synchronize()
            diff = (d_k.float() - d_gt.float()).abs()
            md = diff.max().item()
            rl = diff.mean().item() / (d_gt.float().abs().mean().item() + 1e-8)
            sym.destroy()
            dist.barrier()
            return md, rl

        md0, rl0 = run_and_diff(bf16_a2a_transpose_gemm_nt)            # default (M0)
        mdf, rlf = run_and_diff(bf16_a2a_transpose_gemm_nt_fused)      # fused (M1)
        max_diff = max(md0, mdf)
        rel = max(rl0, rlf)
        # bf16 GEMM rounding tolerance; both paths must pass
        passed = (rel_cand < 1e-3) and (rl0 < 0.02) and (rlf < 0.02)
        if rank == 0:
            print(f"  ({bs},{nheads},{seq},{head_dim},{N})".ljust(26) +
                  f" | {max_diff:>10.6f} {rel:>10.7f} | {'PASS' if passed else 'FAIL'}"
                  f"  (M0 rel={rl0:.1e}, fused rel={rlf:.1e})")
            num_passed += int(passed)
            if not passed:
                fails.append((bs, nheads, seq, head_dim, N))

    if rank == 0:
        print(f"\n  Summary: {num_passed}/{len(SHAPES)} passed" +
              ("  ALL PASS" if not fails else f"  FAILED: {fails}"))
    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    port = find_free_port()
    print(f"Launching A2A-transpose semantics-lock with {num_gpus} GPUs on port {port}...")
    mp.spawn(run_test, args=(num_gpus, port), nprocs=num_gpus, join=True)
