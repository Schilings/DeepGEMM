"""Wan2.1 14B Ulysses SP Attention Benchmark - FWD+BWD+SYNC (FSDP2-style grad sync).

Strategies:
  1. serial:     matmul + NCCL A2A (baseline, no overlap)
  2. fused_std:  GEMM+A2A (PRE) + A2A+GEMM (POST), fused kernels
  3. fused_var:  GEMM+A2A (PRE) + GEMM+RS (POST variant), fused kernels

Gradient sync (SYNC): FSDP2-style reduce-scatter via DTensor/reduce_scatter_tensor.
  - Wqkv: always reduce-scatter (replicated weight, partial grads)
  - Wo: reduce-scatter for serial/fused_std; SKIP for fused_var (Wo row-split, grad is local)

Usage: python benchmarks/bench_wan21_strategies.py <num_gpus> [iters] [--verify]
"""

import os, sys, math, argparse
import torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.config import Wan21Config, SPConfig, TrainConfig
from wan21.model import WanSelfAttention, build_wqkv_rankmajor
from wan21.bench_utils import find_free_port, time_call, rel_diff, gather_to_rank0
from wan21.grad_sync import sync_grads, sync_grads_all_reduce

WAN21_SHAPES = [
    (40, 8192,   128,  4, 16, 128, '1x8K'),
    (40, 32768,  128, 21, 30, 52,  '1x32K'),
    (40, 75776,  128, 21, 45, 80,  '1x74K'),
    (40, 172032, 128, 21, 67, 120, '1x168K'),
    (40, 65536,  128, 42, 30, 52,  '1x64K'),
    (40, 151552, 128, 42, 45, 80,  '1x148K'),
]

def get_strategy(name, cfg, sp_cfg):
    if name == 'serial':
        from wan21.sp.serial import SerialUlysses; return SerialUlysses(cfg, sp_cfg)
    elif name == 'fused_std':
        from wan21.sp.fused_standard import FusedStandardUlysses; return FusedStandardUlysses(cfg, sp_cfg)
    elif name == 'fused_var':
        from wan21.sp.fused_variant import FusedVariantUlysses; return FusedVariantUlysses(cfg, sp_cfg)
    raise ValueError(name)

def run(rank, ng, port, iters, verify, strategies):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd, scale = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim, model_cfg.scale

    if rank == 0:
        print(f"\n{'='*150}")
        print(f"  Wan2.1 14B Ulysses SP Attention Benchmark - {ng} GPUs, iters={iters}, THD")
        print(f"  Strategies: {', '.join(strategies)}")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng}")
        print(f"  SYNC = FSDP2-style reduce-scatter (Wqkv always; Wo unless row-sharded)")
        print(f"{'='*150}")
        print(f"{'shape':<10} {'strategy':<12} | {'FWD':>7} {'ATTN':>6} {'BWD':>7} {'SYNC':>6} {'F+B+S':>8} | {'fwd':>6} {'bX':>6} {'bW':>6} | {'st':>4}")
        print('-' * 150)

    for (nheads, seq, head_dim, gt, gh, gw, label) in WAN21_SHAPES:
        hidden = nheads * head_dim
        if nheads % ng or seq % ng or (seq // ng) % 128:
            if rank == 0: print(f"  {label} SKIP")
            dist.barrier(); continue

        bs = 1; llseq = seq // ng; lm = bs * llseq
        grid = torch.tensor([[gt, gh, gw]], dtype=torch.long)
        g2 = torch.Generator(device=dev).manual_seed(42)
        X_full = torch.randn(bs, seq, hidden, dtype=torch.bfloat16, device=dev, generator=g2)
        X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, hidden).contiguous()

        ref_out = ref_gX = ref_gW_parts = ref_grad_y = None
        do_verify = verify and '8K' in label

        for strat_name in strategies:
            sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
            try:
                strat = get_strategy(strat_name, model_cfg, sp_cfg).to(dev)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} SKIP ({e})")
                dist.barrier(); continue
            g = torch.Generator(device=dev).manual_seed(42)
            with torch.no_grad():
                for p in strat.model.parameters():
                    p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
            try:
                strat.setup_shape(bs, seq, nheads, head_dim)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} SKIP (setup: {e})")
                dist.barrier(); continue

            if do_verify and rank == 0 and ref_out is None:
                ref_m = WanSelfAttention(model_cfg, device=dev).to(dev)
                ref_m.load_state_dict(strat.model.state_dict())
                Xr = X_full.clone().requires_grad_(True)
                yr = ref_m(Xr, grid)
                torch.manual_seed(123)
                gy_full = torch.randn_like(yr)
                grads = torch.autograd.grad(yr, [Xr] + list(ref_m.parameters()), gy_full)
                ref_out = yr.detach(); ref_gX = grads[0].detach()
                ref_gW_parts = [grads[1], grads[2], grads[3]]; ref_grad_y = gy_full
                del Xr, yr, ref_m

            if do_verify:
                local_N_verify = hidden // ng if strat_name == 'fused_var' else hidden
                grad_y_full = ref_grad_y.clone() if rank == 0 else torch.empty(bs, seq, hidden, dtype=torch.bfloat16, device=dev)
                dist.broadcast(grad_y_full, src=0, group=group)
                if strat_name == 'fused_var':
                    grad_y_local = grad_y_full[:, rank*llseq:(rank+1)*llseq, rank*local_N_verify:(rank+1)*local_N_verify].reshape(llseq, local_N_verify).contiguous()
                else:
                    grad_y_local = grad_y_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, hidden).contiguous()
            else:
                grad_y_local = None

            resets = []
            if hasattr(strat, 'sym_post') and hasattr(strat.sym_post, 'reset_barriers'):
                resets = [strat.sym_post.reset_barriers]
            if hasattr(strat, 'sym_pre_bwd') and hasattr(strat.sym_pre_bwd, 'reset_barriers'):
                resets.append(strat.sym_pre_bwd.reset_barriers)

            # --- FWD timing ---
            def fwd():
                strat.layout = 'THD'
                return strat.forward(X_local, grid, llseq)
            try:
                t_fwd = time_call(fwd, iters, warmup=2, group=group, resets=resets)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} FWD FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

            # --- BWD timing (local grad compute, NO sync) ---
            def bwd():
                strat.forward(X_local, grid, llseq)
                if grad_y_local is not None:
                    gy = grad_y_local
                else:
                    local_N = hidden // ng if strat_name == 'fused_var' else hidden
                    gy = torch.randn((lm, local_N), dtype=torch.bfloat16, device=dev)
                gX, gWqkv, gWo = strat.backward(gy, X_local, grid, llseq)
                strat._last_grad_Wqkv = gWqkv
                strat._last_grad_Wo = gWo
                return gX
            try:
                t_bwd = time_call(bwd, iters, warmup=2, group=group, resets=resets)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} BWD FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

            # --- SYNC timing (FSDP2-style reduce-scatter) ---
            def sync():
                sync_grads(strat, group)
            try:
                t_sync = time_call(sync, iters, warmup=2, group=group)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} SYNC FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

            # --- ATTN timing ---
            local_nh = nheads // ng
            qb = torch.randn((bs, seq, local_nh, head_dim), dtype=torch.bfloat16, device=dev)
            kb = torch.randn_like(qb); vb = torch.randn_like(qb)
            def attn_only():
                from flash_attn.cute import flash_attn_func
                o = flash_attn_func(qb, kb, vb, softmax_scale=scale, causal=False)
                return o[0] if isinstance(o, tuple) else o
            t_attn = time_call(attn_only, iters, warmup=2, group=group)

            # --- Verify ---
            fwd_rel = bX_rel = bW_rel = -1.0; status = 'SKIP'
            if do_verify:
                if strat_name != 'fused_var':
                    if resets:
                        for r in resets: r()
                        dist.barrier(group)
                    with torch.no_grad():
                        y = strat.forward(X_local, grid, llseq)
                    y_full = gather_to_rank0(y, group, ng)
                    if rank == 0 and ref_out is not None:
                        fwd_rel = rel_diff(y_full.reshape(-1, hidden)[:ref_out.reshape(-1,hidden).shape[0]],
                                           ref_out.reshape(-1, hidden))
                # bwd + sync (all-reduce for verification, gives full grads)
                if resets:
                    for r in resets: r()
                    dist.barrier(group)
                strat.forward(X_local, grid, llseq)
                gX, gWqkv, gWo = strat.backward(grad_y_local, X_local, grid, llseq)
                strat._last_grad_Wqkv = gWqkv; strat._last_grad_Wo = gWo
                sync_grads_all_reduce(strat, group)  # all-reduce for full grad comparison
                gX_full = gather_to_rank0(gX, group, ng)
                if rank == 0 and ref_gX is not None:
                    bX_rel = rel_diff(gX_full[:ref_gX.reshape(-1,hidden).shape[0]].reshape(-1,hidden),
                                      ref_gX.reshape(-1, hidden))
                    if ref_gW_parts is not None and gWqkv is not None:
                        ref_gW = build_wqkv_rankmajor(ref_gW_parts[0], ref_gW_parts[1], ref_gW_parts[2],
                                                       ng, local_nh, head_dim)
                        bW_rel = rel_diff(gWqkv, ref_gW)
                status = 'PASS' if (fwd_rel < 0.05 and bX_rel < 0.05 and bW_rel < 0.05) else 'FAIL'

            if rank == 0:
                f_r = f"{fwd_rel:.4f}" if fwd_rel >= 0 else "  -   "
                x_r = f"{bX_rel:.4f}" if bX_rel >= 0 else "  -   "
                w_r = f"{bW_rel:.4f}" if bW_rel >= 0 else "  -   "
                print(f"{label:<10} {strat_name:<12} | {t_fwd:>7.0f} {t_attn:>6.0f} {t_bwd:>7.0f} {t_sync:>6.0f} {t_fwd+t_bwd+t_sync:>8.0f} | {f_r:>6} {x_r:>6} {w_r:>6} | {status:>4}")

            strat.destroy_buffers()
            dist.barrier()

    if rank == 0: print('=' * 150 + '\n')
    dist.destroy_process_group()
    os._exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('iters', type=int, nargs='?', default=10)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--strategies', default='serial,fused_std,fused_var')
    args = parser.parse_args()
    strategies = args.strategies.split(',')
    port = find_free_port()
    print(f"Launching: {args.num_gpus} GPUs, {args.iters} iters, strategies={strategies}, verify={args.verify}")
    mp.spawn(run, args=(args.num_gpus, port, args.iters, args.verify, strategies),
             nprocs=args.num_gpus, join=True)
