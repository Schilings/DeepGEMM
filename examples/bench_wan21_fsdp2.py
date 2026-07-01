"""Wan2.1 14B Ulysses SP Attention Benchmark with REAL FSDP2 (fully_shard).

Autograd-based forward + FSDP2 automatic gradient sync:
  - Forward: autograd graph (fused ops as torch.autograd.Function)
  - Backward: torch.autograd.backward() — FSDP2 hooks auto reduce-scatter weight grads
  - FSDP2: from torch.distributed.fsdp import fully_shard (composable API, DTensor-based)

Strategies:
  1. serial:     matmul + NCCL A2A (autograd.Function), baseline
  2. fused_std:  GEMM+A2A (Function) + A2A+GEMM (Function), fused kernels
  3. fused_var:  GEMM+A2A (Function) + GEMM+RS (Function), Wo row-split

FSDP2 wrapping:
  - Wqkv (nn.Parameter, replicated): FSDP2 shards + auto reduce-scatter grad
  - Wo: serial/fused_std → FSDP2; fused_var → ignored_params (row-split, no sync)

Usage: python benchmarks/bench_wan21_fsdp2.py <num_gpus> [iters] [--verify]
"""

import os, sys, math, argparse
import torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.config import Wan21Config, SPConfig
from wan21.model import WanSelfAttention, build_wqkv_rankmajor
from wan21.bench_utils import find_free_port, time_call, rel_diff, gather_to_rank0
from wan21.fsdp2_utils import apply_fsdp2

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
        print(f"  Wan2.1 14B Ulysses SP Attention Benchmark with FSDP2 (fully_shard) - {ng} GPUs, iters={iters}, THD")
        print(f"  Strategies: {', '.join(strategies)}")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng}")
        print(f"  Autograd-based: fused ops as torch.autograd.Function, FSDP2 auto reduce-scatter in backward")
        print(f"{'='*150}")
        print(f"{'shape':<10} {'strategy':<12} | {'FWD':>7} {'BWD':>7} {'F+B':>8} | {'fwd':>6} {'bX':>6} | {'st':>4}")
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

        ref_out = ref_gX = ref_grad_y = None
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
                strat.model = strat.model.to(torch.bfloat16)
                strat.setup_shape(bs, seq, nheads, head_dim)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} SKIP (setup: {e})")
                dist.barrier(); continue

            # Build reference BEFORE FSDP2 (it converts params to DTensor)
            if do_verify and rank == 0 and ref_out is None:
                ref_m = WanSelfAttention(model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim,
                                         qk_norm=model_cfg.qk_norm, eps=model_cfg.eps).to(dev).to(torch.bfloat16)
                ref_m.load_state_dict(strat.model.state_dict())
                Xr = X_full.clone().requires_grad_(True)
                yr = ref_m(Xr, grid, ref_m.freqs)
                torch.manual_seed(123)
                gy_full = torch.randn_like(yr)
                grads = torch.autograd.grad(yr, [Xr] + list(ref_m.parameters()), gy_full)
                ref_out = yr.detach(); ref_gX = grads[0].detach(); ref_grad_y = gy_full
                del Xr, yr, ref_m

            # Apply FSDP2: shard Wqkv Parameter; ignore model params (we use Wqkv not nn.Linear)
            ignored = set(strat.model.parameters())
            if strat_name == 'fused_var':
                ignored |= {strat.Wo_r_local, strat.Wo_r_local_t}
            try:
                apply_fsdp2(strat, group, reshard_after_forward=True, ignored_params=ignored)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} FSDP2 FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

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
                resets.append(strat.sym_post.reset_barriers)
            if hasattr(strat, 'sym_pre_bwd') and hasattr(strat.sym_pre_bwd, 'reset_barriers'):
                resets.append(strat.sym_pre_bwd.reset_barriers)

            # --- FWD timing (autograd) ---
            def fwd():
                for r in resets: r()
                strat.layout = 'THD'
                return strat(X_local.detach().requires_grad_(True), grid, llseq)
            try:
                t_fwd = time_call(fwd, iters, warmup=2, group=group)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} FWD FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

            # --- BWD timing (autograd backward, FSDP2 auto reduce-scatter) ---
            def bwd():
                for r in resets: r()
                strat.layout = 'THD'
                X_in = X_local.detach().requires_grad_(True)
                y = strat.forward(X_in, grid, llseq)
                if grad_y_local is not None:
                    gy = grad_y_local
                else:
                    local_N = hidden // ng if strat_name == 'fused_var' else hidden
                    gy = torch.randn((lm, local_N), dtype=torch.bfloat16, device=dev)
                # Autograd backward — FSDP2 hooks reduce-scatter weight grads automatically
                y.backward(gy)
                return X_in.grad
            try:
                t_bwd = time_call(bwd, iters, warmup=2, group=group)
            except Exception as e:
                if rank == 0: print(f"  {label:<12} {strat_name:<12} BWD FAIL: {e}")
                strat.destroy_buffers(); dist.barrier(); continue

            # --- Verify ---
            fwd_rel = bX_rel = -1.0; status = 'SKIP'
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
                # bwd verify
                if resets:
                    for r in resets: r()
                    dist.barrier(group)
                Xv = X_local.detach().requires_grad_(True)
                strat.layout = 'THD'
                yv = strat(Xv, grid, llseq)
                torch.autograd.backward(yv, grad_y_local)
                gX = Xv.grad
                gX_full = gather_to_rank0(gX, group, ng)
                if rank == 0 and ref_gX is not None:
                    bX_rel = rel_diff(gX_full[:ref_gX.reshape(-1,hidden).shape[0]].reshape(-1,hidden),
                                      ref_gX.reshape(-1, hidden))
                status = 'PASS' if (fwd_rel < 0.05 and bX_rel < 0.05) else 'FAIL'

            if rank == 0:
                f_r = f"{fwd_rel:.4f}" if fwd_rel >= 0 else "  -   "
                x_r = f"{bX_rel:.4f}" if bX_rel >= 0 else "  -   "
                print(f"{label:<10} {strat_name:<12} | {t_fwd:>7.0f} {t_bwd:>7.0f} {t_fwd+t_bwd:>8.0f} | {f_r:>6} {x_r:>6} | {status:>4}")

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
