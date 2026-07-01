"""Measure peak GPU memory for each Ulysses SP strategy (serial vs fused_var).

Measures torch.cuda.max_memory_allocated() during FWD+BWD for various seq lengths.
"""

import os, sys, math, argparse
import torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.config import Wan21Config, SPConfig
from wan21.model import WanSelfAttention
from wan21.bench_utils import find_free_port
from wan21.fsdp2_utils import apply_fsdp2


WAN21_SHAPES = [
    (40, 8192,   128,  4, 16, 128, '1x8K'),
    (40, 32768,  128, 21, 30, 52,  '1x32K'),
    (40, 75776,  128, 21, 45, 80,  '1x74K'),
    (40, 172032, 128, 21, 67, 120, '1x168K'),
]


def get_strategy(name, cfg, sp_cfg):
    if name == 'serial':
        from wan21.sp.serial import SerialUlysses; return SerialUlysses(cfg, sp_cfg)
    elif name == 'fused_std':
        from wan21.sp.fused_standard import FusedStandardUlysses; return FusedStandardUlysses(cfg, sp_cfg)
    elif name == 'fused_var':
        from wan21.sp.fused_variant import FusedVariantUlysses; return FusedVariantUlysses(cfg, sp_cfg)
    raise ValueError(name)


def run(rank, ng, port, strategies):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim

    if rank == 0:
        print(f"\n{'='*110}")
        print(f"  Wan2.1 14B Peak Memory Comparison — {ng} GPUs")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng}")
        print(f"{'='*110}")
        print(f"{'shape':<10} {'strategy':<12} | {'FWD_peak(MB)':>12} {'BWD_peak(MB)':>12} {'sym_buf(MB)':>12} | {'Wo_wt(MB)':>10} {'Wo_grad(MB)':>11}")
        print('-' * 110)

    for (nheads, seq, head_dim, gt, gh, gw, label) in WAN21_SHAPES:
        if nheads % ng or seq % ng or (seq // ng) % 128:
            if rank == 0: print(f"  {label} SKIP")
            dist.barrier(); continue

        bs = 1; llseq = seq // ng; lm = bs * llseq
        grid = torch.tensor([[gt, gh, gw]], dtype=torch.long)
        g2 = torch.Generator(device=dev).manual_seed(42)
        X_full = torch.randn(bs, seq, hidden := nheads * head_dim, dtype=torch.bfloat16, device=dev, generator=g2)
        X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, hidden).contiguous()

        for strat_name in strategies:
            sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
            strat = get_strategy(strat_name, model_cfg, sp_cfg).to(dev)
            g = torch.Generator(device=dev).manual_seed(42)
            with torch.no_grad():
                for p in strat.model.parameters():
                    p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
            strat.model = strat.model.to(torch.bfloat16)
            strat.setup_shape(bs, seq, nheads, head_dim)

            ignored = set(strat.model.parameters())
            if strat_name == 'fused_var':
                ignored |= {strat.Wo_r_local, strat.Wo_r_local_t}
            apply_fsdp2(strat, group, reshard_after_forward=False, ignored_params=ignored)

            # Measure sym buffer size
            sym_mb = 0.0
            if hasattr(strat, 'sym_post') and strat.sym_post is not None:
                sym_mb += strat.sym_post.buffer.numel() / 1024 / 1024
            if hasattr(strat, 'sym_post_bwd') and strat.sym_post_bwd is not None:
                sym_mb += strat.sym_post_bwd.buffer.numel() / 1024 / 1024

            # Measure Wo weight size
            wo_mb = 0.0
            if strat_name == 'fused_var':
                wo_mb = strat.Wo_r_local.numel() * 2 / 1024 / 1024  # bf16=2 bytes
            else:
                wo_mb = strat.Wo.numel() * 2 / 1024 / 1024

            # FWD peak
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            X_in = X_local.detach().requires_grad_(True)
            y = strat(X_in, grid, llseq)
            torch.cuda.synchronize(dev)
            fwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

            # BWD peak (cumulative from FWD)
            gy = torch.randn((lm, hidden), dtype=torch.bfloat16, device=dev)
            y.backward(gy)
            torch.cuda.synchronize(dev)
            bwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

            # Wo grad size (if exists)
            wo_grad_mb = 0.0
            if strat_name == 'fused_var':
                if strat.Wo_r_local.grad is not None:
                    wo_grad_mb = strat.Wo_r_local.grad.numel() * 2 / 1024 / 1024
            else:
                wo_grad_mb = hidden * hidden * 2 / 1024 / 1024  # [dim, dim]

            if rank == 0:
                print(f"{label:<10} {strat_name:<12} | {fwd_peak:>12.1f} {bwd_peak:>12.1f} {sym_mb:>12.1f} | {wo_mb:>10.1f} {wo_grad_mb:>11.1f}")

            strat.destroy_buffers()
            del strat, X_in, y, gy
            torch.cuda.empty_cache()
            dist.barrier()

    if rank == 0: print('=' * 110 + '\n')
    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    strategies = sys.argv[2].split(',') if len(sys.argv) > 2 else 'serial,fused_var'.split(',')
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    port = find_free_port()
    print(f"Launching: {ng} GPUs, strategies={strategies}")
    mp.spawn(run, args=(ng, port, strategies), nprocs=ng, join=True)
