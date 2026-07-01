"""Track memory delta at each step of forward+backward for a single layer."""
import os, sys, math
import torch, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.config import Wan21Config, SPConfig
from wan21.fsdp2_utils import apply_fsdp2
from wan21.bench_utils import find_free_port


def get_strategy(name, cfg, sp_cfg):
    if name == 'serial':
        from wan21.sp.serial import SerialUlysses; return SerialUlysses(cfg, sp_cfg)
    elif name == 'fused_var':
        from wan21.sp.fused_variant import FusedVariantUlysses; return FusedVariantUlysses(cfg, sp_cfg)
    raise ValueError(name)


def mem_mb(dev):
    torch.cuda.synchronize(dev)
    return torch.cuda.memory_allocated(dev) / 1024 / 1024


def run(rank, ng, port, seq):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim

    grid = torch.tensor([[21, 30, 52]], dtype=torch.long)
    bs = 1; llseq = seq // ng; lm = bs * llseq
    g2 = torch.Generator(device=dev).manual_seed(42)
    X_full = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=dev, generator=g2)
    X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, dim).contiguous()

    for strat_name in ['serial', 'fused_var']:
        sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
        strat = get_strategy(strat_name, model_cfg, sp_cfg).to(dev)
        g = torch.Generator(device=dev).manual_seed(42)
        with torch.no_grad():
            for p in strat.model.parameters():
                p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
        strat.model = strat.model.to(torch.bfloat16)
        strat.setup_shape(bs, seq, nh, hd)

        ignored = set(strat.model.parameters())
        if strat_name == 'fused_var':
            ignored |= {strat.Wo_r_local}
        apply_fsdp2(strat, group, reshard_after_forward=False, ignored_params=ignored)

        sym_mb = 0.0
        if strat_name == 'fused_var':
            sym_mb = (strat.sym_post.buffer.numel() + strat.sym_post_bwd.buffer.numel()) / 1024 / 1024

        if rank == 0:
            print(f"\n=== {strat_name} (sym_buf={sym_mb:.0f} MB) ===")

        m0 = mem_mb(dev)
        if rank == 0: print(f"  baseline:        {m0:>10.1f} MB")

        X_in = X_local.detach().requires_grad_(True)
        m1 = mem_mb(dev)
        if rank == 0: print(f"  after X_in:      {m1:>10.1f} MB  (delta={m1-m0:+.1f})")

        y = strat(X_in, grid, llseq)
        m2 = mem_mb(dev)
        if rank == 0: print(f"  after fwd:       {m2:>10.1f} MB  (delta={m2-m1:+.1f})")

        gy = torch.randn((lm, dim), dtype=torch.bfloat16, device=dev)
        m3 = mem_mb(dev)
        if rank == 0: print(f"  after gy:        {m3:>10.1f} MB  (delta={m3-m2:+.1f})")

        y.backward(gy)
        m4 = mem_mb(dev)
        if rank == 0: print(f"  after bwd:       {m4:>10.1f} MB  (delta={m4-m3:+.1f})")
        if rank == 0: print(f"  total (excl sym_buf): {m4:>10.1f} MB")
        if rank == 0: print(f"  total (incl sym_buf): {m4 + sym_mb:>10.1f} MB")

        strat.destroy_buffers()
        del strat, X_in, y, gy
        torch.cuda.empty_cache()
        dist.barrier()

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
    port = find_free_port()
    mp.spawn(run, args=(ng, port, seq), nprocs=ng, join=True)
