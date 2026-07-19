"""Debug fwd_rel: compare SP forward output vs reference, step by step."""
import os, sys, math, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),'RANK':str(rank),'WORLD_SIZE':str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    from wan21.config import Wan21Config, SPConfig
    from wan21.model import WanSelfAttention, build_wqkv_rankmajor
    from wan21.bench_utils import rel_diff, gather_to_rank0
    from wan21.sp.serial import SerialUlysses

    cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = cfg.dim, cfg.num_heads, cfg.head_dim
    sp_cfg = SPConfig(sp_size=ng, group=dist.group.WORLD, layout='THD')
    strat = SerialUlysses(cfg, sp_cfg).to(dev).to(torch.bfloat16)
    g = torch.Generator(device=dev).manual_seed(42)
    with torch.no_grad():
        for p in strat.model.parameters():
            p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
    strat.setup_shape(1, 8192, 40, 128)

    bs, seq = 1, 8192
    llseq = seq // ng
    grid = torch.tensor([[4, 16, 128]], dtype=torch.long)
    X_full = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=dev, generator=torch.Generator(device=dev).manual_seed(42))
    X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, dim).contiguous()

    # Reference: full single-GPU forward
    if rank == 0:
        ref_m = WanSelfAttention(dim, nh, hd, qk_norm=True, eps=1e-6).to(dev).to(torch.bfloat16)
        ref_m.load_state_dict(strat.model.state_dict())
        Xr = X_full.clone()
        with torch.no_grad():
            yr = ref_m(Xr, grid, ref_m.freqs)
        print(f"ref output shape: {list(yr.shape)}, norm={yr.float().norm().item():.4f}", flush=True)

    dist.barrier()

    # SP forward (all ranks)
    with torch.no_grad():
        y_sp = strat(X_local, grid, llseq)
    if rank == 0:
        print(f"SP output shape: {list(y_sp.shape)}, norm={y_sp.float().norm().item():.4f}", flush=True)
        ref_shard = yr[0, :llseq, :]
        rel = rel_diff(y_sp[:llseq, :dim], ref_shard.reshape(-1, dim))
        print(f"fwd_rel (rank0 shard): {rel:.6f}", flush=True)
        print(f"SP y[0,:5]={y_sp[0,:5].float().tolist()}", flush=True)
        print(f"ref y[0,:5]={ref_shard[0,:5].float().tolist()}", flush=True)

    dist.destroy_process_group()
    os._exit(0)

if __name__ == '__main__':
    import socket
    s = socket.socket(); s.bind(('',0)); port = s.getsockname()[1]; s.close()
    mp.spawn(run, args=(2, port), nprocs=2, join=True)
