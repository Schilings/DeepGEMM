"""Debug: compare serial vs fused_var grad_X."""
import os, sys, math, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR':'127.0.0.1','MASTER_PORT':str(port),'RANK':str(rank),'WORLD_SIZE':str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')
    from wan21.config import Wan21Config, SPConfig
    from wan21.model import WanSelfAttention
    from wan21.sp.serial import SerialUlysses
    from wan21.sp.fused_variant import FusedVariantUlysses
    cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim = cfg.dim
    s_cfg = SPConfig(sp_size=ng, group=dist.group.WORLD, layout='THD')
    serial = SerialUlysses(cfg, s_cfg).to(dev).to(torch.bfloat16)
    var = FusedVariantUlysses(cfg, s_cfg).to(dev).to(torch.bfloat16)
    g = torch.Generator(device=dev).manual_seed(42)
    with torch.no_grad():
        for p in serial.model.parameters():
            p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
    var.model.load_state_dict(serial.model.state_dict())
    serial.setup_shape(1, 8192, 40, 128)
    var.setup_shape(1, 8192, 40, 128)
    bs, seq = 1, 8192
    llseq = seq // ng
    grid = torch.tensor([[4, 16, 128]], dtype=torch.long)
    X = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=dev, generator=torch.Generator(device=dev).manual_seed(42))
    X_local = X[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, dim).contiguous()
    torch.manual_seed(123)
    gy = torch.randn(llseq, dim, dtype=torch.bfloat16, device=dev)
    # serial bwd
    Xs = X_local.clone().requires_grad_(True)
    ys = serial(Xs, grid, llseq)
    ys.backward(gy)
    # var bwd
    Xv = X_local.clone().requires_grad_(True)
    yv = var(Xv, grid, llseq)
    yv.backward(gy)
    rel = (Xs.grad.float() - Xv.grad.float()).norm().item() / (Xs.grad.float().norm().item() + 1e-12)
    print(f'[r{rank}] grad_X rel (serial vs var): {rel:.6f}', flush=True)
    with torch.no_grad():
        ys2 = serial(X_local, grid, llseq)
        yv2 = var(X_local, grid, llseq)
        frel = (ys2.float() - yv2.float()).norm().item() / (ys2.float().norm().item() + 1e-12)
    print(f'[r{rank}] fwd rel (serial vs var): {frel:.6f}', flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    import socket
    s = socket.socket(); s.bind(('',0)); port = s.getsockname()[1]; s.close()
    mp.spawn(run, args=(2, port), nprocs=2, join=True)
