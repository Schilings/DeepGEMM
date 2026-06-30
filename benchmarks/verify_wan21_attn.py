"""Quick correctness verification for Wan2.1 SP attention (small config, 2 GPUs)."""
import os, sys, math
import torch, torch.distributed as dist, torch.multiprocessing as mp

def run(rank):
    os.environ["RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    dev = torch.device(f"cuda:{rank}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from wan21.config import Wan21Config, SPConfig
    from wan21.sp.standard import UlyssesStandardAttention
    from wan21.model import WanSelfAttention, build_wqkv_rankmajor
    from wan21.bench_utils import rel_diff, gather_to_rank0

    cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    sp = SPConfig(sp_size=2, group=dist.group.WORLD, layout="THD", use_fused_ops=True)
    attn = UlyssesStandardAttention(cfg, sp).to(dev)
    g = torch.Generator(device=dev).manual_seed(42)
    with torch.no_grad():
        for p in attn.model.parameters():
            p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(cfg.dim)
    bs, seq, hidden = 1, 8192, cfg.dim
    attn.setup_shape(bs, seq, 40, 128)
    grid = torch.tensor([[4, 16, 128]], dtype=torch.long)
    g2 = torch.Generator(device=dev).manual_seed(42)
    X_full = torch.randn(bs, seq, hidden, dtype=torch.bfloat16, device=dev, generator=g2)
    llseq = seq // 2
    X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, hidden).contiguous()

    # Reference: rank 0 computes full single-GPU FWD+BWD
    ref_out = ref_gX = ref_gW = ref_grad_y = None
    if rank == 0:
        m = WanSelfAttention(cfg, device=dev).to(dev)
        m.load_state_dict(attn.model.state_dict())
        xr = X_full.clone().requires_grad_(True)
        yr = m(xr, grid)
        torch.manual_seed(123)
        gy_full = torch.randn_like(yr)  # [bs, seq, dim]
        grads = torch.autograd.grad(yr, [xr] + list(m.parameters()), gy_full)
        ref_out = yr.detach()
        ref_gX = grads[0].detach()
        ref_gW = [grads[1], grads[2], grads[3]]
        ref_grad_y = gy_full

    # Broadcast grad_y to all ranks (rank 0 has the full grad_y)
    grad_y_full = ref_grad_y if rank == 0 else torch.empty(bs, seq, hidden, dtype=torch.bfloat16, device=dev)
    dist.broadcast(grad_y_full, src=0, group=dist.group.WORLD)

    # SP forward
    y = attn.forward(X_local, grid, llseq)

    # FWD correctness: gather SP output → compare with ref
    y_full = gather_to_rank0(y, dist.group.WORLD, 2)
    fwd_rel = 0.0
    if rank == 0 and ref_out is not None:
        fwd_rel = rel_diff(y_full.reshape(-1, hidden)[:ref_out.reshape(-1, hidden).shape[0]],
                           ref_out.reshape(-1, hidden))

    # Extract local grad_y shard for this rank
    grad_y_local = grad_y_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, hidden).contiguous()

    # BWD
    o_cache = attn._attn_forward(attn._pre_forward(X_local, llseq), grid, 1, seq)
    gX, gWqkv, gWo = attn.backward(grad_y_local, X_local, grid, cache=o_cache, llseq=llseq)
    # Note: backward uses all-gather internally, so weight grads are already full (same on all ranks).
    # No all-reduce needed for this path. For the A2A-inverse path, all-reduce WOULD be needed.
    gX_full = gather_to_rank0(gX, dist.group.WORLD, 2)
    bX_rel = bW_rel = 0.0
    if rank == 0:
        bX_rel = rel_diff(gX_full[:ref_gX.shape[0]].reshape(-1, hidden),
                          ref_gX.reshape(-1, hidden))
        ref_gWqkv = build_wqkv_rankmajor(ref_gW[0], ref_gW[1], ref_gW[2], 2, 20, 128)
        bW_rel = rel_diff(gWqkv, ref_gWqkv)

    if rank == 0:
        print(f"\n  FWD rel = {fwd_rel:.6f}")
        print(f"  BWD grad_X rel = {bX_rel:.6f}")
        print(f"  BWD grad_Wqkv rel = {bW_rel:.6f}")
        ok = fwd_rel < 0.05 and bX_rel < 0.05 and bW_rel < 0.05
        print(f"  Status: {'PASS ✓' if ok else 'FAIL ✗'}\n")
    dist.destroy_process_group()
    os._exit(0)

if __name__ == "__main__":
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29505"})
    mp.spawn(run, nprocs=2, join=True)
