"""
Wan2.1 14B Ulysses SP Attention Benchmark (FWD + BWD) — THD only (PackedSequence).

Usage:
  python benchmarks/bench_wan21_ulysses_attn.py <num_gpus> [iters] [--variant] [--verify]
  --verify  : correctness check on small shape only, then large shapes timing only
  --variant : GEMM+RS POST instead of standard A2A+GEMM
"""

import os, sys, math, argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from wan21.config import Wan21Config, SPConfig, TrainConfig
from wan21.model import WanSelfAttention, build_wqkv_rankmajor
from wan21.bench_utils import find_free_port, time_call, rel_diff, gather_to_rank0
from wan21.fsdp2_utils import wrap_fsdp2

# THD shapes: bs=1 always (packed sequence). (nheads, seq, head_dim, gt, gh, gw, label)
WAN21_SHAPES = [
    (40, 8192,  128,  4, 16, 128, '8K verify'),
    (40, 32768, 128, 21, 30, 52,  '480p 32K'),
    (40, 75776, 128, 21, 45, 80,  '720p 74K'),
]


def run(rank, ng, port, iters, use_variant, use_fsdp2, verify):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')

    import deep_gemm

    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    train_cfg = TrainConfig(use_fsdp2=use_fsdp2)
    dim, nh, hd, scale = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim, model_cfg.scale

    if use_variant:
        from wan21.sp.variant import UlyssesVariantAttention as StrategyClass
        strat_name = 'VARIANT(GEMM+RS)'
    else:
        from wan21.sp.standard import UlyssesStandardAttention as StrategyClass
        strat_name = 'STANDARD(A2A+GEMM)'
    from wan21.sp.torch_baseline import TorchUlyssesAttention

    if rank == 0:
        print(f"\n{'='*120}")
        print(f"  Wan2.1 14B Ulysses SP Attention Benchmark — {ng} GPUs, iters={iters}, THD")
        print(f"  Strategy: {strat_name} | FSDP2: {'YES' if use_fsdp2 else 'NO'}")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng}")
        print(f"{'='*120}")
        print(f"{'shape':<14} | {'FWD f/t':>12} {'ATTN':>6} {'BWD f/t':>12} | {'fwd':>6} {'bX':>6} {'bW':>6} | {'st':>4}")
        print('-' * 120)

    for (nheads, seq, head_dim, gt, gh, gw, label) in WAN21_SHAPES:
        hidden = nheads * head_dim
        if nheads % ng or seq % ng or (seq // ng) % 128:
            if rank == 0: print(f"  {label:<12} SKIP")
            dist.barrier(); continue
        if use_variant and (hidden % ng or (hidden // ng) % 128):
            if rank == 0: print(f"  {label:<12} SKIP (variant)")
            dist.barrier(); continue

        bs = 1
        local_seq = seq // ng
        local_m = bs * local_seq
        grid_sizes = torch.tensor([[gt, gh, gw]], dtype=torch.long)

        sp_cfg_f = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True,
                            post_strategy='gemm_rs' if use_variant else 'a2a_gemm')
        sp_cfg_t = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=False)
        attn_fused = StrategyClass(model_cfg, sp_cfg_f).to(dev)
        attn_torch = TorchUlyssesAttention(model_cfg, sp_cfg_t).to(dev)

        g = torch.Generator(device=dev).manual_seed(42)
        with torch.no_grad():
            for m in [attn_fused.model, attn_torch.model]:
                for p in m.parameters():
                    p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
            attn_torch.model.load_state_dict(attn_fused.model.state_dict())

        attn_fused.setup_shape(bs, seq, nheads, head_dim)
        attn_torch.setup_shape(bs, seq, nheads, head_dim)

        if use_fsdp2:
            attn_fused = wrap_fsdp2(attn_fused, train_cfg)
            attn_torch = wrap_fsdp2(attn_torch, train_cfg)

        g2 = torch.Generator(device=dev).manual_seed(42)
        X_full = torch.randn((bs, seq, hidden), dtype=torch.bfloat16, device=dev, generator=g2)
        X_local = X_full[:, rank * local_seq:(rank + 1) * local_seq, :].reshape(local_m, hidden).contiguous()

        lbs, lseq, llseq, lm = 1, seq, local_seq, local_m

        # REFERENCE (verify + small shape only)
        ref_out = ref_grad_X = ref_grad_Wqkv_parts = None
        do_verify = verify and 'verify' in label
        if do_verify and rank == 0:
            ref_model = WanSelfAttention(model_cfg, device=dev).to(dev)
            ref_model.load_state_dict(attn_fused.model.state_dict())
            Xr = X_full.clone().requires_grad_(True)
            yr = ref_model(Xr, grid_sizes)
            grad_y_ref = torch.randn_like(yr)
            grads = torch.autograd.grad(yr, [Xr] + list(ref_model.parameters()), grad_y_ref)
            ref_out = yr.detach()
            ref_grad_X = grads[0].detach()
            ref_grad_Wqkv_parts = [grads[1], grads[2], grads[3]]
            del Xr, yr, ref_model

        def fwd_fused():
            return attn_fused.forward(X_local, grid_sizes, llseq)
        def fwd_torch():
            return attn_torch.forward(X_local, grid_sizes, llseq)

        resets_f = []
        if hasattr(attn_fused, 'sym_post') and hasattr(attn_fused.sym_post, 'reset_barriers'):
            resets_f = [attn_fused.sym_post.reset_barriers]

        t_fwd_f = time_call(fwd_fused, iters, warmup=2, group=group, resets=resets_f)
        t_fwd_t = time_call(fwd_torch, iters, warmup=2, group=group)

        def fwd_fused_cache():
            qkv = attn_fused._pre_forward(X_local, llseq)
            o = attn_fused._attn_forward(qkv, grid_sizes, lbs, lseq)
            y = attn_fused._post_forward(o, lbs=lbs, lseq=lseq, llseq=llseq, lm=lm, grid_sizes=grid_sizes)
            return y, o

        def bwd_fused_timed():
            y, o_cache = fwd_fused_cache()
            torch.manual_seed(train_cfg.grad_seed)
            grad_y = torch.randn_like(y)
            return attn_fused.backward(grad_y, X_local, grid_sizes, cache=o_cache, llseq=llseq)

        def bwd_torch_timed():
            y = fwd_torch()
            torch.manual_seed(train_cfg.grad_seed)
            grad_y = torch.randn_like(y)
            qkv = attn_torch._pre_forward(X_local, llseq)
            o_cache = attn_torch._attn_forward(qkv, grid_sizes, lbs, lseq)
            return attn_torch.backward(grad_y, X_local, grid_sizes, cache=o_cache, llseq=llseq)

        t_bwd_f = time_call(bwd_fused_timed, iters, warmup=2, group=group, resets=resets_f)
        t_bwd_t = time_call(bwd_torch_timed, iters, warmup=2, group=group)

        local_nh = nheads // ng
        qb = torch.randn((bs, seq, local_nh, head_dim), dtype=torch.bfloat16, device=dev)
        kb = torch.randn_like(qb); vb = torch.randn_like(qb)
        def attn_only():
            from flash_attn.cute import flash_attn_func
            o = flash_attn_func(qb, kb, vb, softmax_scale=scale, causal=False)
            return o[0] if isinstance(o, tuple) else o
        t_attn = time_call(attn_only, iters, warmup=2, group=group)

        fwd_rel = bX_rel = bW_rel = -1.0
        status = 'SKIP'
        if do_verify:
            with torch.no_grad():
                y_fused = fwd_fused()
            y_full = gather_to_rank0(y_fused, group, ng)
            if rank == 0 and ref_out is not None:
                ref_flat = ref_out.reshape(-1, hidden)
                y_flat = y_full.reshape(-1, hidden)[:ref_flat.shape[0]]
                fwd_rel = rel_diff(y_flat, ref_flat)

            torch.manual_seed(train_cfg.grad_seed)
            y_test, o_test = fwd_fused_cache()
            grad_y_test = torch.randn_like(y_test)
            gX, gWqkv, gWo = attn_fused.backward(grad_y_test, X_local, grid_sizes, cache=o_test, llseq=llseq)
            if gWqkv is not None: dist.all_reduce(gWqkv, op=dist.ReduceOp.SUM, group=group)
            if gWo is not None: dist.all_reduce(gWo, op=dist.ReduceOp.SUM, group=group)
            gX_full = gather_to_rank0(gX, group, ng)
            if rank == 0 and ref_grad_X is not None:
                ref_X = ref_grad_X.reshape(-1, hidden)
                bX_rel = rel_diff(gX_full[:ref_X.shape[0]], ref_X)
                if ref_grad_Wqkv_parts is not None and gWqkv is not None:
                    ref_gW = build_wqkv_rankmajor(ref_grad_Wqkv_parts[0], ref_grad_Wqkv_parts[1],
                                                  ref_grad_Wqkv_parts[2], ng, local_nh, head_dim)
                    bW_rel = rel_diff(gWqkv, ref_gW)
            status = 'PASS' if (fwd_rel < 0.05 and bX_rel < 0.05 and bW_rel < 0.05) else 'FAIL'

        if rank == 0:
            fwd_r = f"{fwd_rel:.4f}" if fwd_rel >= 0 else "  -   "
            bX_r = f"{bX_rel:.4f}" if bX_rel >= 0 else "  -   "
            bW_r = f"{bW_rel:.4f}" if bW_rel >= 0 else "  -   "
            print(f"{label:<14} | {t_fwd_f:>5.0f}/{t_fwd_t:<5.0f} {t_attn:>6.0f} "
                  f"{t_bwd_f:>5.0f}/{t_bwd_t:<5.0f} | {fwd_r:>6} {bX_r:>6} {bW_r:>6} | {status:>4}")

        dist.barrier()
        attn_fused.destroy_buffers()
        attn_torch.destroy_buffers()

    if rank == 0:
        print('=' * 120 + '\n')
    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wan2.1 14B Ulysses SP Attention Benchmark (THD)')
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('iters', type=int, nargs='?', default=10)
    parser.add_argument('--variant', action='store_true', help='GEMM+RS POST')
    parser.add_argument('--fsdp2', action='store_true', help='FSDP2 grad sync')
    parser.add_argument('--verify', action='store_true', help='Correctness check on small shape')
    args = parser.parse_args()

    port = find_free_port()
    print(f"Launching: {args.num_gpus} GPUs, {args.iters} iters, THD, variant={args.variant}, verify={args.verify}")
    mp.spawn(run, args=(args.num_gpus, port, args.iters, args.variant, args.fsdp2, args.verify),
             nprocs=args.num_gpus, join=True)
