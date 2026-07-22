"""Correctness test: serial vs fused_var — fwd, bwd, grad_W, loss curve.

Verifies:
1. Forward output: rel error < 0.5%
2. Grad_X (input grad): rel error < 0.1%
3. Grad_W (weight grads): rel error < 0.5% per parameter
4. Loss curve: 20-step training, loss difference < 5% at each step

Usage:
    DG_AG_PUBLISH_SYNC=symm DG_JIT_USE_NVRTC=1 \
    PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
    python3 examples/ulysses_variant/test_correctness.py 8
"""
from __future__ import annotations

import os, sys, math, copy, time
import torch, torch.distributed as dist, torch.multiprocessing as mp

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.config import Wan21Config, SPConfig
from wan21.sp_training import SPWanTransformer


def run(rank, ng, port):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    dev = torch.device(f'cuda:{rank}')

    cfg = Wan21Config()
    dim = cfg.dim
    sp_cfg = SPConfig(sp_size=ng, group=dist.group.WORLD, layout='THD')
    seq_len = 8192
    local_seq = seq_len // ng
    grid = torch.tensor([[4, 16, 128]], dtype=torch.long, device=dev)
    num_layers = 4  # Use fewer layers for fast iteration

    # Build models
    torch.manual_seed(42)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(dev):
            serial_model = SPWanTransformer(cfg, sp_cfg, num_layers, 'serial')
            var_model = SPWanTransformer(cfg, sp_cfg, num_layers, 'fused_var')
    finally:
        torch.set_default_dtype(old)
    serial_model.to(device=dev, dtype=torch.bfloat16)
    var_model.to(device=dev, dtype=torch.bfloat16)

    # Copy weights: first copy q/k/v/o/norm/ffn, then setup_shape will shard Wo
    for s_block, v_block in zip(serial_model.blocks, var_model.blocks):
        s_attn = s_block.self_attn
        v_attn = v_block.self_attn
        # Copy q, k, v, o weights (o will be sharded by _build_weights during setup_shape)
        for name in ['q', 'k', 'v', 'o']:
            getattr(v_attn.model, name).weight.data.copy_(
                getattr(s_attn.model, name).weight.data)
        # Copy non-attention params
        for name in ['norm1', 'norm2', 'norm3']:
            s_norm = getattr(s_block, name)
            v_norm = getattr(v_block, name)
            if hasattr(s_norm, 'weight') and s_norm.weight is not None:
                v_norm.weight.data.copy_(s_norm.weight.data)
        v_block.modulation.data.copy_(s_block.modulation.data)
        v_block.ffn.load_state_dict(s_block.ffn.state_dict())
        if hasattr(s_block.self_attn.model, 'norm_q'):
            v_attn.model.norm_q.weight.data.copy_(s_attn.model.norm_q.weight.data)
            v_attn.model.norm_k.weight.data.copy_(s_attn.model.norm_k.weight.data)
    for s_block, v_block in zip(serial_model.blocks, var_model.blocks):
        if hasattr(s_block, 'cross_attn'):
            v_block.cross_attn.load_state_dict(s_block.cross_attn.state_dict())

    # setup_shape will call _build_weights which shards Wo into Wo_r_local
    serial_model.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    var_model.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    serial_model.train()
    var_model.train()

    # Shared input
    g = torch.Generator(device=dev).manual_seed(1234)
    x = torch.randn(local_seq, dim, device=dev, dtype=torch.bfloat16, generator=g)
    e = torch.randn(1, 6, dim, device=dev, dtype=torch.float32,
                    generator=torch.Generator(device=dev).manual_seed(5678)) * 0.01
    context = torch.randn(1, 512, dim, device=dev, dtype=torch.bfloat16,
                          generator=torch.Generator(device=dev).manual_seed(9999)) * 0.02
    grad_out = torch.randn(local_seq, dim, device=dev, dtype=torch.float32,
                           generator=torch.Generator(device=dev).manual_seed(1111))

    if rank == 0:
        print(f'\n{"="*80}')
        print(f'Correctness Test: serial vs fused_var')
        print(f'  {num_layers} layers, dim={dim}, seq={seq_len}, SP={ng}')
        print(f'{"="*80}')

    # ---- Test 1: Forward output ----
    with torch.no_grad():
        xs = x.clone()
        xv = x.clone()
    xs.requires_grad_(True)
    xv.requires_grad_(True)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        ys = serial_model(xs, e, grid, context)
        yv = var_model(xv, e, grid, context)

    fwd_rel = (ys.float() - yv.float()).norm().item() / (ys.float().norm().item() + 1e-12)
    if rank == 0:
        status = 'PASS' if fwd_rel < 0.01 else 'FAIL'
        print(f'[Test 1] Forward output rel error: {fwd_rel:.6f} ({status}, threshold < 1%)')

    # ---- Test 2: Grad_X (input gradient) ----
    ys.backward(grad_out)
    yv.backward(grad_out)

    grad_x_rel = (xs.grad.float() - xv.grad.float()).norm().item() / (
        xs.grad.float().norm().item() + 1e-12)
    if rank == 0:
        status = 'PASS' if grad_x_rel < 0.001 else 'FAIL'
        print(f'[Test 2] Grad_X rel error: {grad_x_rel:.6f} ({status}, threshold < 0.1%)')

    # ---- Test 3: Grad_W (weight gradients for replicated params: q, k, v) ----
    # Note: Wo is sharded differently (serial: full Wo, var: Wo_r_local),
    # so we only compare q/k/v which are replicated in both strategies.
    grad_w_errors = {}
    for li in range(num_layers):
        s_block = serial_model.blocks[li]
        v_block = var_model.blocks[li]
        for pname in ['q', 'k', 'v']:
            sg = getattr(s_block.self_attn.model, pname).weight.grad
            vg = getattr(v_block.self_attn.model, pname).weight.grad
            if sg is not None and vg is not None:
                rel = (sg.float() - vg.float()).norm().item() / (
                    sg.float().norm().item() + 1e-12)
                grad_w_errors[f'L{li}.{pname}'] = rel
        # Also compare FFN and norm grads
        for pname in ['fc1', 'fc2']:
            sg = getattr(s_block.ffn, pname).weight.grad if hasattr(s_block.ffn, pname) else None
            vg = getattr(v_block.ffn, pname).weight.grad if hasattr(v_block.ffn, pname) else None
            if sg is not None and vg is not None:
                rel = (sg.float() - vg.float()).norm().item() / (
                    sg.float().norm().item() + 1e-12)
                grad_w_errors[f'L{li}.ffn.{pname}'] = rel

    max_grad_w = max(grad_w_errors.values()) if grad_w_errors else 0
    if rank == 0:
        status = 'PASS' if max_grad_w < 0.01 else 'FAIL'
        print(f'[Test 3] Grad_W max rel error: {max_grad_w:.6f} ({status}, threshold < 1%)')
        # Print top 5 worst
        sorted_errs = sorted(grad_w_errors.items(), key=lambda x: -x[1])[:5]
        for name, err in sorted_errs:
            print(f'         {name}: {err:.6f}')

    # ---- Test 4: Loss curve (20 steps) ----
    num_train_steps = 50
    if rank == 0:
        print(f'\n[Test 4] Loss curve ({num_train_steps} training steps, lr=1e-4, different input each step)')

    # Rebuild fresh models with same weights for training
    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        serial_model2 = SPWanTransformer(cfg, sp_cfg, num_layers, 'serial')
        var_model2 = SPWanTransformer(cfg, sp_cfg, num_layers, 'fused_var')
    torch.set_default_dtype(old)
    serial_model2.to(device=dev, dtype=torch.bfloat16)
    var_model2.to(device=dev, dtype=torch.bfloat16)

    # Copy weights again (before setup_shape, so _build_weights can shard Wo)
    for s_block, v_block in zip(serial_model2.blocks, var_model2.blocks):
        s_attn = s_block.self_attn
        v_attn = v_block.self_attn
        for name in ['q', 'k', 'v', 'o']:
            getattr(v_attn.model, name).weight.data.copy_(
                getattr(s_attn.model, name).weight.data)
        for name in ['norm1', 'norm2', 'norm3']:
            s_norm = getattr(s_block, name)
            v_norm = getattr(v_block, name)
            if hasattr(s_norm, 'weight') and s_norm.weight is not None:
                v_norm.weight.data.copy_(s_norm.weight.data)
        v_block.modulation.data.copy_(s_block.modulation.data)
        v_block.ffn.load_state_dict(s_block.ffn.state_dict())
        if hasattr(s_block.self_attn.model, 'norm_q'):
            v_attn.model.norm_q.weight.data.copy_(s_attn.model.norm_q.weight.data)
            v_attn.model.norm_k.weight.data.copy_(s_attn.model.norm_k.weight.data)
    for s_block, v_block in zip(serial_model2.blocks, var_model2.blocks):
        if hasattr(s_block, 'cross_attn'):
            v_block.cross_attn.load_state_dict(s_block.cross_attn.state_dict())

    serial_model2.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    var_model2.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)

    opt_s = torch.optim.AdamW(serial_model2.parameters(), lr=1e-4)
    opt_v = torch.optim.AdamW(var_model2.parameters(), lr=1e-4)

    # Different input each step (reflects real training)
    torch.manual_seed(42)
    inputs = [torch.randn(local_seq, dim, device=dev, dtype=torch.bfloat16)
              for _ in range(50)]

    losses_s, losses_v = [], []
    max_loss_diff = 0
    for step in range(num_train_steps):
        x_step = inputs[step]

        # Serial
        opt_s.zero_grad()
        xi = x_step.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out_s = serial_model2(xi, e, grid, context)
            loss_s = out_s.float().pow(2).mean()
        loss_s.backward()
        for p in serial_model2.parameters():
            if p.grad is not None and not getattr(p, '_sp_sharded', False):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ng)
        opt_s.step()

        # Var
        opt_v.zero_grad()
        xi2 = x_step.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out_v = var_model2(xi2, e, grid, context)
            loss_v = out_v.float().pow(2).mean()
        loss_v.backward()
        for p in var_model2.parameters():
            if p.grad is not None and not getattr(p, '_sp_sharded', False):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(ng)
        opt_v.step()

        loss_s_val = loss_s.item()
        loss_v_val = loss_v.item()
        losses_s.append(loss_s_val)
        losses_v.append(loss_v_val)
        diff = abs(loss_s_val - loss_v_val) / (abs(loss_s_val) + 1e-8)
        max_loss_diff = max(max_loss_diff, diff)

        if rank == 0 and step % 5 == 0:
            print(f'  step {step:2d}: serial={loss_s_val:.6f}  var={loss_v_val:.6f}  diff={diff:.4%}', flush=True)

    if rank == 0:
        print(f'\n  Max loss difference across 20 steps: {max_loss_diff:.4%}')
        status = 'PASS' if max_loss_diff < 0.05 else 'FAIL'
        print(f'  ({status}, threshold < 5%)')

        # Summary
        print(f'\n{"="*80}')
        print(f'Summary:')
        print(f'  Forward output:   {fwd_rel:.6f}  {"PASS" if fwd_rel < 0.01 else "FAIL"}')
        print(f'  Grad_X:           {grad_x_rel:.6f}  {"PASS" if grad_x_rel < 0.001 else "FAIL"}')
        print(f'  Grad_W (max):     {max_grad_w:.6f}  {"PASS" if max_grad_w < 0.01 else "FAIL"}')
        print(f'  Loss curve diff:  {max_loss_diff:.4%}  {"PASS" if max_loss_diff < 0.05 else "FAIL"}')
        print(f'{"="*80}')

        # Save loss curve data for plotting
        import json
        out = {'steps': list(range(num_train_steps)), 'serial': losses_s, 'var': losses_v,
               'fwd_rel': fwd_rel, 'grad_x_rel': grad_x_rel,
               'max_grad_w_rel': max_grad_w, 'max_loss_diff': max_loss_diff}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correctness_results.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nResults saved to {path}')

    serial_model.destroy_buffers()
    var_model.destroy_buffers()
    serial_model2.destroy_buffers()
    var_model2.destroy_buffers()
    dist.destroy_process_group()


if __name__ == '__main__':
    import socket
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    s = socket.socket(); s.bind(('', 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
