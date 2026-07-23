"""Correctness test: serial vs fused_var_v2 — fwd, bwd, grad_W, loss curve.

Verifies that the v2 variant (native AG+GEMM backward with deferred QKV
weight-grad overlap) produces results consistent with the serial baseline:

1. Forward output: rel error < 1%
2. Grad_X (input grad): rel error < 0.1%
3. Grad_W (weight grads): q/k/v/FFN rel error < 1%
4. Loss curve: 50-step training, loss difference < 5%

Usage:
    DG_JIT_USE_NVRTC=1 \\
    PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \\
    python3 examples/ulysses_variant_v2/test_correctness.py 8
"""
from __future__ import annotations

import os, sys, json
import torch, torch.distributed as dist, torch.multiprocessing as mp

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.autograd_ops_v2 import finalize_deferred_grads, sync_deferred_grads
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
    num_layers = 4

    torch.manual_seed(42)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(dev):
            serial_model = SPWanTransformer(cfg, sp_cfg, num_layers, 'serial')
            v2_model = SPWanTransformer(cfg, sp_cfg, num_layers, 'fused_var_v2')
    finally:
        torch.set_default_dtype(old)
    serial_model.to(device=dev, dtype=torch.bfloat16)
    v2_model.to(device=dev, dtype=torch.bfloat16)

    # Copy weights, then setup_shape (which shards Wo + replaces QKV)
    for s_block, v_block in zip(serial_model.blocks, v2_model.blocks):
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
    for s_block, v_block in zip(serial_model.blocks, v2_model.blocks):
        if hasattr(s_block, 'cross_attn'):
            v_block.cross_attn.load_state_dict(s_block.cross_attn.state_dict())

    serial_model.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    v2_model.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    serial_model.train()
    v2_model.train()

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
        print(f'Correctness Test: serial vs fused_var_v2')
        print(f'  {num_layers} layers, dim={dim}, seq={seq_len}, SP={ng}')
        print(f'  v2: native AG+GEMM backward, deferred QKV weight-grad overlap')
        print(f'{"="*80}')

    # ---- Test 1: Forward output ----
    xs = x.clone().requires_grad_(True)
    xv = x.clone().requires_grad_(True)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        ys = serial_model(xs, e, grid, context)
        yv = v2_model(xv, e, grid, context)

    fwd_rel = (ys.float() - yv.float()).norm().item() / (ys.float().norm().item() + 1e-12)
    if rank == 0:
        status = 'PASS' if fwd_rel < 0.01 else 'FAIL'
        print(f'[Test 1] Forward output rel error: {fwd_rel:.6f} ({status}, threshold < 1%)')

    # ---- Test 2: Grad_X ----
    ys.backward(grad_out)
    finalize_deferred_grads(v2_model)
    yv.backward(grad_out)
    finalize_deferred_grads(v2_model)

    grad_x_rel = (xs.grad.float() - xv.grad.float()).norm().item() / (
        xs.grad.float().norm().item() + 1e-12)
    if rank == 0:
        status = 'PASS' if grad_x_rel < 0.001 else 'FAIL'
        print(f'[Test 2] Grad_X rel error: {grad_x_rel:.6f} ({status}, threshold < 0.1%)')

    # ---- Test 3: Grad_W (replicated params: q, k, v, FFN) ----
    grad_w_errors = {}
    for li in range(num_layers):
        s_block = serial_model.blocks[li]
        v_block = v2_model.blocks[li]
        for pname in ['q', 'k', 'v']:
            sg = getattr(s_block.self_attn.model, pname).weight.grad
            vg = getattr(v_block.self_attn.model, pname).weight.grad
            if sg is not None and vg is not None:
                rel = (sg.float() - vg.float()).norm().item() / (
                    sg.float().norm().item() + 1e-12)
                grad_w_errors[f'L{li}.{pname}'] = rel
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
        sorted_errs = sorted(grad_w_errors.items(), key=lambda x: -x[1])[:5]
        for name, err in sorted_errs:
            print(f'         {name}: {err:.6f}')

    # ---- Test 4: Loss curve (50 steps) ----
    num_train_steps = 50
    if rank == 0:
        print(f'\n[Test 4] Loss curve ({num_train_steps} steps, lr=1e-4)')

    torch.manual_seed(42)
    torch.set_default_dtype(torch.bfloat16)
    with torch.device(dev):
        serial_model2 = SPWanTransformer(cfg, sp_cfg, num_layers, 'serial')
        v2_model2 = SPWanTransformer(cfg, sp_cfg, num_layers, 'fused_var_v2')
    torch.set_default_dtype(old)
    serial_model2.to(device=dev, dtype=torch.bfloat16)
    v2_model2.to(device=dev, dtype=torch.bfloat16)

    for s_block, v_block in zip(serial_model2.blocks, v2_model2.blocks):
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
    for s_block, v_block in zip(serial_model2.blocks, v2_model2.blocks):
        if hasattr(s_block, 'cross_attn'):
            v_block.cross_attn.load_state_dict(s_block.cross_attn.state_dict())

    serial_model2.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)
    v2_model2.setup_shape(1, seq_len, cfg.num_heads, cfg.head_dim)

    opt_s = torch.optim.AdamW(serial_model2.parameters(), lr=1e-4)
    opt_v = torch.optim.AdamW(v2_model2.parameters(), lr=1e-4)

    torch.manual_seed(42)
    inputs = [torch.randn(local_seq, dim, device=dev, dtype=torch.bfloat16)
              for _ in range(num_train_steps)]

    losses_s, losses_v = [], []
    max_loss_diff = 0
    for step in range(num_train_steps):
        x_step = inputs[step]

        opt_s.zero_grad(set_to_none=True)
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

        opt_v.zero_grad(set_to_none=True)
        xi2 = x_step.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out_v = v2_model2(xi2, e, grid, context)
            loss_v = out_v.float().pow(2).mean()
        loss_v.backward()
        finalize_deferred_grads(v2_model2)
        for p in v2_model2.parameters():
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
            print(f'  step {step:2d}: serial={loss_s_val:.6f}  v2={loss_v_val:.6f}  diff={diff:.4%}', flush=True)

    if rank == 0:
        print(f'\n  Max loss difference across {num_train_steps} steps: {max_loss_diff:.4%}')
        status = 'PASS' if max_loss_diff < 0.05 else 'FAIL'
        print(f'  ({status}, threshold < 5%)')

        print(f'\n{"="*80}')
        print(f'Summary:')
        print(f'  Forward output:   {fwd_rel:.6f}  {"PASS" if fwd_rel < 0.01 else "FAIL"}')
        print(f'  Grad_X:           {grad_x_rel:.6f}  {"PASS" if grad_x_rel < 0.001 else "FAIL"}')
        print(f'  Grad_W (max):     {max_grad_w:.6f}  {"PASS" if max_grad_w < 0.01 else "FAIL"}')
        print(f'  Loss curve diff:  {max_loss_diff:.4%}  {"PASS" if max_loss_diff < 0.05 else "FAIL"}')
        print(f'{"="*80}')

        out = {'steps': list(range(num_train_steps)), 'serial': losses_s, 'v2': losses_v,
               'fwd_rel': fwd_rel, 'grad_x_rel': grad_x_rel,
               'max_grad_w_rel': max_grad_w, 'max_loss_diff': max_loss_diff}
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'correctness_results.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nResults saved to {path}')

    serial_model.destroy_buffers()
    v2_model.destroy_buffers()
    serial_model2.destroy_buffers()
    v2_model2.destroy_buffers()
    dist.destroy_process_group()


if __name__ == '__main__':
    import socket
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    s = socket.socket(); s.bind(('', 0)); port = s.getsockname()[1]; s.close()
    mp.spawn(run, args=(ng, port), nprocs=ng, join=True)
