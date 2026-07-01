"""Benchmark + correctness verification utilities for Wan2.1 SP attention.

Handles:
  - Timing (event-based, with warmup + barrier sync)
  - FWD correctness (distributed → gather → compare vs single-GPU ref)
  - BWD gradient correctness (grad_X + weight grads, with FSDP2 sync)
  - Distributed gradient correctness (per-rank grad_X_local vs ref slice)
"""

import math
import socket
import torch
import torch.distributed as dist


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def time_call(fn, iters, warmup=3, resets=(), group=None):
    """Event-based timing with warmup + barrier sync."""
    for _ in range(warmup):
        for r in resets:
            r()
        torch.cuda.synchronize()
        if group:
            dist.barrier(group)
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    tot = 0.0
    for _ in range(iters):
        for r in resets:
            r()
        torch.cuda.synchronize()
        if group:
            dist.barrier(group)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        tot += s.elapsed_time(e)
    return tot / iters * 1000.0  # us


def rel_diff(a, b):
    """Relative difference: ||a-b|| / ||b||."""
    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-12)


def gather_to_rank0(tensor, group, world_size):
    """All-gather tensor across ranks, return concatenated."""
    full = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(full, tensor, group=group)
    return torch.cat(full, dim=0)


def sync_weight_grads_manual(module, group):
    """Manual all-reduce of all weight gradients (FSDP2 fallback)."""
    grads = []
    params = []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad)
            params.append(p)
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def kfmt(n):
    return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)
