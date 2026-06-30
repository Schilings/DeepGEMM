"""FSDP2 utilities for weight gradient synchronization in SP training.

In Sequence Parallelism, each rank computes PARTIAL weight gradients (from its local seq shard).
These partials must be summed (all-reduced) to get the correct full gradient — exactly what
FSDP2 does automatically via DTensor + reduce-scatter/gather.

Two modes:
  1. FSDP2 (if available): wrap module with `fully_shard()` — weight grads auto-synced in backward
  2. Manual fallback: all-reduce weight grads after backward (equivalent semantics, simpler)

Usage in SP training:
  attn = UlyssesStandardAttention(config, sp_config)   # SP strategy wraps model
  attn = wrap_fsdp2(attn, train_config)                 # FSDP2 wraps SP strategy
  y = attn(x_local)                                      # forward
  loss = y.sum()
  loss.backward()                                         # backward → weight grads auto-synced (FSDP2)
  # or: sync_weight_grads(attn, group)                   # manual fallback if no FSDP2
"""

import torch
import torch.nn as nn
import torch.distributed as dist

# Try to import FSDP2 (PyTorch >= 2.4 experimental, stable in 2.9)
_FSDP2_AVAILABLE = False
_fully_shard = None
try:
    from torch.distributed.fsdp2 import fully_shard as _fully_shard
    _FSDP2_AVAILABLE = True
except ImportError:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP1
        _fully_shard = None
    except ImportError:
        _FSDP1 = None


def is_fsdp2_available() -> bool:
    return _FSDP2_AVAILABLE


def wrap_fsdp2(module: nn.Module, train_config, mesh=None) -> nn.Module:
    """Wrap a module with FSDP2 for weight sharding + gradient sync.

    If FSDP2 is available, uses `fully_shard()` which:
      - Shards parameters across ranks (DTensor)
      - Automatically reduces gradients during backward
      - Handles prefetch/reshard for performance

    If FSDP2 is not available, returns module unchanged (use sync_weight_grads manually).

    Args:
      module: The module to wrap (e.g., UlyssesStandardAttention)
      train_config: TrainConfig with fsdp settings
      mesh: Optional DeviceMesh for FSDP2 (if None, uses default world mesh)
    """
    if not _FSDP2_AVAILABLE:
        if train_config.use_fsdp2:
            import warnings
            warnings.warn(
                "FSDP2 requested but not available (need PyTorch >= 2.4 with fsdp2). "
                "Falling back to manual gradient sync. Use sync_weight_grads() after backward.")
        return module

    # FSDP2 fully_shard API
    kwargs = {}
    if mesh is not None:
        kwargs['mesh'] = mesh

    _fully_shard(module, **kwargs)
    return module


def sync_weight_grads(module: nn.Module, group: dist.ProcessGroup = None):
    """Manual gradient synchronization (FSDP2 fallback).

    All-reduces all parameter gradients across the group. This is equivalent to what
    FSDP2 does automatically, but simpler and more explicit.

    In SP training:
      - Input grad (grad_X): each rank has its local seq shard's grad → already correct
      - Weight grads (grad_Wqkv, grad_Wo): each rank has PARTIAL grad from local seq → need all-reduce

    Call this after backward() but before optimizer.step().
    """
    grads = []
    params = []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad)
            params.append(p)
    if not grads:
        return
    # Coalesce all-reduce for efficiency
    flat_grads = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, group=group)
    offset = 0
    for p, g in zip(params, grads):
        n = g.numel()
        g.copy_(flat_grads[offset:offset + n].view_as(g))
        offset += n


def get_weight_grad_norms(module: nn.Module, group: dist.ProcessGroup = None):
    """Get per-parameter gradient norms after sync. Useful for debugging / logging."""
    results = {}
    for name, p in module.named_parameters():
        if p.grad is not None:
            results[name] = {
                'norm': p.grad.float().norm().item(),
                'shape': tuple(p.shape),
                'dtype': str(p.dtype),
            }
    return results
