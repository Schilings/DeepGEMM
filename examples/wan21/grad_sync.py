"""FSDP2-style gradient sync via reduce-scatter.

Since torch.distributed.fsdp2 is not available in this PyTorch build, we use
the underlying reduce-scatter primitive to implement the same semantics:
  - Replicated params → reduce-scatter grads to shards (FSDP2 behavior)
  - Already-sharded params (fused_var Wo row-split) → no sync needed

reduce_scatter_tensor = all-reduce + scatter (each rank gets 1/N of the summed grad).
This is exactly what FSDP2 does internally during backward().

Usage in bench:
  gX, gWqkv, gWo = strat.backward(gy, X_local, grid, llseq)
  strat._last_grad_Wqkv = gWqkv; strat._last_grad_Wo = gWo
  sync_grads(strat, group)  # FSDP2-style reduce-scatter
"""

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _all_reduce_bucket(grads, group, average):
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    if average:
        flat.div_(dist.get_world_size(group))
    for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        grad.copy_(synced)


def sync_replicated_grads(module, group, bucket_cap_mb=64.0, average=False):
    """Synchronize replicated SP parameters and leave native SP shards local.

    Standard Ulysses parameters receive partial-sequence gradients and therefore
    need a reduction across the SP group. Parameters tagged ``_sp_sharded`` own
    distinct logical shards on different ranks and must not be reduced together.
    Large gradients are reduced in place; smaller gradients are packed into
    bounded buckets to avoid one model-sized temporary allocation.
    """
    cap_bytes = max(1, int(bucket_cap_mb * 1024 * 1024))
    buckets = {}
    bucket_bytes = {}
    synced_elements = 0
    sharded_elements = 0

    def flush(key):
        grads = buckets.get(key, [])
        _all_reduce_bucket(grads, group, average)
        buckets[key] = []
        bucket_bytes[key] = 0

    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if getattr(parameter, "_sp_sharded", False):
            sharded_elements += parameter.numel()
            continue
        if grad.is_sparse:
            raise TypeError("Sparse gradients are not supported by SP synchronization")
        grad = grad.detach()
        synced_elements += grad.numel()
        num_bytes = grad.numel() * grad.element_size()
        key = (grad.device, grad.dtype)
        buckets.setdefault(key, [])
        bucket_bytes.setdefault(key, 0)
        if num_bytes >= cap_bytes:
            flush(key)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            if average:
                grad.div_(dist.get_world_size(group))
            continue
        if bucket_bytes[key] + num_bytes > cap_bytes:
            flush(key)
        buckets[key].append(grad)
        bucket_bytes[key] += num_bytes

    for key in list(buckets):
        flush(key)
    return synced_elements, sharded_elements


def sync_grads(strat, group):
    """FSDP2-style gradient sync: reduce-scatter weight grads.

    - Wqkv (always replicated): reduce-scatter → each rank gets 1/N of the grad.
    - Wo:
      - serial/fused_std (replicated): reduce-scatter → 1/N per rank.
      - fused_var (_wo_sharded=True): row-split → NO sync (grad is local).
    """
    ng = group.size()
    grad_Wqkv = strat._last_grad_Wqkv
    grad_Wo = strat._last_grad_Wo
    assert grad_Wqkv is not None, "Set strat._last_grad_Wqkv before calling sync"

    # Wqkv: reduce-scatter (FSDP2 semantics)
    assert grad_Wqkv.shape[0] % ng == 0, f"Wqkv grad rows {grad_Wqkv.shape[0]} not divisible by {ng}"
    shard_rows = grad_Wqkv.shape[0] // ng
    wqkv_shard = torch.empty((shard_rows, grad_Wqkv.shape[1]),
                             dtype=grad_Wqkv.dtype, device=grad_Wqkv.device)
    dist.reduce_scatter_tensor(wqkv_shard, grad_Wqkv, op=dist.ReduceOp.SUM, group=group)

    # Wo: reduce-scatter unless row-sharded (fused_var)
    if grad_Wo is not None and not getattr(strat, '_wo_sharded', False):
        assert grad_Wo.shape[0] % ng == 0, f"Wo grad rows {grad_Wo.shape[0]} not divisible by {ng}"
        wo_shard_rows = grad_Wo.shape[0] // ng
        wo_shard = torch.empty((wo_shard_rows, grad_Wo.shape[1]),
                               dtype=grad_Wo.dtype, device=grad_Wo.device)
        dist.reduce_scatter_tensor(wo_shard, grad_Wo, op=dist.ReduceOp.SUM, group=group)
    else:
        wo_shard = grad_Wo  # already local (fused_var row-split)

    return wqkv_shard, wo_shard


def sync_grads_all_reduce(strat, group):
    """All-reduce weight grads (non-FSDP2, keeps full grad on every rank).

    Used for correctness verification (need full grads to compare with reference).
    """
    grad_Wqkv = strat._last_grad_Wqkv
    grad_Wo = strat._last_grad_Wo

    dist.all_reduce(grad_Wqkv, op=dist.ReduceOp.SUM, group=group)
    if grad_Wo is not None and not getattr(strat, '_wo_sharded', False):
        dist.all_reduce(grad_Wo, op=dist.ReduceOp.SUM, group=group)

    return grad_Wqkv, grad_Wo
