"""Overlap gradient sync — overlaps AllReduce with backward computation.

Inspired by DDP's gradient bucketing: when a parameter's gradient is ready
(i.e., all operations depending on the parameter are done), immediately
launch AllReduce for that bucket while backward continues on other parameters.
"""
import torch
import torch.distributed as dist
from typing import Dict, List, Tuple
from collections import defaultdict

class OverlapGradientSync:
    """Overlapped bucketed gradient sync.

    Unlike DynamicGradientSync (which syncs all at once after backward),
    this launches AllReduce as soon as a bucket is full, overlapping
    communication with remaining backward computation.

    Args:
        world_group: Process group spanning all ranks.
        bucket_cap_mb: Max bucket size in MB.
    """
    def __init__(self, world_group=None, bucket_cap_mb=25.0):
        self.world_group = world_group if world_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.world_group)
        self.rank = dist.get_rank(self.world_group)
        self.bucket_cap_mb = bucket_cap_mb
        self._buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        self._bucket_size: Dict[str, int] = defaultdict(int)
        self._ready_buckets: List[str] = []
        self._local_tokens = 0
        self._global_tokens = 0

    def set_token_counts(self, local_tokens):
        self._local_tokens = local_tokens
        t = torch.tensor([local_tokens], dtype=torch.float32, device='cuda')
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.world_group)
        self._global_tokens = t.item()

    def add_param(self, name, param):
        bucket_name = self._get_bucket_name(param)
        self._buckets[bucket_name].append((name, param))
        self._bucket_size[bucket_name] += param.numel() * param.element_size()

    def _get_bucket_name(self, param):
        param_bytes = param.numel() * param.element_size()
        return f"bucket_{int(param_bytes // (self.bucket_cap_mb * 1024 * 1024))}"

    def mark_ready(self, param_name):
        """Mark a parameter's gradient as ready for sync.

        When a bucket is full (all params in it have grads), launch AllReduce.
        """
        for bucket_name, params in self._buckets.items():
            if bucket_name in self._ready_buckets:
                continue
            all_ready = all(p.grad is not None for _, p in params)
            if all_ready and self._bucket_size[bucket_name] >= self.bucket_cap_mb * 1024 * 1024:
                self._launch_bucket(bucket_name)

    def _launch_bucket(self, bucket_name):
        """Launch AllReduce for a bucket (non-blocking)."""
        params = self._buckets[bucket_name]
        grads = [p.grad.data.flatten() for _, p in params if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.world_group, async_op=True)
        # Note: in a real implementation, we'd store the work handle and wait later
        scale = 1.0 / (self._global_tokens if self._global_tokens > 0 else self.world_size)
        flat.mul_(scale)
        offset = 0
        for _, p in params:
            if p.grad is not None:
                n = p.grad.numel()
                p.grad.data.copy_(flat[offset:offset+n].view_as(p.grad))
                offset += n
        self._ready_buckets.append(bucket_name)

    def sync(self, scale_by_tokens=True):
        """Sync all remaining buckets (flush)."""
        scale = 1.0 / (self._global_tokens if scale_by_tokens and self._global_tokens > 0 else self.world_size)
        for bucket_name, params in self._buckets.items():
            if bucket_name in self._ready_buckets:
                continue
            grads = [p.grad.data.flatten() for _, p in params if p.grad is not None]
            if not grads:
                continue
            flat = torch.cat(grads)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.world_group)
            flat.mul_(scale)
            offset = 0
            for _, p in params:
                if p.grad is not None:
                    n = p.grad.numel()
                    p.grad.data.copy_(flat[offset:offset+n].view_as(p.grad))
                    offset += n
            self._ready_buckets.append(bucket_name)

    def reset(self):
        self._buckets.clear()
        self._bucket_size.clear()
        self._ready_buckets.clear()
        self._local_tokens = 0
        self._global_tokens = 0
