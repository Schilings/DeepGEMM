"""Overlap gradient sync — trigger AllReduce during backward computation.

Instead of waiting for all microbatches to finish backward, this module
overlaps gradient AllReduce with the backward computation of subsequent
microbatches. This is similar to DDP's gradient bucketing overlap.

Strategy:
1. After each microbatch's backward, check if any gradient bucket is full
2. If full, launch AllReduce on that bucket (non-blocking)
3. Continue with next microbatch's forward/backward
4. Before final gradient sync, wait for all pending AllReduces

This reduces the "sync" phase at the end of the step.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading


class OverlapGradientSync:
    """Overlap gradient AllReduce with backward computation.

    Args:
        world_group: The world process group (all ranks).
        bucket_cap_mb: Bucket capacity in MB (like DDP).
    """

    def __init__(self,
                 world_group: 'dist.ProcessGroup' = None,
                 bucket_cap_mb: float = 25.0):
        self.world_group = world_group if world_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.world_group)
        self.rank = dist.get_rank(self.world_group)
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)

        # Registered parameters grouped by bucket
        self._buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        self._bucket_sizes: Dict[str, int] = defaultdict(int)

        # Pending AllReduce handles
        self._pending_works: List[torch.distributed.Work] = []

        # Token scaling
        self._local_tokens = 0
        self._global_tokens = 0

    def set_token_counts(self, local_tokens: int):
        """Set local token count and compute global total."""
        self._local_tokens = local_tokens
        token_tensor = torch.tensor([local_tokens], dtype=torch.float32, device='cuda')
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, group=self.world_group)
        self._global_tokens = token_tensor.item()

    def register_param(self, name: str, param: torch.Tensor):
        """Register a parameter for gradient sync."""
        bucket_name = self._get_bucket_name(param)
        self._buckets[bucket_name].append((name, param))
        self._bucket_sizes[bucket_name] += param.numel() * param.element_size()

    def _get_bucket_name(self, param: torch.Tensor) -> str:
        param_bytes = param.numel() * param.element_size()
        bucket_idx = int(param_bytes // self.bucket_cap_bytes)
        return f"bucket_{bucket_idx}"

    def maybe_sync_bucket(self, bucket_name: str, scale: float):
        """Launch non-blocking AllReduce for a bucket if it's ready.

        Called after a microbatch's backward. Only syncs buckets where
        all parameters have gradients.
        """
        params = self._buckets.get(bucket_name, [])
        if not params:
            return

        # Check if all params in this bucket have gradients
        grads = []
        for name, param in params:
            if param.grad is None:
                return  # not ready yet
            grads.append(param.grad.data.flatten())

        if not grads:
            return

        # Flatten and launch non-blocking AllReduce
        flat_grad = torch.cat(grads)
        work = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM,
                               group=self.world_group, async_op=True)
        self._pending_works.append((work, flat_grad, params, scale))

    def sync_all(self, scale_by_tokens: bool = True):
        """Launch AllReduce for all remaining buckets and wait for completion."""
        if scale_by_tokens and self._global_tokens > 0:
            scale = 1.0 / self._global_tokens
        else:
            scale = 1.0 / self.world_size

        # Launch AllReduce for all buckets that haven't been synced yet
        for bucket_name in self._buckets:
            self.maybe_sync_bucket(bucket_name, scale)

        # Wait for all pending AllReduces
        for work, flat_grad, params, s in self._pending_works:
            work.wait()
            flat_grad.mul_(s)
            # Unflatten back
            offset = 0
            for name, param in params:
                if param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.data.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                    offset += numel

        self._pending_works.clear()

    def reset(self):
        """Clear all state for the next step."""
        self._buckets.clear()
        self._bucket_sizes.clear()
        self._pending_works.clear()
        self._local_tokens = 0
        self._global_tokens = 0
