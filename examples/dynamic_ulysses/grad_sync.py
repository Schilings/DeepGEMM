"""Dynamic gradient sync — bucketed AllReduce across all ranks.

All ranks participate in gradient sync regardless of their SP group size.
The key insight (from ByteScale): parameters are replicated across all HDP
ranks, local gradients are partial sums, and AllReduce aggregates them.

Scaling: gradients are scaled by token count (not sample count) to ensure
correctness when different SP groups process different token counts.

For Wo (output projection):
  - SP=1 rank: Wo gradient is complete (no SP replication)
  - SP>1 rank: Wo gradient is replicated within SP group
    → AllReduce across all ranks, then divide by DP size
    → SP group's internal replication means each rank's gradient is 1× the
      correct gradient (not 1/sp), so no extra scaling needed
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class DynamicGradientSync:
    """Bucketed gradient sync across ranks with different SP group sizes.

    Args:
        world_group: The world process group (all ranks).
        token_counts: Per-rank token count for loss scaling.
    """

    def __init__(self,
                 world_group: 'dist.ProcessGroup' = None,
                 bucket_cap_mb: float = 25.0):
        self.world_group = world_group if world_group is not None else dist.group.WORLD
        self.world_size = dist.get_world_size(self.world_group)
        self.rank = dist.get_rank(self.world_group)
        self.bucket_cap_mb = bucket_cap_mb

        # Gradient buckets: {param_name: (param, grad_accumulator)}
        self._buckets: Dict[str, List[Tuple[str, torch.Tensor]]] = defaultdict(list)
        self._bucket_size: Dict[str, int] = defaultdict(int)

        # Token scaling
        self._local_tokens = 0
        self._global_tokens = 0

    def set_token_counts(self, local_tokens: int):
        """Set local token count and compute global total via AllReduce."""
        self._local_tokens = local_tokens
        token_tensor = torch.tensor([local_tokens], dtype=torch.float32,
                                     device='cuda')
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM, group=self.world_group)
        self._global_tokens = token_tensor.item()

    def add_param(self, name: str, param: torch.Tensor):
        """Register a parameter for gradient sync.

        Args:
            name: Parameter name (for bucketing).
            param: The parameter tensor. Must have .grad set before sync.
        """
        bucket_name = self._get_bucket_name(param)
        self._buckets[bucket_name].append((name, param))
        self._bucket_size[bucket_name] += param.numel() * param.element_size()

    def _get_bucket_name(self, param: torch.Tensor) -> str:
        """Assign parameter to a bucket based on size (like DDP)."""
        param_bytes = param.numel() * param.element_size()
        bucket_idx = int(param_bytes // (self.bucket_cap_mb * 1024 * 1024))
        return f"bucket_{bucket_idx}"

    def sync(self, scale_by_tokens: bool = True):
        """Synchronize gradients across all ranks.

        For each bucket:
        1. Flatten all parameter gradients into a single buffer
        2. AllReduce across the world group
        3. Divide by world_size (average) or by global_tokens (token-scaled)

        Args:
            scale_by_tokens: If True, scale by 1/global_tokens. If False,
                             scale by 1/world_size (standard DDP).
        """
        if scale_by_tokens and self._global_tokens > 0:
            scale = 1.0 / self._global_tokens
        else:
            scale = 1.0 / self.world_size

        for bucket_name, params in self._buckets.items():
            if not params:
                continue

            # Flatten gradients
            grads = []
            for name, param in params:
                if param.grad is not None:
                    grads.append(param.grad.data.flatten())

            if not grads:
                continue

            flat_grad = torch.cat(grads)

            # AllReduce
            dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, group=self.world_group)

            # Scale
            flat_grad.mul_(scale)

            # Unflatten back to parameters
            offset = 0
            for name, param in params:
                if param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.data.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                    offset += numel

    def reset(self):
        """Clear all buckets for the next step."""
        self._buckets.clear()
        self._bucket_size.clear()
        self._local_tokens = 0
        self._global_tokens = 0
