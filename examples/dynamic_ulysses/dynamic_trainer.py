"""Dynamic SP trainer — complete training loop with dynamic SP scheduling.

For each training step:
1. BalancedDataLoader produces a list of microbatches with assigned SP sizes
2. Each microbatch is processed by all ranks (organized into SP groups)
3. Forward + backward for each microbatch, accumulating gradients
4. After all microbatches: DynamicGradientSync across all ranks

Architecture (8 GPU example):

  Step with sequences [32K, 8K, 8K, 4K, 4K, 2K, 2K, 1K]:

  MB 0: SP=4, seq=32K  → ranks [0-3] and [4-7] each process 32K
  MB 1: SP=2, seq=8K   → 4 SP groups, each processes 8K
  MB 2: SP=2, seq=8K   → 4 SP groups, each processes 8K
  MB 3: SP=2, seq=4K   → 4 SP groups
  ...
  MB 7: SP=1, seq=1K   → 8 independent ranks

  All ranks accumulate gradients, then AllReduce.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .sp_group_manager import DynamicSPGroupManager, SPGroupInfo
from .balanced_loader import BalancedDataLoader, Microbatch
from .grad_sync import DynamicGradientSync


class DynamicTrainer:
    """Complete training loop with dynamic SP scheduling.

    Wraps a model (nn.ModuleList of transformer blocks) and handles:
    - Microbatch scheduling based on sequence lengths
    - Per-microbatch SP group selection
    - Forward + backward + gradient accumulation
    - Cross-rank gradient synchronization

    Args:
        model:        The transformer model (must support set_sp_config).
        group_manager: DynamicSPGroupManager with pre-created groups.
        world_size:   Total GPU count.
        hidden:       Hidden dimension.
        num_heads:    Total attention heads.
        head_dim:     Head dimension.
        num_layers:   Number of transformer layers.
        causal:       Whether attention is causal.
    """

    def __init__(self,
                 model: nn.Module,
                 group_manager: DynamicSPGroupManager,
                 world_size: int,
                 hidden: int,
                 num_heads: int,
                 head_dim: int,
                 num_layers: int,
                 causal: bool = True):
        self.model = model
        self.gm = group_manager
        self.world_size = world_size
        self.rank = group_manager.rank
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.causal = causal

        self.loader = BalancedDataLoader(world_size)
        self.grad_sync = DynamicGradientSync()

        # Register all model parameters for gradient sync
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_sync.add_param(name, param)

        # Track total tokens for loss scaling
        self._total_tokens = 0

    def train_step(self,
                   sequence_lengths: List[int],
                   device: torch.device,
                   seed: int = 42) -> Dict[str, float]:
        """Execute one training step with dynamic SP.

        Args:
            sequence_lengths: List of sequence lengths to process.
            device:           CUDA device.
            seed:             Random seed for input generation.

        Returns:
            Dict with timing and stats.
        """
        import time

        # Schedule microbatches
        microbatches = self.loader.schedule(sequence_lengths)
        total_tokens = self.loader.total_tokens(microbatches)

        # Zero gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Process each microbatch
        step_start = time.time()
        fwd_times = []
        bwd_times = []

        for mb_idx, mb in enumerate(microbatches):
            info = self.gm.get_groups(mb.sp_size)

            # Only ranks in this SP group's DP copies participate
            # Since all ranks participate (each rank is in exactly one SP group
            # of each size), all ranks process this microbatch.

            # Set SP config on the model
            self._set_sp_config(mb, info)

            # Generate input data for this microbatch
            x_local = self._generate_input(mb, device, seed + mb_idx)

            # Forward
            fwd_start = time.time()
            output = self._forward(mb, x_local)
            fwd_time = time.time() - fwd_start
            fwd_times.append(fwd_time)

            # Backward
            bwd_start = time.time()
            self._backward(mb, output)
            bwd_time = time.time() - bwd_start
            bwd_times.append(bwd_time)

            # Barrier between microbatches (ensure all ranks are in sync)
            dist.barrier()

        # Gradient sync across all ranks
        sync_start = time.time()
        self.grad_sync.set_token_counts(total_tokens // self.world_size)
        self.grad_sync.sync(scale_by_tokens=True)
        sync_time = time.time() - sync_start

        total_time = time.time() - step_start

        return {
            'num_microbatches': len(microbatches),
            'total_tokens': total_tokens,
            'total_time': total_time,
            'fwd_time': sum(fwd_times),
            'bwd_time': sum(bwd_times),
            'sync_time': sync_time,
            'sp_sizes': [mb.sp_size for mb in microbatches],
        }

    def _set_sp_config(self, mb: Microbatch, info: SPGroupInfo):
        """Configure the model for the current SP size."""
        sp = mb.sp_size
        local_nh = self.num_heads // sp

        # Update each layer's SP configuration
        for block in self.model.blocks if hasattr(self.model, 'blocks') else [self.model]:
            attn = block.self_attn if hasattr(block, 'self_attn') else block
            attn.sp_size = sp
            attn.group = info.sp_group
            attn.local_nh = local_nh
            attn.local_n = local_nh * self.head_dim
            attn.local_hidden = attn.local_n
            attn.local_nqkv = 3 * attn.local_n
            attn.local_seq = mb.local_seq
            attn.local_m = 1 * mb.local_seq  # bs=1
            attn.bs = 1
            attn.seq = mb.seq_len
            attn._shape_set = True

    def _generate_input(self, mb: Microbatch, device: torch.device, seed: int) -> torch.Tensor:
        """Generate random input for the microbatch."""
        g = torch.Generator(device=device).manual_seed(seed)
        # Each rank generates its local sequence shard
        # Use different seeds for different DP copies to simulate different data
        dp_copy_offset = self.rank // mb.sp_size
        g = torch.Generator(device=device).manual_seed(seed + dp_copy_offset * 10000)
        return torch.randn(mb.local_seq, self.hidden, dtype=torch.bfloat16,
                           device=device, generator=g)

    def _forward(self, mb: Microbatch, x_local: torch.Tensor) -> torch.Tensor:
        """Run forward pass for one microbatch."""
        # Create a simple grid for RoPE
        seq = mb.seq_len
        assert seq % (16 * 128) == 0 or seq % (8 * 128) == 0, \
            f"seq={seq} must be divisible by 16*128 or 8*128"
        if seq % (16 * 128) == 0:
            grid = torch.tensor([[seq // (16 * 128), 16, 128]], dtype=torch.long, device=x_local.device)
        else:
            grid = torch.tensor([[seq // (8 * 128), 8, 128]], dtype=torch.long, device=x_local.device)

        if hasattr(self.model, 'blocks'):
            # Full transformer: need e and context
            e = torch.zeros(1, 6, self.hidden, dtype=torch.float32, device=x_local.device)
            context = torch.randn(1, 512, self.hidden, dtype=torch.bfloat16, device=x_local.device)
            return self.model(x_local, e, grid, context)
        else:
            # Single attention layer
            return self.model(x_local, grid, mb.local_seq)

    def _backward(self, mb: Microbatch, output: torch.Tensor):
        """Run backward pass for one microbatch."""
        # Create dummy gradient
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

    def destroy(self):
        """Cleanup."""
        self.gm.destroy()
        self.grad_sync.reset()
