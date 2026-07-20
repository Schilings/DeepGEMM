"""Dynamic Ulysses layer — runtime SP-size-aware attention layer.

Wraps the existing SerialUlysses / FusedUlysses strategies and dynamically
switches the SP group at runtime. Each forward/backward call uses the SP
group specified by the current microbatch.

Key challenge: UnifiedSymmBuffer is sized for a specific SP configuration.
We pre-allocate buffers for each SP size and select the right one at runtime.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

from .sp_group_manager import DynamicSPGroupManager, SPGroupInfo


@dataclass
class SPConfig:
    """Configuration for a specific SP size."""
    sp_size: int
    sp_group: 'torch.distributed.ProcessGroup'
    sp_rank: int
    local_nheads: int
    local_seq: int
    sym_buffer: Optional[object] = None  # UnifiedSymmBuffer (lazy alloc)


class DynamicUlyssesLayer(nn.Module):
    """A Transformer self-attention layer that supports dynamic SP.

    On each forward, the caller specifies which SP size to use. The layer
    selects the pre-allocated communication group and buffer, then runs
    the attention with that SP configuration.

    Args:
        model: The base attention model (WanSelfAttention).
        num_heads: Total attention heads.
        head_dim: Head dimension.
        hidden: Hidden dimension.
        group_manager: DynamicSPGroupManager with pre-created groups.
        world_size: Total GPU count.
    """

    def __init__(self,
                 model: nn.Module,
                 num_heads: int,
                 head_dim: int,
                 hidden: int,
                 group_manager: DynamicSPGroupManager,
                 world_size: int):
        super().__init__()
        self.model = model  # WanSelfAttention
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden = hidden
        self.gm = group_manager
        self.world_size = world_size

        # Pre-compute SP configs for each valid SP size
        self.sp_configs: Dict[int, SPConfig] = {}
        for sp_size in group_manager.get_valid_sp_sizes():
            info = group_manager.get_groups(sp_size)
            local_nheads = num_heads // sp_size
            self.sp_configs[sp_size] = SPConfig(
                sp_size=sp_size,
                sp_group=info.sp_group,
                sp_rank=info.sp_rank,
                local_nheads=local_nheads,
                local_seq=0,  # set per-forward
                sym_buffer=None,  # lazy alloc
            )

        self._current_sp: Optional[SPConfig] = None

    def set_sp_size(self, sp_size: int, local_seq: int):
        """Set the SP configuration for the current forward pass."""
        cfg = self.sp_configs[sp_size]
        cfg.local_seq = local_seq
        self._current_sp = cfg

    def forward(self, x_local: torch.Tensor, grid: torch.Tensor, llseq: int) -> torch.Tensor:
        """Run attention with the current SP configuration.

        Args:
            x_local: [bs*local_seq, hidden] — local sequence shard.
            grid:    Spatial grid for RoPE.
            llseq:   Local sequence length (per rank).

        Returns:
            y: [bs*local_seq, hidden] — output.
        """
        cfg = self._current_sp
        assert cfg is not None, "Call set_sp_size() before forward()"

        if cfg.sp_size == 1:
            # Pure DP: no SP communication, just standard attention
            return self._forward_dp(x_local, grid, llseq)
        else:
            # Ulysses SP: A2A-based attention
            return self._forward_ulysses(x_local, grid, llseq, cfg)

    def _forward_dp(self, x_local: torch.Tensor, grid: torch.Tensor, llseq: int) -> torch.Tensor:
        """SP=1: no sequence parallelism, pure data parallel."""
        # Standard attention without A2A
        bs = x_local.shape[0] // llseq
        q = self.model.q(x_local)
        k = self.model.k(x_local)
        v = self.model.v(x_local)

        # QK RMSNorm
        q = self.model.norm_q(q)
        k = self.model.norm_k(k)

        # Reshape for attention
        q = q.view(bs, llseq, self.num_heads, self.head_dim)
        k = k.view(bs, llseq, self.num_heads, self.head_dim)
        v = v.view(bs, llseq, self.num_heads, self.head_dim)

        # RoPE
        from examples.wan21.sp.base import rope_apply
        freqs = self.model.freqs[:llseq] if hasattr(self.model, 'freqs') else None
        if freqs is not None:
            q = rope_apply(q, grid, freqs).to(torch.bfloat16)
            k = rope_apply(k, grid, freqs).to(torch.bfloat16)

        # FlashAttention
        scale = self.head_dim ** -0.5
        from examples.wan21.sp.base import _fa4_attn
        o = _fa4_attn(q, k, v, scale, causal=True)

        # Wo projection
        o = o.reshape(bs * llseq, -1)
        y = self.model.o(o)
        return y

    def _forward_ulysses(self, x_local: torch.Tensor, grid: torch.Tensor,
                         llseq: int, cfg: SPConfig) -> torch.Tensor:
        """SP>1: Ulysses attention with A2A."""
        bs = x_local.shape[0] // llseq
        sp = cfg.sp_size
        local_nh = cfg.local_nheads

        # QKV projection
        q = self.model.q(x_local)
        k = self.model.k(x_local)
        v = self.model.v(x_local)

        # QK RMSNorm (before A2A, on full hidden dim)
        q = self.model.norm_q(q)
        k = self.model.norm_k(k)

        # Reshape: [bs, local_seq, sp, local_nh, hd]
        q = q.view(bs, llseq, sp, local_nh, self.head_dim)
        k = k.view(bs, llseq, sp, local_nh, self.head_dim)
        v = v.view(bs, llseq, sp, local_nh, self.head_dim)

        # A2A: scatter heads, gather sequence
        def scatter_heads(tensor):
            send = tensor.permute(2, 0, 1, 3, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=cfg.sp_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(bs, sp * llseq, local_nh, self.head_dim)

        import torch.distributed as dist
        q = scatter_heads(q)
        k = scatter_heads(k)
        v = scatter_heads(v)

        # RoPE
        seq = sp * llseq
        from examples.wan21.sp.base import rope_apply
        freqs = self.model.freqs[:seq] if hasattr(self.model, 'freqs') else None
        if freqs is not None:
            q = rope_apply(q, grid, freqs).to(torch.bfloat16)
            k = rope_apply(k, grid, freqs).to(torch.bfloat16)

        # FlashAttention
        scale = self.head_dim ** -0.5
        from examples.wan21.sp.base import _fa4_attn
        o = _fa4_attn(q, k, v, scale, causal=True)

        # A2A inverse: scatter sequence, gather heads
        def gather_heads(tensor):
            send = tensor.view(bs, sp, llseq, local_nh, self.head_dim).permute(1, 0, 2, 3, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=cfg.sp_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(bs * llseq, -1)

        o = gather_heads(o)

        # Wo projection
        y = self.model.o(o)
        return y
