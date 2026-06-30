"""Wan2.1 Self-Attention — pure nn.Module, framework-agnostic.

This is the MODEL layer: defines WHAT is computed (QKV projection, QK-norm, 3D RoPE,
FlashAttention-4, Wo projection) without HOW it's parallelized.

SP strategies (sp/standard.py, sp/variant.py) wrap this module to add sharding/communication.
FSDP2 (fsdp2_utils.py) wraps the SP strategy to add weight sharding + gradient sync.

Design: forward() takes full (non-sharded) inputs → full output. The SP strategy
intercepts between sub-ops to insert communication. For autograd-based backward, this
module works standalone on a single GPU (used for reference correctness).
"""

import math
import torch
import torch.nn as nn

from .norm import WanRMSNorm
from .rope import build_wan21_freqs, rope_apply
from .config import Wan21Config


def _fa4_attn(q, k, v, scale, causal=False):
    """FlashAttention-4 wrapper. q/k/v: [B, S, H, D] → [B, S, H, D]."""
    from flash_attn.cute import flash_attn_func
    o = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    return o[0] if isinstance(o, tuple) else o


class WanSelfAttention(nn.Module):
    """Wan2.1 self-attention (single-GPU, full sequence, full heads).

    Sub-operations exposed as separate methods so SP strategies can interleave communication:
      qkv_proj(x)      → q, k, v  [B, S, H, D]
      apply_rope(q, k) → q_rot, k_rot
      attention(q,k,v) → o        [B, S, H, D]
      wo_proj(o)       → y        [B, S, dim]
    """

    def __init__(self, config: Wan21Config, device=None):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.scale
        self.causal = config.causal

        # QKV projections (separate, as in Wan2.1) — bf16 for FA4 compatibility
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)

        # QK normalization
        self.norm_q = WanRMSNorm(self.dim, eps=config.eps) if config.qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.dim, eps=config.eps) if config.qk_norm else nn.Identity()

        # RoPE frequencies (registered as buffer, not a parameter)
        dev = device if device is not None else 'cpu'
        self.register_buffer('freqs', build_wan21_freqs(self.head_dim, device=dev), persistent=False)

    def qkv_proj(self, x: torch.Tensor):
        """x [B, S, dim] → q, k, v [B, S, H, D] (pre-RoPE, pre-norm for q/k)."""
        q = self.norm_q(self.q_proj(x)).view(*x.shape[:2], self.num_heads, self.head_dim)
        k = self.norm_k(self.k_proj(x)).view(*x.shape[:2], self.num_heads, self.head_dim)
        v = self.v_proj(x).view(*x.shape[:2], self.num_heads, self.head_dim)
        return q, k, v

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, grid_sizes: torch.Tensor):
        """Apply 3D RoPE to q, k. v is not rotated."""
        q = rope_apply(q, grid_sizes, self.freqs)
        k = rope_apply(k, grid_sizes, self.freqs)
        return q, k

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """FlashAttention-4. q/k/v: [B, S, H, D] → o: [B, S, H, D]."""
        return _fa4_attn(q, k, v, self.scale, self.causal)

    def wo_proj(self, o: torch.Tensor) -> torch.Tensor:
        """o [B, S, H, D] → y [B, S, dim]."""
        return self.o_proj(o.flatten(2))

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Full forward: x [B, S, dim] → y [B, S, dim]. Single-GPU reference path."""
        q, k, v = self.qkv_proj(x)
        q, k = self.apply_rope(q, k, grid_sizes)
        o = self.attention(q, k, v)
        y = self.wo_proj(o)
        return y


def build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, hd):
    """Rank-major [Q,K,V] head-group blocks for fused PRE GEMM.

    rows[d*local_n:(d+1)*local_n] = [Q(d), K(d), V(d)] for rank d.
    This reorders weight rows so the fused GEMM+A2A scatters each rank's Q/K/V head group together.
    """
    rows = local_nh * hd
    blocks = []
    for d in range(sp):
        sl = slice(d * rows, (d + 1) * rows)
        blocks += [Wq[sl], Wk[sl], Wv[sl]]
    return torch.cat(blocks, dim=0).contiguous()
