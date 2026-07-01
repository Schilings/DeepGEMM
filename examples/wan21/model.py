"""Wan2.1 model — complete, faithful architecture with fused QKV projection.

QKV is a single large matrix (Wqkv [3*dim, dim]) instead of separate q_proj/k_proj/v_proj.
This is faster (one GEMM vs three) and is the standard approach unless async Ulysses
overlap is needed (which would require splitting Q/K/V for independent A2A).

Architecture: WanSelfAttention (fused QKV + QK-norm + 3D RoPE + FA4 + Wo) + WanFeedForward (SiLU).
Full transformer block = WanSelfAttention + WanFeedForward + residual + RMSNorm.

Model configs:
  14B: dim=5120, nh=40, hd=128, ffn=13824, layers=40
  1.3B: dim=2048, nh=16, hd=128, ffn=8192, layers=30
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.cute import flash_attn_func

from .norm import WanRMSNorm
from .rope import build_wan21_freqs, rope_apply
from .config import Wan21Config


def _fa4_attn(q, k, v, scale, causal=False):
    """FlashAttention-4. q/k/v: [B, S, H, D] → [B, S, H, D]."""
    o = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    return o[0] if isinstance(o, tuple) else o


class WanSelfAttention(nn.Module):
    """Wan2.1 self-attention with fused QKV projection.

    Single qkv_proj weight [3*dim, dim] (NT layout) — one GEMM produces [Q, K, V] concatenated.
    Sub-operations exposed for SP strategies to interleave communication.
    """

    def __init__(self, config: Wan21Config, device=None):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = config.scale
        self.causal = config.causal

        # Fused QKV projection: one big weight [3*dim, dim] (NT layout, like nn.Linear.weight)
        self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False, dtype=torch.bfloat16)
        # Output projection
        self.o_proj = nn.Linear(self.dim, self.dim, bias=False, dtype=torch.bfloat16)

        # QK normalization (applied after QKV split, before RoPE)
        self.norm_q = WanRMSNorm(self.dim, eps=config.eps) if config.qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(self.dim, eps=config.eps) if config.qk_norm else nn.Identity()

        dev = device if device is not None else 'cpu'
        self.register_buffer('freqs', build_wan21_freqs(self.head_dim, device=dev), persistent=False)

    @property
    def wqkv(self):
        """Fused QKV weight [3*dim, dim] (NT layout, = qkv_proj.weight)."""
        return self.qkv_proj.weight

    def qkv_proj_fn(self, x: torch.Tensor):
        """x [*, dim] → qkv [*, 3*dim] (single GEMM, autograd-compatible)."""
        return self.qkv_proj(x)

    def split_qkv(self, qkv: torch.Tensor):
        """qkv [B, S, 3*dim] → q, k, v [B, S, H, D] (with QK-norm applied)."""
        B, S, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.norm_q(q).view(B, S, self.num_heads, self.head_dim)
        k = self.norm_k(k).view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        return q, k, v

    def apply_rope(self, q, k, grid_sizes):
        return rope_apply(q, grid_sizes, self.freqs), rope_apply(k, grid_sizes, self.freqs)

    def attention(self, q, k, v):
        return _fa4_attn(q, k, v, self.scale, self.causal)

    def wo_proj(self, o):
        return self.o_proj(o.flatten(2))

    def forward(self, x, grid_sizes):
        qkv = self.qkv_proj_fn(x)
        q, k, v = self.split_qkv(qkv)
        q, k = self.apply_rope(q, k, grid_sizes)
        o = self.attention(q, k, v)
        return self.wo_proj(o)


class WanFeedForward(nn.Module):
    """Wan2.1 FFN: dim → ffn_dim → dim (SiLU activation, gate + up projection fused)."""

    def __init__(self, config: Wan21Config):
        super().__init__()
        self.dim = config.dim
        self.ffn_dim = config.ffn_dim
        # Fused gate+up projection [2*ffn_dim, dim], then SiLU gate, then down [dim, ffn_dim]
        self.gate_up_proj = nn.Linear(self.dim, 2 * self.ffn_dim, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(self.ffn_dim, self.dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class WanTransformerBlock(nn.Module):
    """Full Wan2.1 transformer block: pre-norm → attn → residual → pre-norm → FFN → residual."""

    def __init__(self, config: Wan21Config, device=None):
        super().__init__()
        self.norm1 = WanRMSNorm(config.dim, eps=config.eps)
        self.attn = WanSelfAttention(config, device=device)
        self.norm2 = WanRMSNorm(config.dim, eps=config.eps)
        self.ffn = WanFeedForward(config)

    def forward(self, x, grid_sizes):
        x = x + self.attn(self.norm1(x), grid_sizes)
        x = x + self.ffn(self.norm2(x))
        return x


class WanModel(nn.Module):
    """Complete Wan2.1 model: input norm → N transformer blocks → output norm.

    Configs:
      14B:  Wan21Config(dim=5120, num_heads=40, ffn_dim=13824, num_layers=40)
      1.3B: Wan21Config(dim=2048, num_heads=16, ffn_dim=8192, num_layers=30)
    """

    PRESETS = {
        '1.3B': Wan21Config(dim=2048, num_heads=16, head_dim=128, ffn_dim=8192, num_layers=30),
        '14B':  Wan21Config(dim=5120, num_heads=40, head_dim=128, ffn_dim=13824, num_layers=40),
    }

    def __init__(self, config: Wan21Config = None, preset: str = None, device=None):
        super().__init__()
        if preset is not None:
            config = self.PRESETS[preset]
        if config is None:
            config = self.PRESETS['1.3B']
        self.config = config

        self.norm_in = WanRMSNorm(config.dim, eps=config.eps)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(config, device=device) for _ in range(config.num_layers)
        ])
        self.norm_out = WanRMSNorm(config.dim, eps=config.eps)

    def forward(self, x, grid_sizes):
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x, grid_sizes)
        return self.norm_out(x)


def build_wqkv_rankmajor(Wqkv_weight, sp, local_nh, hd):
    """Reorder fused QKV weight rows to rank-major [Q,K,V] head-group blocks for SP.

    Input Wqkv_weight [3*dim, dim] has rows laid out as [Q(all heads), K(all heads), V(all heads)].
    Output reorders to rank-major: rows[d*local_n:(d+1)*local_n] = [Q(d), K(d), V(d)] for rank d.
    This makes the fused GEMM+A2A scatter each rank's Q/K/V head group together.
    """
    dim = Wqkv_weight.shape[1]
    nheads = dim // hd
    rows = local_nh * hd
    Wq = Wqkv_weight[:dim]
    Wk = Wqkv_weight[dim:2*dim]
    Wv = Wqkv_weight[2*dim:]
    blocks = []
    for d in range(sp):
        sl = slice(d * rows, (d + 1) * rows)
        blocks += [Wq[sl], Wk[sl], Wv[sl]]
    return torch.cat(blocks, dim=0).contiguous()
