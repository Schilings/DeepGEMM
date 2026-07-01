"""Wan2.1 model — faithful to official Wan-Video/Wan2.1 implementation.

Key differences from my earlier (wrong) version:
  1. Q/K/V are SEPARATE nn.Linear (q, k, v), NOT fused qkv_proj
  2. Pre-norm uses WanLayerNorm (LayerNorm, not RMSNorm) for norm1/norm2
  3. FFN uses GELU(tanh), not SiLU
  4. Has modulation (time embedding e → scale/shift per block)
  5. Has cross-attention (text context) — omitted here for SP attention bench
  6. RoPE: 3D (T/H/W) with split dims (d-4*(d//6), 2*(d//6), 2*(d//6))

Reference: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
"""

import math
import torch
import torch.nn as nn

from flash_attn.cute import flash_attn_func

from .norm import WanRMSNorm, WanLayerNorm
from .rope import build_wan21_freqs, rope_apply
from .config import Wan21Config


def _fa4_attn(q, k, v, scale, causal=False):
    o = flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)
    return o[0] if isinstance(o, tuple) else o


class WanSelfAttention(nn.Module):
    """Wan2.1 self-attention — faithful to official implementation.

    Q/K/V are SEPARATE nn.Linear (as in official code: self.q, self.k, self.v).
    QK-norm (WanRMSNorm) applied after projection, before RoPE.
    """

    def __init__(self, dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = False

        self.q = nn.Linear(dim, num_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.k = nn.Linear(dim, num_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.v = nn.Linear(dim, num_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.o = nn.Linear(num_heads * self.head_dim, dim, bias=False, dtype=torch.bfloat16)

        self.norm_q = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()

        # RoPE frequencies (3D: T/H/W split as in official)
        self.register_buffer('freqs', build_wan21_freqs(self.head_dim), persistent=False)

    def forward(self, x, grid_sizes, freqs):
        B, S, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d)
        k = self.norm_k(self.k(x)).view(B, S, n, d)
        v = self.v(x).view(B, S, n, d)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        o = _fa4_attn(q, k, v, self.scale, self.causal)
        return self.o(o.flatten(2))


class WanFeedForward(nn.Module):
    """Wan2.1 FFN: GELU(tanh), as in official."""

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False, dtype=torch.bfloat16),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim, bias=False, dtype=torch.bfloat16),
        )

    def forward(self, x):
        return self.ffn(x)


class WanAttentionBlock(nn.Module):
    """Wan2.1 attention block with modulation (self-attn + FFN, no cross-attn for SP bench)."""

    def __init__(self, dim, ffn_dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, head_dim, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim, dtype=torch.float32) / dim**0.5)

    def forward(self, x, grid_sizes, freqs, e=None):
        if e is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e_chunks = (self.modulation + e).chunk(6, dim=1)
            h = self.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]
            y = self.self_attn(h.to(x.dtype), grid_sizes, freqs)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e_chunks[2]
            h = self.norm2(x).float() * (1 + e_chunks[4]) + e_chunks[3]
            y = self.ffn(h.to(x.dtype))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e_chunks[5]
        else:
            x = x + self.self_attn(self.norm1(x).to(x.dtype), grid_sizes, freqs)
            x = x + self.ffn(self.norm2(x).to(x.dtype))
        return x


class WanModel(nn.Module):
    """Wan2.1 model (self-attn blocks for SP benchmarking).

    Configs: 14B (dim=5120,nh=40,ffn=13824,layers=40), 1.3B (dim=2048,nh=16,ffn=8192,layers=30)
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

        self.blocks = nn.ModuleList([
            WanAttentionBlock(config.dim, config.ffn_dim, config.num_heads, config.head_dim,
                              config.qk_norm, config.eps)
            for _ in range(config.num_layers)
        ])

        dev = device if device is not None else 'cpu'
        self.register_buffer('freqs', build_wan21_freqs(config.head_dim, device=dev), persistent=False)

    def forward(self, x, grid_sizes, e=None):
        for block in self.blocks:
            x = block(x, grid_sizes, self.freqs, e)
        return x


def build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, hd):
    """Rank-major [Q,K,V] head-group blocks for fused SP PRE GEMM.

    Takes separate Wq, Wk, Wv (official layout) and reorders to rank-major.
    """
    dim = Wq.shape[0]
    rows = local_nh * hd
    blocks = []
    for d in range(sp):
        sl = slice(d * rows, (d + 1) * rows)
        blocks += [Wq[sl], Wk[sl], Wv[sl]]
    return torch.cat(blocks, dim=0).contiguous()
