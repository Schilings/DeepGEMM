"""Wan2.1 model — faithful to official Wan-Video/Wan2.1 implementation.

Matches official weight keys exactly:
  blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}     — with bias=True
  blocks.{i}.self_attn.{norm_q,norm_k}.weight      — RMSNorm scale
  blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}    — cross-attention (T2V)
  blocks.{i}.cross_attn.{norm_q,norm_k}.weight
  blocks.{i}.norm3.{weight,bias}                   — LayerNorm (elementwise_affine=True)
  blocks.{i}.ffn.0.{weight,bias} + ffn.2.{weight,bias}  — GELU FFN
  blocks.{i}.modulation                            — [1, 6, dim] float32

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
    """Wan2.1 self-attention — faithful to official (bias=True on all projections)."""

    def __init__(self, dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = False

        # bias=True to match official weights
        self.q = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.k = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.v = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.o = nn.Linear(num_heads * self.head_dim, dim, bias=True)

        self.norm_q = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.register_buffer('freqs', build_wan21_freqs(self.head_dim), persistent=False)

    def forward(self, x, grid_sizes, freqs):
        B, S, _ = x.shape
        n, d = self.num_heads, self.head_dim
        # FA4 requires bf16; compute QKV in input dtype, cast to bf16 for attention
        q = self.norm_q(self.q(x)).view(B, S, n, d).to(torch.bfloat16)
        k = self.norm_k(self.k(x)).view(B, S, n, d).to(torch.bfloat16)
        v = self.v(x).view(B, S, n, d).to(torch.bfloat16)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        o = _fa4_attn(q, k, v, self.scale, self.causal)
        return self.o(o.flatten(2).to(x.dtype))


class WanT2VCrossAttention(nn.Module):
    """Wan2.1 T2V cross-attention (text context → video). Faithful to official."""

    def __init__(self, dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.k = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.v = nn.Linear(dim, num_heads * self.head_dim, bias=True)
        self.o = nn.Linear(num_heads * self.head_dim, dim, bias=True)

        self.norm_q = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None):
        B = x.size(0)
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, -1, n, d).to(torch.bfloat16)
        k = self.norm_k(self.k(context)).view(B, -1, n, d).to(torch.bfloat16)
        v = self.v(context).view(B, -1, n, d).to(torch.bfloat16)
        o = _fa4_attn(q, k, v, self.scale, causal=False)
        return self.o(o.flatten(2).to(x.dtype))


class WanFeedForward(nn.Module):
    """Wan2.1 FFN: Sequential(Linear, GELU(tanh), Linear) — with bias, as in official."""

    def __init__(self, dim, ffn_dim):
        super().__init__()
        # Official: nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))
        # ffn.0 = first Linear, ffn.2 = second Linear (index 1 is GELU)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim, bias=True),
        )

    def forward(self, x):
        return self.ffn(x)


class WanAttentionBlock(nn.Module):
    """Wan2.1 attention block — faithful to official WanAttentionBlock.

    Structure: norm1 → self_attn (with modulation) → residual → norm3 → cross_attn → residual →
               norm2 → ffn (with modulation) → residual.
    """

    def __init__(self, dim, ffn_dim, num_heads, head_dim=None,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim

        # norm1/norm2: LayerNorm without affine (modulation provides scale/shift)
        self.norm1 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, head_dim, qk_norm, eps)

        # norm3: LayerNorm with affine (for cross-attention) or Identity
        self.norm3 = (WanLayerNorm(dim, eps=eps, elementwise_affine=True)
                      if cross_attn_norm else nn.Identity())
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, head_dim, qk_norm, eps)

        self.norm2 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, ffn_dim)

        # Modulation: [1, 6, dim] float32
        self.modulation = nn.Parameter(torch.randn(1, 6, dim, dtype=torch.float32) / dim**0.5)

    def forward(self, x, grid_sizes, freqs, e, context, context_lens=None):
        """x [B, L, dim], e [B, 6, dim] modulation, context [B, L2, dim] text embeddings."""
        assert e is not None, "Modulation e is required (time embedding)"

        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_chunks = (self.modulation + e).chunk(6, dim=1)

        # self-attention with modulation
        h = self.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(h.to(x.dtype), grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_chunks[2]

        # cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # FFN with modulation
        h = self.norm2(x).float() * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(h.to(x.dtype))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_chunks[5]

        return x


class WanModel(nn.Module):
    """Wan2.1 model (transformer blocks for SP benchmarking).

    Full official WanModel also has patch_embedding, text_embedding, time_embedding,
    time_projection, and head. This version includes the transformer blocks for SP
    benchmarking of the self-attention layer.

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
            config = self.PRESETS['14B']
        self.config = config

        self.blocks = nn.ModuleList([
            WanAttentionBlock(config.dim, config.ffn_dim, config.num_heads, config.head_dim,
                              config.qk_norm, config.cross_attn_norm, config.eps)
            for _ in range(config.num_layers)
        ])

        dev = device if device is not None else 'cpu'
        self.register_buffer('freqs', build_wan21_freqs(config.head_dim, device=dev), persistent=False)

    def forward(self, x, grid_sizes, e, context, context_lens=None):
        """x [B, L, dim], e [B, 6, dim], context [B, L2, dim] → [B, L, dim]."""
        for block in self.blocks:
            x = block(x, grid_sizes, self.freqs, e, context, context_lens)
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
