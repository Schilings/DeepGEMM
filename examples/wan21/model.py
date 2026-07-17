"""Wan2.1 model — complete, faithful implementation of official Wan-Video/Wan2.1.

Implements the full WanModel: patch_embedding, text_embedding, time_embedding,
time_projection, transformer blocks (self-attn + cross-attn + FFN + modulation),
and head. Weight keys match official checkpoint exactly.

Reference: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import WanRMSNorm, WanLayerNorm
from .config import Wan21Config


def sinusoidal_embedding_1d(dim, position):
    """Sinusoidal time embedding (official)."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """RoPE frequency params (complex, official)."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """3D RoPE apply (T/H/W axes, official). x [B, S, N, D] → [B, S, N, D]."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def build_wan21_freqs(head_dim, device='cpu'):
    """Build RoPE freqs exactly as official: cat([rope_params(1024, d-4*(d//6)), ...])."""
    d = head_dim
    return torch.cat([
        rope_params(1024, d - 4 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
        rope_params(1024, 2 * (d // 6)),
    ], dim=1).to(device)


def _torch_attn(q, k, v, scale, causal=False):
    """Torch SDPA for BSHD tensors; CUDA selects its memory-efficient backend."""
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=causal, scale=scale)
    return o.transpose(1, 2).contiguous()


class WanSelfAttention(nn.Module):
    """Self-attention (official: separate q/k/v/o with bias, QK RMSNorm)."""

    def __init__(self, dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.causal = False

        self.q = nn.Linear(dim, num_heads * self.head_dim)
        self.k = nn.Linear(dim, num_heads * self.head_dim)
        self.v = nn.Linear(dim, num_heads * self.head_dim)
        self.o = nn.Linear(num_heads * self.head_dim, dim)

        self.norm_q = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        # Keep RoPE frequencies complex when the module is cast to bf16.
        self.register_buffer('freqs', build_wan21_freqs(self.head_dim), persistent=False)

    def _apply(self, fn, recurse=True):
        freqs = self._buffers.pop('freqs')
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._buffers['freqs'] = freqs.to(device=self.q.weight.device)
        return result

    def forward(self, x, grid_sizes, freqs):
        B, S, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d).to(torch.bfloat16)
        k = self.norm_k(self.k(x)).view(B, S, n, d).to(torch.bfloat16)
        v = self.v(x).view(B, S, n, d).to(torch.bfloat16)
        q = rope_apply(q, grid_sizes, freqs).to(torch.bfloat16)
        k = rope_apply(k, grid_sizes, freqs).to(torch.bfloat16)
        o = _torch_attn(q, k, v, self.scale, self.causal)
        return self.o(o.flatten(2).to(x.dtype))


class WanT2VCrossAttention(nn.Module):
    """T2V cross-attention (text context → video)."""

    def __init__(self, dim, num_heads, head_dim=None, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q = nn.Linear(dim, num_heads * self.head_dim)
        self.k = nn.Linear(dim, num_heads * self.head_dim)
        self.v = nn.Linear(dim, num_heads * self.head_dim)
        self.o = nn.Linear(num_heads * self.head_dim, dim)

        self.norm_q = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(num_heads * self.head_dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens=None):
        B = x.size(0)
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, -1, n, d).to(torch.bfloat16)
        k = self.norm_k(self.k(context)).view(B, -1, n, d).to(torch.bfloat16)
        v = self.v(context).view(B, -1, n, d).to(torch.bfloat16)
        o = _torch_attn(q, k, v, self.scale, causal=False)
        return self.o(o.flatten(2).to(x.dtype))


class WanFeedForward(nn.Module):
    """FFN: Sequential(Linear, GELU(tanh), Linear) — with bias."""

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        return self.ffn(x)


class WanAttentionBlock(nn.Module):
    """Full attention block: self-attn + cross-attn + FFN, with modulation."""

    def __init__(self, dim, ffn_dim, num_heads, head_dim=None,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = WanSelfAttention(dim, num_heads, head_dim, qk_norm, eps)
        self.norm3 = (WanLayerNorm(dim, eps=eps, elementwise_affine=True)
                      if cross_attn_norm else nn.Identity())
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, head_dim, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes, freqs, context, context_lens=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_chunks = (self.modulation + e).chunk(6, dim=1)
        # self-attention with modulation
        h = self.norm1(x).float() * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(h, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_chunks[2]
        # cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        # FFN with modulation
        h = self.norm2(x).float() * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(h)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e_chunks[5]
        return x


class Head(nn.Module):
    """Output head with modulation (official)."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        out_dim_proj = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim_proj)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e_chunks = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + e_chunks[1]) + e_chunks[0])


class WanModel(nn.Module):
    """Complete Wan2.1 model — faithful to official implementation.

    Includes: patch_embedding, text_embedding, time_embedding, time_projection,
    transformer blocks, head, unpatchify.

    Configs:
      14B:  dim=5120, nh=40, ffn=13824, layers=40, text_dim=4096
      1.3B: dim=2048, nh=16, ffn=8192, layers=30, text_dim=4096
    """

    PRESETS = {
        '1.3B': dict(dim=2048, num_heads=16, head_dim=128, ffn_dim=8192, num_layers=30),
        '14B':  dict(dim=5120, num_heads=40, head_dim=128, ffn_dim=13824, num_layers=40),
    }

    def __init__(self, dim=5120, num_heads=40, head_dim=128, ffn_dim=13824, num_layers=40,
                 patch_size=(1, 2, 2), text_len=512, in_dim=16, freq_dim=256,
                 text_dim=4096, out_dim=16, qk_norm=True, cross_attn_norm=True,
                 eps=1e-6, preset=None, device=None):
        super().__init__()
        if preset is not None:
            p = self.PRESETS[preset]
            dim, num_heads, head_dim, ffn_dim, num_layers = (p['dim'], p['num_heads'],
                p['head_dim'], p['ffn_dim'], p['num_layers'])

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, head_dim, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE freqs (not a buffer — official keeps it as plain attribute to avoid dtype cast)
        d = head_dim
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ], dim=1)
        if device is not None:
            self.freqs = self.freqs.to(device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        """Full forward pass.

        Args:
            x: List of [C_in, F, H, W] video tensors
            t: Timesteps [B]
            context: List of [L, C] text embeddings
            seq_len: Max sequence length for padding
        Returns:
            List of [C_out, F, H, W] output tensors
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embedding
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))

        # transformer blocks
        kwargs = dict(e=e0, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=context_lens)
        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        return self.unpatchify(x, grid_sizes)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return [u.float() for u in out]


def build_wqkv_rankmajor(Wq, Wk, Wv, sp, local_nh, hd):
    """Rank-major [Q,K,V] head-group blocks for fused SP PRE GEMM."""
    dim = Wq.shape[0]
    rows = local_nh * hd
    blocks = []
    for d in range(sp):
        sl = slice(d * rows, (d + 1) * rows)
        blocks += [Wq[sl], Wk[sl], Wv[sl]]
    return torch.cat(blocks, dim=0).contiguous()
