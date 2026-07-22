"""Wan2.1 transformer-block training core with Ulysses self-attention."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from .model import WanFeedForward, WanLayerNorm, WanT2VCrossAttention
from .sp.fused import FusedUlysses
from .sp.serial import SerialUlysses
from .sp.variant import FusedVariantUlysses


def _make_self_attention(strategy: str, config, sp_config):
    layer_sp_config = replace(sp_config)
    if strategy == "serial":
        return SerialUlysses(config, layer_sp_config)
    if strategy == "fused":
        return FusedUlysses(config, layer_sp_config)
    if strategy == "fused_var":
        return FusedVariantUlysses(config, layer_sp_config)
    raise ValueError(f"Unknown strategy: {strategy}")


class SPWanAttentionBlock(nn.Module):
    """Official T2V block whose self-attention is sequence parallel."""

    def __init__(self, config, sp_config, strategy: str):
        super().__init__()
        dim = config.dim
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.self_attn = _make_self_attention(strategy, config, sp_config)
        self.norm3 = WanLayerNorm(dim, eps=config.eps, elementwise_affine=True)
        self.cross_attn = WanT2VCrossAttention(
            dim, config.num_heads, config.head_dim, config.qk_norm, config.eps
        )
        self.norm2 = WanLayerNorm(dim, eps=config.eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim, config.ffn_dim)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x_local, e, grid, context, cu_seqlens=None):
        llseq = x_local.shape[0]
        e_chunks = tuple(
            chunk.squeeze(0) for chunk in
            (self.modulation.float() + e.float()).chunk(6, dim=1)
        )

        h = self.norm1(x_local).float() * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(h, grid, llseq, cu_seqlens=cu_seqlens)
        x_local = x_local.float() + y.float() * e_chunks[2]

        x_batched = x_local.unsqueeze(0)
        cross = self.cross_attn(self.norm3(x_batched), context).squeeze(0)
        x_local = x_local + cross.float()

        h = self.norm2(x_local).float() * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(h)
        return x_local + y.float() * e_chunks[5]


class SPWanTransformer(nn.Module):
    """The 40-block Wan2.1 14B transformer training core.

    Patch/text/time embeddings and the output head are intentionally outside this
    benchmark: the 40 transformer blocks contain nearly all model parameters and
    are the region affected by the POST self-attention ablation.
    """

    def __init__(self, config, sp_config, num_layers: int, strategy: str):
        super().__init__()
        self.strategy = strategy
        self.blocks = nn.ModuleList(
            SPWanAttentionBlock(config, sp_config, strategy) for _ in range(num_layers)
        )

    @staticmethod
    def official_key(local_name: str) -> str:
        return (
            local_name.replace(".self_attn.model.", ".self_attn.")
            .replace(".ffn.ffn.", ".ffn.")
        )

    def setup_shape(self, batch_size, sequence_length, num_heads, head_dim):
        for index, block in enumerate(self.blocks):
            attention = block.self_attn
            if index > 0:
                attention._skip_buffer_creation = True
            attention.setup_shape(batch_size, sequence_length, num_heads, head_dim)
        if self.blocks:
            owner = self.blocks[0].self_attn
            for block in self.blocks[1:]:
                if getattr(owner, "sym_post", None) is not None:
                    block.self_attn.share_buffers_from(owner)

    def forward(self, x_local, e, grid, context, cu_seqlens=None):
        for block in self.blocks:
            x_local = block(x_local, e, grid, context, cu_seqlens=cu_seqlens)
        return x_local

    def destroy_buffers(self):
        for block in self.blocks:
            block.self_attn.destroy_buffers()

    def sym_buffer_bytes(self) -> int:
        if not self.blocks:
            return 0
        workspace = getattr(self.blocks[0].self_attn, "sym_post", None)
        return 0 if workspace is None else workspace.num_bytes
