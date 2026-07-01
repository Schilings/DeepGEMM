"""Wan2.1 normalization layers — faithful to official implementation.

WanRMSNorm: for QK normalization (applied after Q/K projection, before RoPE).
  x * rsqrt(mean(x^2) + eps) * scale  (scale = nn.Parameter(ones(dim)))

WanLayerNorm: for pre-norm in attention blocks (norm1/norm2).
  Official: nn.LayerNorm with elementwise_affine=False, computed in float32 then cast back.
"""

import torch
import torch.nn as nn


class WanRMSNorm(nn.Module):
    """RMSNorm for QK normalization. Scale parameter in bf16."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Official: self.weight = nn.Parameter(torch.ones(dim))
        self.scale = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official: self._norm(x.float()).type_as(x) * self.weight
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.scale


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm for pre-norm in attention blocks.

    Official: eps=1e-6, elementwise_affine=False (for norm1/norm2) or True (for cross_attn_norm).
    Computed in float32 then cast back to input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official: super().forward(x.float()).type_as(x)
        return super().forward(x.float()).to(x.dtype)
