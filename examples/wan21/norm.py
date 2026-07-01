"""Wan2.1 normalization layers — faithful to official implementation.

WanRMSNorm: parameter name is 'weight' (not 'scale') to match official checkpoint keys.
  x * rsqrt(mean(x^2) + eps) * weight  (weight = nn.Parameter(ones(dim)))

WanLayerNorm: computed in float32 then cast back (official: super().forward(x.float()).type_as(x)).
"""

import torch
import torch.nn as nn


class WanRMSNorm(nn.Module):
    """RMSNorm for QK normalization. Parameter name 'weight' matches official."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official: self._norm(x.float()).type_as(x) * self.weight
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight.to(x.dtype)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm for pre-norm in attention blocks.

    Official: eps=1e-6, elementwise_affine=False (norm1/norm2) or True (norm3).
    Computed in float32 then cast back.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official: super().forward(x.float()).type_as(x)
        # Need weight/bias in float32 too (may be bf16 after model.to(bf16))
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return torch.nn.functional.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)
