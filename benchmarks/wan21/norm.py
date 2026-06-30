"""WanRMSNorm — QK normalization used in Wan2.1 attention.

RMSNorm: x * rsqrt(mean(x^2) + eps) * scale
Preserves input dtype (bf16) throughout.
"""

import torch
import torch.nn as nn


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return (x * norm * self.scale).to(x.dtype)
