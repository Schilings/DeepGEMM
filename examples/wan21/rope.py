"""3D RoPE for Wan2.1 — faithful port of the official implementation.

Splits head_dim into three axes: time (T), height (H), width (W), applying independent
rotary embeddings to each axis. This encodes 3D video structure (frame × height × width)
into the attention positional information.
"""

import math
import torch


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Pre-compute RoPE frequencies as complex numbers [max_seq_len, dim//2]."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def build_wan21_freqs(head_dim: int, max_seq_len: int = 1024,
                      device='cpu', dtype=torch.complex64) -> torch.Tensor:
    """Build Wan2.1 3D RoPE frequency table.

    head_dim split: [d-4*(d//6), 2*(d//6), 2*(d//6)] for T/H/W axes.
    For head_dim=128: [44, 42, 42] → 128 real = 64 complex.
    """
    d = head_dim
    freqs = torch.cat([
        rope_params(max_seq_len, d - 4 * (d // 6)),   # time axis
        rope_params(max_seq_len, 2 * (d // 6)),        # height axis
        rope_params(max_seq_len, 2 * (d // 6)),         # width axis
    ], dim=1)
    return freqs.to(device=device, dtype=dtype)


def rope_apply(q: torch.Tensor, grid_sizes: torch.Tensor,
               freqs: torch.Tensor) -> torch.Tensor:
    """Apply 3D RoPE to q [B, S, H, D]. Returns rotated q [B, S, H, D].

    grid_sizes = (F, H, W) for the video grid.
    Only the first F*H*W tokens get RoPE; rest are padding (unchanged).
    Preserves input dtype (bf16).
    """
    B, S, H, D = q.shape
    c = D // 2
    freqs_parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            q[i, :seq_len].to(torch.float64).reshape(seq_len, H, -1, 2))
        freqs_i = torch.cat([
            freqs_parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        if S > seq_len:
            x_i = torch.cat([x_i, q[i, seq_len:].float()])
        output.append(x_i)
    return torch.stack(output).float()


def rope_inverse(grad_q: torch.Tensor, grid_sizes: torch.Tensor,
                 freqs: torch.Tensor) -> torch.Tensor:
    """Inverse RoPE for backward (conjugate rotation). Same as rope_apply with conj freqs."""
    return rope_apply(grad_q, grid_sizes, torch.conj(freqs))
