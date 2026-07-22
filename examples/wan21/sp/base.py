"""SP strategy base — defines the forward/backward interface for all strategies.

Key insight on the dual relationship:
  Forward GEMM+A2A (PRE)  <->  Backward A2A+GEMM (uses the POST forward kernel)
  Forward A2A+GEMM (POST) <->  Backward GEMM+A2A (uses the PRE forward kernel)
  Forward GEMM+RS (POSTv) <->  Backward AG+GEMM (uses bf16_ag_gemm_nt)

Fused kernels compute ONLY activation gradients (not weight gradients) because
the communication is overlapped with the GEMM epilogue, and weight gradients
need the full input which isn't available during the overlapped GEMM.
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist

from ..config import Wan21Config, SPConfig
from ..model import WanSelfAttention, _fa4_attn
from ..rope import rope_apply

try:
    from flash_attn import flash_attn_varlen_func as _fa_varlen
    _HAS_VARLEN = True
except ImportError:
    _HAS_VARLEN = False


def _fa_varlen_attn(q, k, v, cu_seqlens, max_seqlen, scale, causal=False):
    """FlashAttention varlen for packed THD tensors.

    q/k/v: [total_tokens, nh, hd] (no batch dim, no padding)
    cu_seqlens: [0, s1, s1+s2, ...] cumulative sequence lengths
    """
    # flash_attn_varlen_func expects int32 cu_seqlens on device
    cu = cu_seqlens.to(q.device, dtype=torch.int32)
    out = _fa_varlen(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=scale, causal=causal,
    )
    return out[0] if isinstance(out, tuple) else out


class NCCLAllToAll(torch.autograd.Function):
    """Synchronous NCCL all-to-all with the inverse collective in backward."""

    @staticmethod
    def forward(ctx, send, group):
        ctx.group = group
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def backward(ctx, grad_recv):
        grad_recv = grad_recv.contiguous()
        grad_send = torch.empty_like(grad_recv)
        dist.all_to_all_single(grad_send, grad_recv, group=ctx.group)
        return grad_send, None


class UlyssesBase(nn.Module):
    """Base class. Holds model + SP config + shared forward logic (pre/attn)."""

    def __init__(self, config: Wan21Config, sp_config: SPConfig):
        super().__init__()
        self.cfg = config
        self.sp = sp_config
        self.model = WanSelfAttention(config.dim, config.num_heads, config.head_dim,
                                      qk_norm=config.qk_norm, eps=config.eps)
        self.scale = config.scale
        self.sp_size = sp_config.sp_size
        self.group = sp_config.group
        self.layout = sp_config.layout
        self.use_fused = sp_config.use_fused_ops
        self._shape_set = False
        self._skip_buffer_creation = False

    def setup_shape(self, bs, seq, nheads, head_dim):
        sp = self.sp_size
        assert nheads % sp == 0 and seq % sp == 0
        self.bs = bs
        self.seq = seq
        self.local_nh = nheads // sp
        self.head_dim = head_dim
        self.local_n = self.local_nh * head_dim
        self.local_hidden = self.local_n
        self.local_nqkv = 3 * self.local_n
        self.local_seq = seq // sp
        self.local_m = bs * (seq // sp)
        self._shape_set = True
        self._build_weights()
        if not self._skip_buffer_creation:
            self._create_buffers()

    def _build_weights(self):
        """Hook for POST-specific parameter layouts; PRE always uses the model Q/K/V."""

    def _create_buffers(self):
        pass

    def destroy_buffers(self):
        pass

    def _pre_forward(self, x_local, llseq):
        """Pure PyTorch synchronous Ulysses PRE, shared by every ablation arm.

        Works for both fixed-shape and THD packed modes:
        - Fixed: x_local [local_seq, dim], llseq = local_seq
        - Packed: x_local [total_local_tokens, dim], llseq = total_local_tokens
        """
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        # lseq = total tokens after A2A gather = llseq * sp
        lseq = llseq * sp
        hd = self.head_dim

        q = self.model.norm_q(self.model.q(x_local)).view(lbs, llseq, sp, self.local_nh, hd)
        k = self.model.norm_k(self.model.k(x_local)).view(lbs, llseq, sp, self.local_nh, hd)
        v = self.model.v(x_local).view(lbs, llseq, sp, self.local_nh, hd)

        def scatter_heads(tensor):
            send = tensor.permute(2, 0, 1, 3, 4).contiguous()
            recv = NCCLAllToAll.apply(send, self.group)
            return recv.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)

        q = scatter_heads(q)
        k = scatter_heads(k)
        v = scatter_heads(v)
        return torch.cat((q, k, v), dim=2).reshape(lbs, lseq, -1)

    def _attn_forward(self, qkv, grid, lbs, lseq, cu_seqlens=None):
        """FlashAttention, shared by all strategies.

        Two modes:
        - Fixed shape (default): qkv [bs, seq, 3*local_n] → flash_attn_func
        - Packed THD (cu_seqlens given): qkv [total_tokens, 3*local_n]
          → flash_attn_varlen_func, no padding
        """
        ln = self.local_n
        nh, hd = self.local_nh, self.head_dim

        if cu_seqlens is not None:
            # THD packed mode: qkv from _pre_forward is [1, total_tokens, 3*local_n]
            # → [total_tokens, nh, hd]
            qkv_flat = qkv.reshape(-1, ln * 3)  # [total_tokens, 3*local_n]
            total = qkv_flat.shape[0]
            q = qkv_flat[:, :ln].view(total, nh, hd).contiguous()
            k = qkv_flat[:, ln:2*ln].view(total, nh, hd).contiguous()
            v = qkv_flat[:, 2*ln:3*ln].view(total, nh, hd).contiguous()
            # RoPE: apply per-segment using grid_sizes
            # grid is [num_seqs, 3], each row = (F, H, W)
            q = self._rope_packed(q, grid, cu_seqlens).to(torch.bfloat16)
            k = self._rope_packed(k, grid, cu_seqlens).to(torch.bfloat16)
            max_s = max(cu_seqlens[i+1] - cu_seqlens[i] for i in range(len(cu_seqlens)-1))
            return _fa_varlen_attn(q, k, v, cu_seqlens, max_s, self.scale, self.cfg.causal)
        else:
            # Fixed shape mode (original)
            q = qkv[:, :, :ln].view(lbs, lseq, nh, hd).contiguous()
            k = qkv[:, :, ln:2 * ln].view(lbs, lseq, nh, hd).contiguous()
            v = qkv[:, :, 2 * ln:3 * ln].view(lbs, lseq, nh, hd).contiguous()
            q = rope_apply(q, grid, self.model.freqs).to(torch.bfloat16)
            k = rope_apply(k, grid, self.model.freqs).to(torch.bfloat16)
            return _fa4_attn(q, k, v, self.scale, self.cfg.causal)

    def _rope_packed(self, x, grid_sizes, cu_seqlens):
        """Apply 3D RoPE to packed THD tensor.

        x: [total_tokens, nh, hd]
        grid_sizes: [num_seqs, 3] — each row (F, H, W)
        cu_seqlens: [num_seqs+1] — cumulative offsets

        Fast path: if all sequences have the same grid, apply rope_apply
        once to the full tensor (treating it as batched).
        """
        # Fast path: single sequence or all same grid
        if grid_sizes.shape[0] == 1:
            x = x.unsqueeze(0)  # [1, total, nh, hd]
            return rope_apply(x, grid_sizes, self.model.freqs).squeeze(0)

        # Check if all grids are identical
        first = grid_sizes[0]
        if (grid_sizes == first).all():
            # All same grid — but cu_seqlens may have different lengths
            # Use per-segment rope_apply but batch the grid
            x = x.unsqueeze(0)
            return rope_apply(x, grid_sizes, self.model.freqs).squeeze(0)

        # Slow path: per-segment RoPE
        x = x.unsqueeze(0)  # [1, total, nh, hd]
        out = torch.empty_like(x)
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            seg = x[:, s:e]
            seg = rope_apply(seg, torch.tensor([[f, h, w]], device=x.device),
                             self.model.freqs)
            out[:, s:e] = seg
        return out.squeeze(0)

    def _post_forward(self, o, **kw):
        raise NotImplementedError

    def forward(self, x_local, grid, llseq=None, cu_seqlens=None):
        """Forward pass — autograd graph (for FSDP2 + loss.backward()).

        Args:
            x_local:    [local_seq, dim] or [total_local_tokens, dim] (packed).
            grid:       RoPE grid [1, 3] or [num_seqs, 3].
            llseq:      Local token count (defaults to self.local_seq).
            cu_seqlens: If given, use flash_attn_varlen (THD packed mode).
        """
        assert self._shape_set
        sp = self.sp_size
        if llseq is None: llseq = self.local_seq
        lbs = self.bs if self.layout == 'BSHD' else 1
        if cu_seqlens is not None:
            # THD packed: lseq = total_tokens after A2A gather
            lseq = llseq * sp
        else:
            lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        qkv = self._pre_forward(x_local, llseq)
        o = self._attn_forward(qkv, grid, lbs, lseq, cu_seqlens=cu_seqlens)
        y = self._post_forward(o, lbs=lbs, lseq=lseq, llseq=llseq, grid=grid)
        return y
