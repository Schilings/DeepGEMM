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
from ..model import WanSelfAttention, build_wqkv_rankmajor
from ..rope import rope_apply
from ..norm import WanRMSNorm


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

    def setup_shape(self, bs, seq, nheads, head_dim):
        sp = self.sp_size
        assert nheads % sp == 0 and seq % sp == 0 and (seq // sp) % 128 == 0
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
        self._create_buffers()

    def _build_weights(self):
        # Model has separate q/k/v (official Wan2.1 layout) — reorder to rank-major for SP
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, self.sp_size, self.local_nh, self.head_dim)
        self.Wqkv = nn.Parameter(Wqkv.clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        self.Wo = self.model.o.weight  # [dim, dim], managed by FSDP2 via model
        self.Wo_t = self.Wo.t().contiguous()

    def _create_buffers(self):
        pass

    def destroy_buffers(self):
        pass

    def _attn_forward(self, qkv, grid, lbs, lseq):
        """Attention via FA4 — qkv already normed + A2A'd by _pre_forward."""
        ln = self.local_n
        nh, hd = self.local_nh, self.head_dim
        q = qkv[:, :, :ln].view(lbs, lseq, nh, hd).contiguous()
        k = qkv[:, :, ln:2*ln].view(lbs, lseq, nh, hd).contiguous()
        v = qkv[:, :, 2*ln:3*ln].view(lbs, lseq, nh, hd).contiguous()
        q = rope_apply(q, grid, self.model.freqs).to(torch.bfloat16)
        k = rope_apply(k, grid, self.model.freqs).to(torch.bfloat16)
        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=self.cfg.causal)
        return o[0] if isinstance(o, tuple) else o

    def _pre_forward(self, x_local, llseq):
        raise NotImplementedError

    def _post_forward(self, o, **kw):
        raise NotImplementedError

    def forward(self, x_local, grid, llseq=None):
        """Forward pass — autograd graph (for FSDP2 + loss.backward())."""
        assert self._shape_set
        if llseq is None: llseq = self.local_seq
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        qkv = self._pre_forward(x_local, llseq)
        o = self._attn_forward(qkv, grid, lbs, lseq)
        y = self._post_forward(o, lbs=lbs, lseq=lseq, llseq=llseq, grid=grid)
        return y
