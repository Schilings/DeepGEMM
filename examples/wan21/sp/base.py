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
        if not self._skip_buffer_creation:
            self._create_buffers()

    def _build_weights(self):
        """Hook for POST-specific parameter layouts; PRE always uses the model Q/K/V."""

    def _create_buffers(self):
        pass

    def destroy_buffers(self):
        pass

    def _pre_forward(self, x_local, llseq):
        """Pure PyTorch synchronous Ulysses PRE, shared by every ablation arm."""
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
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

    def _attn_forward(self, qkv, grid, lbs, lseq):
        """FlashAttention-4, shared identically by both ablation arms."""
        ln = self.local_n
        nh, hd = self.local_nh, self.head_dim
        q = qkv[:, :, :ln].view(lbs, lseq, nh, hd).contiguous()
        k = qkv[:, :, ln:2 * ln].view(lbs, lseq, nh, hd).contiguous()
        v = qkv[:, :, 2 * ln:3 * ln].view(lbs, lseq, nh, hd).contiguous()
        q = rope_apply(q, grid, self.model.freqs).to(torch.bfloat16)
        k = rope_apply(k, grid, self.model.freqs).to(torch.bfloat16)
        return _fa4_attn(q, k, v, self.scale, self.cfg.causal)

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
