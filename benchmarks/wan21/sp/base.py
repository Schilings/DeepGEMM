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
        self.model = WanSelfAttention(config)
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
        Wq = self.model.q_proj.weight
        Wk = self.model.k_proj.weight
        Wv = self.model.v_proj.weight
        Wo = self.model.o_proj.weight
        self.Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, self.sp_size, self.local_nh, self.head_dim)
        self.Wqkv_t = self.Wqkv.t().contiguous()
        self.Wo = Wo
        self.Wo_t = Wo.t().contiguous()

    def _create_buffers(self):
        pass

    def destroy_buffers(self):
        pass

    def _attn_forward(self, qkv, grid, lbs, lseq):
        ln = self.local_n
        q = qkv[:, :, :ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()
        k = qkv[:, :, ln:2*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()
        v = qkv[:, :, 2*ln:3*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()
        q = self.model.norm_q(q.reshape(-1, self.cfg.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        k = self.model.norm_k(k.reshape(-1, self.cfg.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        q = rope_apply(q, grid, self.model.freqs)
        k = rope_apply(k, grid, self.model.freqs)
        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=self.cfg.causal)
        return o[0] if isinstance(o, tuple) else o

    def _attn_backward(self, grad_attn, qkv_pre_norm, grid, lbs, lseq):
        ln = self.local_n
        q_leaf = qkv_pre_norm[:, :, :ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)
        k_leaf = qkv_pre_norm[:, :, ln:2*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)
        v_leaf = qkv_pre_norm[:, :, 2*ln:3*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)
        q = self.model.norm_q(q_leaf.reshape(-1, self.cfg.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        k = self.model.norm_k(k_leaf.reshape(-1, self.cfg.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        q = rope_apply(q, grid, self.model.freqs)
        k = rope_apply(k, grid, self.model.freqs)
        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v_leaf, softmax_scale=self.scale, causal=self.cfg.causal)
        o = o[0] if isinstance(o, tuple) else o
        return torch.autograd.grad(o, [q_leaf, k_leaf, v_leaf], grad_attn)

    def _pre_forward(self, x_local, llseq):
        raise NotImplementedError

    def _post_forward(self, o, **kw):
        raise NotImplementedError

    def _post_backward(self, grad_y, cache, **kw):
        raise NotImplementedError

    def _pre_backward(self, grad_qkv, **kw):
        raise NotImplementedError

    def forward(self, x_local, grid, llseq=None):
        assert self._shape_set
        if llseq is None: llseq = self.local_seq
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        qkv = self._pre_forward(x_local, llseq)
        o = self._attn_forward(qkv, grid, lbs, lseq)
        y = self._post_forward(o, lbs=lbs, lseq=lseq, llseq=llseq, grid=grid)
        self._cache = (qkv, o)
        return y

    def backward(self, grad_y, x_local, grid, llseq=None):
        assert self._shape_set
        if llseq is None: llseq = self.local_seq
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        lm = lbs * llseq
        qkv, o = self._cache

        grad_attn, grad_Wo = self._post_backward(
            grad_y, cache=o, lbs=lbs, lseq=lseq, llseq=llseq, lm=lm, grid=grid)
        grad_q, grad_k, grad_v = self._attn_backward(grad_attn, qkv, grid, lbs, lseq)
        ln = self.local_n
        grad_qkv = torch.cat([
            grad_q.reshape(lbs, lseq, ln),
            grad_k.reshape(lbs, lseq, ln),
            grad_v.reshape(lbs, lseq, ln)], dim=-1)
        grad_X, grad_Wqkv = self._pre_backward(
            grad_qkv, lbs=lbs, lseq=lseq, llseq=llseq, lm=lm, x_local=x_local)
        # Gradient sync: Wqkv is always replicated → all-reduce.
        # Wo: replicated for serial/fused_std → all-reduce; row-sharded for fused_var → no sync.
        dist.all_reduce(grad_Wqkv, op=dist.ReduceOp.SUM, group=self.group)
        if not getattr(self, '_wo_sharded', False):
            dist.all_reduce(grad_Wo, op=dist.ReduceOp.SUM, group=self.group)
        return grad_X, grad_Wqkv, grad_Wo
