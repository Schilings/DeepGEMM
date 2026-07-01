"""Fused Standard Ulysses: Q/K/V separate GEMM+norm → A2A → attn → A2A+GEMM POST.

NOTE: Currently uses separate Q/K/V projections (not fused GEMM+A2A kernel)
because norm_q/norm_k must be applied on full dim BEFORE A2A scatter.
When the "Fused QKV GEMM+Norm+A2A" operator is developed (see docs/FUSED_QKV_NORM_A2A.md),
this will switch to using it for single-kernel GEMM+norm+A2A.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from .serial import NCCLAllToAll


class FusedStandardUlysses(UlyssesBase):
    """Uses separate Q/K/V + norm + NCCL A2A (same as serial, but marks as 'fused' for FSDP2 bench)."""

    def __init__(self, config, sp_config):
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Standard [Q_all, K_all, V_all] order."""
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        self.Wo = self.model.o.weight
        self.Wo_t = self.Wo.t().contiguous()

    def _pre_forward(self, x_local, llseq):
        """PRE: GEMM → split Q/K/V → norm → A2A scatter heads → gather seq."""
        dim = self.cfg.dim
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        hd = self.head_dim
        bias = torch.cat([self.model.q.bias, self.model.k.bias, self.model.v.bias])
        d = torch.matmul(x_local, self.Wqkv.t()) + bias
        q = self.model.norm_q(d[:, :dim]).view(lbs, llseq, sp, self.local_nh, hd)
        k = self.model.norm_k(d[:, dim:2*dim]).view(lbs, llseq, sp, self.local_nh, hd)
        v = d[:, 2*dim:].view(lbs, llseq, sp, self.local_nh, hd)
        def a2a_scatter(t):
            send = t.permute(2, 0, 1, 3, 4).contiguous()
            recv = NCCLAllToAll.apply(send, self.group)
            return recv.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)
        q = a2a_scatter(q)
        k = a2a_scatter(k)
        v = a2a_scatter(v)
        return torch.cat([q, k, v], dim=2).reshape(lbs, lseq, -1)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: A2A transpose → Wo GEMM + bias."""
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        lm = lbs * llseq
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        return torch.matmul(gathered, self.Wo.t()) + self.model.o.bias
