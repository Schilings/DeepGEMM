"""Fused Variant Ulysses: Q/K/V separate GEMM+norm → A2A → attn → Wo GEMM (row-split POST).

Wo is row-split (N-sharded) → weight grad is local, no FSDP2 sync needed.

NOTE: Currently uses separate Q/K/V + NCCL A2A (not fused kernel) for correctness.
When "Fused QKV GEMM+Norm+A2A" operator is developed, PRE will switch to it.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from .serial import NCCLAllToAll


class FusedVariantUlysses(UlyssesBase):
    """Q/K/V separate + Wo row-split (N-sharded output)."""

    def __init__(self, config, sp_config):
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Standard [Q_all, K_all, V_all] order + Wo row-split (N-sharded)."""
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        # Wo row-split: each rank gets [local_N, dim] slice of Wo [dim, dim]
        local_N = self.cfg.dim // self.sp_size
        rank = self.group.rank()
        Wo = self.model.o.weight  # [dim, dim]
        Wo_r = Wo[rank * local_N:(rank + 1) * local_N, :].contiguous()  # [local_N, dim]
        self.Wo_r_local = nn.Parameter(Wo_r.clone(), requires_grad=True)
        self.Wo_r_local_t = nn.Parameter(self.Wo_r_local.data.t().contiguous(), requires_grad=True)
        self._wo_sharded = True
        self._wo_rank = rank

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
        """POST: A2A transpose (scatter seq, gather heads) → Wo row-split GEMM."""
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        local_N = hidden // sp
        local_hidden = self.local_hidden
        lm = lbs * llseq
        # o [lbs, lseq, local_nh, hd] → A2A → gathered [lm, hidden]
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        # Wo_r_local [local_N, dim] → y = gathered @ Wo_r^T = [lm, local_N]
        return torch.matmul(gathered, self.Wo_r_local.t()) + self.model.o.bias[self._wo_rank * local_N:(self._wo_rank + 1) * local_N]
