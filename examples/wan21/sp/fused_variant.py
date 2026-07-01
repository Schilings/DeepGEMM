"""Fused Variant Ulysses: Q/K/V separate → A2A → attn → Wo GEMM+RS (column-split Wo).

Wo column-split (in Y=XW, W row-split): Wo_i [dim, local_hidden] per rank.
Attention output [full_m, local_hidden] @ Wo_i^T → [full_m, dim] partial → RS → [local_m, dim].

This uses reduce-scatter (not AllReduce) — matches our bf16_gemm_rs_nt operator.
The RS replaces the POST A2A (scatter seq) + full Wo GEMM in serial/fused_std.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from .serial import NCCLAllToAll


class ReduceScatter(torch.autograd.Function):
    """Reduce-scatter with autograd: forward=RS, backward=AllGather."""

    @staticmethod
    def forward(ctx, x, group):
        """x [full_m, dim] → RS → [local_m, dim] (each rank gets 1/sp of the sum)."""
        ctx.group = group
        sp = group.size()
        full_m, dim = x.shape
        local_m = full_m // sp
        out = torch.empty(local_m, dim, dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Backward = AllGather: [local_m, dim] → [full_m, dim]."""
        sp = ctx.group.size()
        full_list = [torch.empty_like(grad_out) for _ in range(sp)]
        dist.all_gather(full_list, grad_out, group=ctx.group)
        return torch.cat(full_list, dim=0), None


class FusedVariantUlysses(UlyssesBase):
    """Q/K/V separate + Wo column-split + GEMM+RS (Ulysses variant)."""

    def __init__(self, config, sp_config):
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Standard [Q_all, K_all, V_all] + Wo column-split (for GEMM+RS)."""
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        # Wo column-split: Wo_i = Wo[:, i*local_hidden:(i+1)*local_hidden] = [dim, local_hidden]
        # Y = attn [full_m, local_hidden] @ Wo_i^T [dim, local_hidden].T = [full_m, dim] (partial) → RS
        local_hidden = self.local_nh * self.head_dim
        rank = self.group.rank()
        Wo = self.model.o.weight  # [dim, dim] = [out, in]
        Wo_i = Wo[:, rank * local_hidden:(rank + 1) * local_hidden].contiguous()  # [dim, local_hidden]
        self.Wo_r_local = nn.Parameter(Wo_i.clone(), requires_grad=True)
        self.Wo_r_local_t = nn.Parameter(self.Wo_r_local.data.t().contiguous(), requires_grad=True)
        self._wo_sharded = True  # Wo grad is local (no FSDP2 sync)

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
        """POST: Wo column-split GEMM → RS → + bias.

        o [lbs, lseq, local_nh, hd] → flatten [full_m, local_hidden]
        → Y_i = attn @ Wo_i^T = [full_m, dim] (partial) → RS → [local_m, dim] → + bias
        """
        full_m = lbs * lseq  # full seq
        local_hidden = self.local_hidden
        # Flatten attention output
        attn_local = o.reshape(full_m, local_hidden).contiguous()
        # Wo column-split GEMM: Y_i = attn @ Wo_i^T = [full_m, local_hidden] @ [dim, local_hidden].T
        # = [full_m, local_hidden] @ [local_hidden, dim] = [full_m, dim] (PARTIAL SUM)
        y_partial = torch.matmul(attn_local, self.Wo_r_local.t())
        # Reduce-scatter: sum partial sums across ranks + scatter seq → [local_m, dim]
        y = ReduceScatter.apply(y_partial, self.group)
        # Add bias (after RS, each rank has local_m tokens)
        return y + self.model.o.bias
