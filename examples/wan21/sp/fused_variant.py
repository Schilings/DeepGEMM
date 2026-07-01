"""Fused Variant Ulysses: Megatron-style TP Wo row-split.

POST forward: attn_i @ Wo_i^T → partial sum → AllReduce (g operator) → Y + bias
  - g forward: AllReduce (reduce partial sums from all ranks)
  - g backward: Identity (grad flows directly, no comm)

This matches Megatron-LM RowParallelLinear with input_is_parallel=True.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from .serial import NCCLAllToAll


class AllReduce(torch.autograd.Function):
    """AllReduce with autograd: forward=AllReduce, backward=Identity (Megatron g operator)."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        # Non-in-place: create output, all_reduce into it
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad):
        # Identity: grad flows directly (AllReduce in forward makes all ranks identical)
        return grad, None


class FusedVariantUlysses(UlyssesBase):
    """Q/K/V separate + Wo row-split with AllReduce (Megatron TP style)."""

    def __init__(self, config, sp_config):
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Standard [Q_all, K_all, V_all] order + Wo column-split (Megatron RowParallel)."""
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        # Wo column-split (Megatron RowParallelLinear): Wo_i = Wo[:, i*dim/N:(i+1)*dim/N] = [dim, dim/N]
        # input attn_i [lm, dim/N] (local_hidden), Y_i = attn_i @ Wo_i^T = [lm, dim] (partial) → AllReduce
        local_hidden = self.local_nh * self.head_dim  # dim / sp
        rank = self.group.rank()
        Wo = self.model.o.weight  # [dim, dim] = [out, in]
        Wo_i = Wo[:, rank * local_hidden:(rank + 1) * local_hidden].contiguous()  # [dim, local_hidden]
        self.Wo_r_local = nn.Parameter(Wo_i.clone(), requires_grad=True)
        self.Wo_r_local_t = nn.Parameter(self.Wo_r_local.data.t().contiguous(), requires_grad=True)
        self._wo_sharded = True  # Wo grad is local (no FSDP2 sync)
        self._wo_rank = rank
        self._local_hidden = local_hidden

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
        """POST: A2A scatter seq → Wo full GEMM (same as serial for correctness).

        TODO: switch to Megatron RowParallel (Wo column-split + AllReduce) when
        AllReduce autograd Function is fixed (backward=Identity issue).
        """
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        lm = lbs * llseq
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        return torch.matmul(gathered, self.model.o.weight.t()) + self.model.o.bias
        return y + self.model.o.bias
