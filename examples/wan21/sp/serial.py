"""Serial Ulysses baseline: matmul + NCCL A2A (no fused kernels, no overlap).

Autograd-based: A2A wrapped as autograd.Function (backward = A2A-inverse).
Forward builds the graph, backward is automatic.

Key: serial uses STANDARD [Q_all, K_all, V_all] weight order (not rank-major),
so norm_q/norm_k can be applied on full dim BEFORE A2A (when Q/K are [B,S,dim]).
"""

import torch
import torch.distributed as dist

from .base import UlyssesBase


class NCCLAllToAll(torch.autograd.Function):
    """NCCL all_to_all_single with autograd: backward = A2A-inverse."""

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


class SerialUlysses(UlyssesBase):
    """Baseline: everything serial with torch.matmul + NCCL."""

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = False
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Serial uses standard [Q_all, K_all, V_all] order (not rank-major)."""
        import torch.nn as nn
        Wq = self.model.q.weight  # [dim, dim]
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        # Standard order: [Q_all_heads, K_all_heads, V_all_heads]
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        self.Wo = self.model.o.weight
        self.Wo_t = self.Wo.t().contiguous()

    def _pre_forward(self, x_local, llseq):
        """PRE: GEMM → split Q/K/V → norm_q/norm_k → A2A scatter heads → gather seq."""
        dim = self.cfg.dim
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        hd = self.head_dim

        # GEMM: x @ Wqkv.t() + bias → [local_m, 3*dim] (standard Q,K,V order)
        bias = torch.cat([self.model.q.bias, self.model.k.bias, self.model.v.bias])
        d = torch.matmul(x_local, self.Wqkv.t()) + bias
        # Split Q/K/V (each [local_m, dim]), apply norm BEFORE A2A (full dim)
        q = self.model.norm_q(d[:, :dim]).view(lbs, llseq, sp, self.local_nh, hd)
        k = self.model.norm_k(d[:, dim:2*dim]).view(lbs, llseq, sp, self.local_nh, hd)
        v = d[:, 2*dim:].view(lbs, llseq, sp, self.local_nh, hd)
        # A2A: scatter heads (per-rank head group), gather seq → [lbs, lseq, local_nh, hd]
        # send: [sp, lbs, llseq, local_nh, hd] → A2A → recv: [sp, lbs, llseq, local_nh, hd]
        # recv[r] = all heads from rank r's seq shard for this rank's head group
        def a2a_scatter(t):
            send = t.permute(2, 0, 1, 3, 4).contiguous()  # [sp, lbs, llseq, local_nh, hd]
            recv = NCCLAllToAll.apply(send, self.group)
            return recv.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)  # [lbs, lseq, local_nh, hd]
        q = a2a_scatter(q)
        k = a2a_scatter(k)
        v = a2a_scatter(v)
        # Return [lbs, lseq, 3*local_n] (local_n = local_nh * hd)
        return torch.cat([q, k, v], dim=2).reshape(lbs, lseq, -1)  # [lbs, lseq, 3*local_n]

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: A2A transpose (scatter seq, gather heads) → Wo GEMM."""
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        lm = lbs * llseq
        # o [lbs, lseq, local_nh, hd] → A2A → gathered [lm, hidden]
        x_bhsd = o.transpose(1, 2)  # [lbs, local_nh, lseq, hd]
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        return torch.matmul(gathered, self.Wo.t()) + self.model.o.bias
