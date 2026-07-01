"""Serial Ulysses baseline: matmul + NCCL A2A (no fused kernels, no overlap).

Autograd-based: A2A wrapped as autograd.Function (backward = A2A-inverse).
Forward builds the graph, backward is automatic.
"""

import torch
import torch.distributed as dist

from .base import UlyssesBase


class NCCLAllToAll(torch.autograd.Function):
    """NCCL all_to_all_single with autograd: backward = A2A-inverse (same permute, reversed)."""

    @staticmethod
    def forward(ctx, send, group, sp, local_nh, lnq, lbs, lseq, llseq, mode):
        """send [sp, lbs, seq_dim, features] → A2A → recv [sp, lbs, seq_dim, features].

        mode='pre': seq_dim=llseq (pre-attn: scatter heads, gather seq)
        mode='post': seq_dim=llseq (post-attn: scatter seq, gather heads)
        The A2A-inverse is the same all_to_all (it's its own inverse with matching permute).
        """
        ctx.group = group; ctx.sp = sp; ctx.local_nh = local_nh; ctx.lnq = lnq
        ctx.lbs = lbs; ctx.lseq = lseq; ctx.llseq = llseq; ctx.mode = mode
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def backward(ctx, grad_recv):
        """Backward = A2A-inverse = all_to_all again (A2A is self-inverse with matching permute)."""
        grad_recv = grad_recv.contiguous()
        grad_send = torch.empty_like(grad_recv)
        dist.all_to_all_single(grad_send, grad_recv, group=ctx.group)
        return grad_send, None, None, None, None, None, None, None, None


class SerialUlysses(UlyssesBase):
    """Baseline: everything serial with torch.matmul + NCCL.

    Forward is autograd-compatible: matmul + A2A(Function) + matmul.
    Backward is automatic via torch.autograd.backward().
    """

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = False
        super().__init__(config, sp_config)

    def _pre_forward(self, x_local, llseq):
        sp = self.sp_size
        lnq = self.local_nqkv
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        d = torch.matmul(x_local, self.Wqkv.t()).view(lbs, llseq, sp, lnq)
        send = d.permute(2, 0, 1, 3).contiguous()
        recv = NCCLAllToAll.apply(send, self.group, sp, self.local_nh, lnq, lbs, lseq, llseq, 'pre')
        return recv.permute(1, 2, 0, 3).reshape(lbs, lseq, lnq)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        lm = lbs * llseq
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = NCCLAllToAll.apply(send, self.group, sp, self.local_nh, hd, lbs, lseq, llseq, 'post')
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        return torch.matmul(gathered, self.Wo.t())
