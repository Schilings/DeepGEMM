"""Serial Ulysses baseline: matmul + NCCL A2A (no fused kernels, no overlap)."""

import torch
import torch.distributed as dist

from .base import UlyssesBase


class SerialUlysses(UlyssesBase):
    """Baseline: everything serial with torch.matmul + NCCL."""

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = False
        super().__init__(config, sp_config)

    def _pre_forward(self, x_local, llseq):
        sp = self.sp_size
        lnq = self.local_nqkv
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        d = torch.matmul(x_local, self.Wqkv_t).view(lbs, llseq, sp, lnq)
        send = d.permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.permute(1, 2, 0, 3).reshape(lbs, lseq, lnq)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        lm = lbs * llseq
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        return torch.matmul(gathered, self.Wo_t)

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid, **kw):
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        o = cache
        grad_gathered = torch.matmul(grad_y, self.Wo)
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        grad_Wo = torch.matmul(grad_y.t(), gathered)
        send_bwd = grad_gathered.view(lbs, llseq, sp, self.local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=self.group)
        grad_attn = recv_bwd.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)
        return grad_attn, grad_Wo

    def _pre_backward(self, grad_qkv, lbs, lseq, llseq, lm, x_local, **kw):
        sp = self.sp_size
        lnq = self.local_nqkv
        n_qkv = self.cfg.n_qkv
        send = grad_qkv.view(lbs, llseq, sp, lnq).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        grad_local = recv.permute(1, 2, 0, 3).reshape(lm, n_qkv)
        grad_X = torch.matmul(grad_local, self.Wqkv)
        grad_Wqkv = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_Wqkv
