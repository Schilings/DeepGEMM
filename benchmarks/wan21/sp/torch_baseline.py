"""Torch-native SP baseline: torch.matmul + NCCL all_to_all (no fused kernels).

Used as the performance/precision reference for comparing against fused strategies.
Same math as the fused strategies, but communication is serial (no overlap).
"""

import torch
import torch.distributed as dist

from .base import UlyssesSPBase


class TorchUlyssesAttention(UlyssesSPBase):
    """Torch-native SP attention — always uses matmul + NCCL, regardless of use_fused_ops."""

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = False  # force torch path
        super().__init__(config, sp_config)

    def _create_symm_buffers(self):
        # No symm buffers needed — torch path uses NCCL directly
        pass

    def destroy_buffers(self):
        pass

    def _post_forward(self, o, lbs, lseq, llseq, grid_sizes, **kw):
        """POST: NCCL A2A + torch.matmul (standard Ulysses)."""
        lm = kw.get('lm', self.bs * llseq)
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.config.dim

        x_bhsd = o.transpose(1, 2)  # [bs, local_nh, seq, hd]
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        y = torch.matmul(gathered, self.Wo_t)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid_sizes, **kw):
        """POST backward: torch.matmul + NCCL A2A."""
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.config.dim

        grad_gathered = torch.matmul(grad_y, self.Wo)

        o = cache
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)

        grad_Wo = torch.matmul(grad_y.t(), gathered)

        send_bwd = grad_gathered.view(lbs, llseq, sp, self.local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=self.group)
        grad_attn = recv_bwd.permute(1, 3, 2, 4, 0).reshape(lbs, lseq, self.local_nh, hd)

        return grad_attn, grad_Wo

    def get_cache(self, o):
        return o
