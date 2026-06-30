"""Fused Standard Ulysses: GEMM+A2A (PRE) + A2A+GEMM (POST)."""

import torch
import torch.distributed as dist
from .base import UlyssesBase


class FusedStandardUlysses(UlyssesBase):
    def _create_buffers(self):
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.a2a_transpose_gemm import get_symm_buffer_for_a2a_transpose_gemm
        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, self.bs, self.seq, self.cfg.n_qkv)
        self.sym_post = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, self.bs, self.cfg.num_heads, self.seq, self.head_dim)

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'): self.sym_pre.destroy()
        if hasattr(self, 'sym_post'): self.sym_post.destroy()

    def _pre_forward(self, x_local, llseq):
        from deep_gemm import bf16_gemm_a2a_transpose_nt
        qkv = bf16_gemm_a2a_transpose_nt(x_local, self.Wqkv, self.sym_pre, llseq)
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        if self.layout == 'THD': qkv = qkv[:lbs, :lseq, :]
        return qkv

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt_fused
        lm = lbs * llseq
        self.sym_post.x.copy_(o.transpose(1, 2).contiguous())
        y = torch.empty((lm, self.cfg.dim), dtype=torch.bfloat16, device=o.device)
        bf16_a2a_transpose_gemm_nt_fused(y, self.Wo, self.sym_post)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid, **kw):
        sp = self.sp_size; hd = self.head_dim; hidden = self.cfg.dim
        o = cache
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        grad_Wo = torch.matmul(grad_y.t(), gathered)
        grad_gathered = torch.matmul(grad_y, self.Wo)
        send_bwd = grad_gathered.view(lbs, llseq, sp, self.local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=self.group)
        grad_attn = recv_bwd.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)
        return grad_attn, grad_Wo

    def _pre_backward(self, grad_qkv, lbs, lseq, llseq, lm, x_local, **kw):
        sp = self.sp_size; lnq = self.local_nqkv; n_qkv = self.cfg.n_qkv
        send = grad_qkv.view(lbs, llseq, sp, lnq).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        grad_local = recv.permute(1, 2, 0, 3).reshape(lm, n_qkv)
        grad_X = torch.matmul(grad_local, self.Wqkv)
        grad_Wqkv = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_Wqkv
