"""Fused Variant Ulysses: GEMM+A2A (PRE) + GEMM+RS (POST).
BWD: PRE uses A2A+GEMM (serial), POST uses AG+GEMM (fused).
"""

import torch
import torch.distributed as dist
from .base import UlyssesBase
from ..model import build_wqkv_rankmajor


class FusedVariantUlysses(UlyssesBase):
    def _create_buffers(self):
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.gemm_rs import get_symm_buffer_for_gemm_rs
        n_qkv = self.cfg.n_qkv
        local_N = self.cfg.dim // self.sp_size
        local_m = self.local_m
        # PRE buffer
        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, self.bs, self.seq, n_qkv)
        # POST forward: GEMM+RS
        self.sym_gemm_rs = get_symm_buffer_for_gemm_rs(self.group, local_m, local_N)
        # POST backward: AG+GEMM (created lazily when needed)
        # Wo row-split per rank
        rank = self.group.rank()
        Wo = self.model.o_proj.weight
        self.Wo_r_local = Wo[
            rank * local_N:(rank + 1) * local_N,
            rank * self.local_hidden:(rank + 1) * self.local_hidden].contiguous()
        self.Wo_r_local_t = self.Wo_r_local.t().contiguous()

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'): self.sym_pre.destroy()
        if hasattr(self, 'sym_gemm_rs'):
            self.sym_gemm_rs.handle = None; self.sym_gemm_rs.buffer = None; self.sym_gemm_rs.group = None

    def _pre_forward(self, x_local, llseq):
        from deep_gemm import bf16_gemm_a2a_transpose_nt
        qkv = bf16_gemm_a2a_transpose_nt(x_local, self.Wqkv, self.sym_pre, llseq)
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        if self.layout == 'THD': qkv = qkv[:lbs, :lseq, :]
        return qkv

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """GEMM+RS: attn_local @ Wo_r_local^T → reduce-scatter → y[lm, local_N]."""
        from deep_gemm.gemm_rs import bf16_gemm_rs_nt
        lm = lbs * llseq if lbs != 1 else self.local_m
        local_N = self.cfg.dim // self.sp_size
        lm_actual = self.bs * (self.seq // self.sp_size) if self.layout == 'THD' else lbs * llseq
        attn_local = o.reshape(self.bs * self.seq, self.local_hidden).contiguous()
        y = torch.empty((lm_actual, local_N), dtype=torch.bfloat16, device=o.device)
        bf16_gemm_rs_nt(y, attn_local, self.Wo_r_local, self.sym_gemm_rs, lm_actual)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid, **kw):
        """POST BWD: AG+GEMM (fused).
        grad_y[lm, local_N] → all-gather → grad_y_full[bs*seq, local_N]
        → GEMM(grad_y_full, Wo_r_local) → grad_attn_local[bs*seq, local_hidden]
        Weight grad grad_Wo_r_local = grad_y_full^T @ attn_local (serial).
        """
        sp = self.sp_size
        local_N = self.cfg.dim // sp
        local_hidden = self.local_hidden
        o = cache
        # All-gather grad_y along seq → grad_y_seq[bs*seq, local_N]
        gy_list = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(gy_list, grad_y, group=self.group)
        grad_y_full = torch.cat(gy_list, dim=0)  # [bs*seq, local_N]
        # grad_attn_local = grad_y_full @ Wo_r_local → [bs*seq, local_hidden]
        attn_local = o.reshape(self.bs * self.seq, local_hidden).contiguous()
        grad_attn_local = torch.matmul(grad_y_full, self.Wo_r_local)
        grad_Wo_r_local = torch.matmul(grad_y_full.t(), attn_local)
        # grad_attn → [bs, seq, local_nh, hd]
        grad_attn = grad_attn_local.reshape(self.bs, self.seq, self.local_nh, self.head_dim)
        return grad_attn, grad_Wo_r_local

    def _pre_backward(self, grad_qkv, lbs, lseq, llseq, lm, x_local, **kw):
        """PRE BWD: A2A-inverse (serial) + GEMM."""
        sp = self.sp_size; lnq = self.local_nqkv; n_qkv = self.cfg.n_qkv
        send = grad_qkv.view(lbs, llseq, sp, lnq).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        grad_local = recv.permute(1, 2, 0, 3).reshape(lm, n_qkv)
        grad_X = torch.matmul(grad_local, self.Wqkv)
        grad_Wqkv = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_Wqkv
