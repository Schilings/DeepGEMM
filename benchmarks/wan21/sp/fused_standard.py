"""Fused Standard Ulysses: GEMM+A2A (PRE fwd) + A2A+GEMM (POST fwd).

Autograd-based: fused ops wrapped as torch.autograd.Function.
  PRE fwd: GemmA2ATransposeFunction (backward = A2A+GEMM dual)
  POST fwd: A2ATransposeGemmFunction (backward = GEMM+A2A dual)
Backward is automatic via torch.autograd.backward().
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from ..autograd_ops import GemmA2ATransposeFunction, A2ATransposeGemmFunction


class FusedStandardUlysses(UlyssesBase):
    def _create_buffers(self):
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.a2a_transpose_gemm import get_symm_buffer_for_a2a_transpose_gemm
        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, self.bs, self.seq, self.cfg.n_qkv)
        self.sym_post = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, self.bs, self.cfg.num_heads, self.seq, self.head_dim)
        # PRE backward (A2A+GEMM dual) reuses the same sym buffer type as POST forward
        assert (3 * self.cfg.num_heads) % self.sp_size == 0, '3*num_heads must divide sp_size'
        self.sym_pre_bwd = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, self.bs, 3 * self.cfg.num_heads, self.seq, self.head_dim)

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'): self.sym_pre.destroy()
        if hasattr(self, 'sym_post'): self.sym_post.destroy()
        if hasattr(self, 'sym_pre_bwd'): self.sym_pre_bwd.destroy()

    def _pre_forward(self, x_local, llseq):
        """PRE: GEMM+A2A via autograd.Function (backward = A2A+GEMM dual)."""
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        layout_info = {
            'lbs': lbs, 'lseq': lseq, 'llseq': llseq, 'lm': lbs * llseq,
            'nheads': self.cfg.num_heads, 'hidden': self.cfg.dim,
            'head_dim': self.head_dim, 'thd': self.layout == 'THD',
        }
        qkv = GemmA2ATransposeFunction.apply(
            x_local, self.Wqkv, self.sym_pre, self.sym_pre_bwd, llseq, self.cfg.n_qkv,
            self.sp_size, self.group, layout_info)
        if self.layout == 'THD':
            qkv = qkv[:lbs, :lseq, :]
        return qkv

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: A2A+GEMM via autograd.Function (backward = GEMM+A2A dual)."""
        layout_info = {
            'lbs': lbs, 'lseq': lseq, 'llseq': llseq, 'local_m': lbs * llseq,
            'local_nh': self.local_nh, 'hd': self.head_dim, 'hidden': self.cfg.dim,
            'sp_size': self.sp_size, 'group': self.group,
        }
        return A2ATransposeGemmFunction.apply(o, self.Wo, self.sym_post, layout_info)
