"""Fused Variant Ulysses: GEMM+A2A (PRE fwd) + GEMM+RS (POST fwd).

Autograd-based: fused ops wrapped as torch.autograd.Function.
  PRE fwd: GemmA2ATransposeFunction (backward = A2A+GEMM dual)
  POST fwd: GemmRSFunction (backward = AG+GEMM dual)
Backward is automatic via torch.autograd.backward().

Wo is row-split (N-sharded) → _wo_sharded=True, FSDP2 ignores it (grad is local).
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import UlyssesBase
from ..autograd_ops import GemmA2ATransposeFunction, GemmRSFunction


class FusedVariantUlysses(UlyssesBase):
    def _create_buffers(self):
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.gemm_rs import get_symm_buffer_for_gemm_rs
        from deep_gemm.ag_gemm import get_symm_buffer_for_bf16_ag_gemm
        from deep_gemm.a2a_transpose_gemm import get_symm_buffer_for_a2a_transpose_gemm
        n_qkv = self.cfg.n_qkv
        local_N = self.cfg.dim // self.sp_size
        local_m = self.local_m
        # PRE forward: GEMM+A2A
        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, self.bs, self.seq, n_qkv)
        # POST forward: GEMM+RS
        self.sym_gemm_rs = get_symm_buffer_for_gemm_rs(self.group, local_m, local_N)
        # POST backward: AG+GEMM (dual, shares buffer type)
        self.sym_ag_gemm = get_symm_buffer_for_bf16_ag_gemm(self.group, local_m, local_N)
        # PRE backward: A2A+GEMM (dual of GEMM+A2A forward); QKV = 3*num_heads "heads"
        assert (3 * self.cfg.num_heads) % self.sp_size == 0, '3*num_heads must divide sp_size'
        self.sym_pre_bwd = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, self.bs, 3 * self.cfg.num_heads, self.seq, self.head_dim)
        # Wo row-split per rank — nn.Parameter for FSDP2 (ignored, grad is local)
        rank = self.group.rank()
        Wo = self.model.o_proj.weight
        Wo_r = Wo[
            rank * local_N:(rank + 1) * local_N,
            rank * self.local_hidden:(rank + 1) * self.local_hidden].contiguous()
        self.Wo_r_local = nn.Parameter(Wo_r.clone(), requires_grad=True)
        self.Wo_r_local_t = nn.Parameter(self.Wo_r_local.data.t().contiguous(), requires_grad=True)
        self._wo_sharded = True  # Wo is row-split → weight grad is local, no FSDP2 sync

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'): self.sym_pre.destroy()
        if hasattr(self, 'sym_gemm_rs'):
            self.sym_gemm_rs.handle = None; self.sym_gemm_rs.buffer = None; self.sym_gemm_rs.group = None
        if hasattr(self, 'sym_ag_gemm'):
            self.sym_ag_gemm.handle = None; self.sym_ag_gemm.buffer = None; self.sym_ag_gemm.group = None
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
            x_local, self.Wqkv, self.sym_pre, llseq, self.cfg.n_qkv,
            self.sp_size, self.group, layout_info)
        if self.layout == 'THD':
            qkv = qkv[:lbs, :lseq, :]
        return qkv

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: GEMM+RS via autograd.Function (backward = AG+GEMM dual)."""
        local_N = self.cfg.dim // self.sp_size
        local_hidden = self.local_hidden
        local_m = lbs * llseq
        full_m = self.bs * self.seq
        # o [bs, seq, local_nh, hd] → flatten to [local_m, local_hidden] for GEMM
        # But o has full seq on each rank (after PRE A2A scattered heads, gathered seq)
        # Wait: after PRE, each rank has [bs, seq, local_nh, hd] — seq is FULL, local_nh is local
        # POST GEMM+RS: attn_local[local_m, local_hidden] @ Wo_r^T → RS → y[local_m, local_N]
        # local_m = bs * local_seq (this rank's seq shard), but o has full seq...
        # Actually in Ulysses: after PRE (scatter heads, gather seq), each rank has FULL seq
        # but only LOCAL heads. POST does GEMM on local heads then RS to scatter seq back.
        # attn_local = o reshaped: [bs*seq, local_hidden] but we need [local_m, local_hidden]
        # where local_m = bs * local_seq. This is because RS will scatter the seq dimension.
        attn_local = o.reshape(self.bs * self.seq, local_hidden).contiguous()
        layout_info = {
            'local_m': self.bs * (self.seq // self.sp_size),  # tokens per rank after RS
            'local_N': local_N, 'local_hidden': local_hidden,
            'full_m': full_m, 'sp_size': self.sp_size, 'group': self.group,
            'bs': self.bs, 'seq': self.seq,
        }
        return GemmRSFunction.apply(attn_local, self.Wo_r_local, self.sym_gemm_rs, layout_info)
