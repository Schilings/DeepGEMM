"""Fused Variant Ulysses: GEMM+A2A (PRE fwd) + GEMM+RS (POST fwd).

BWD uses DUAL fused ops:
  POST_bwd: AG+GEMM (bf16_ag_gemm_nt) — all-gather grad_y then GEMM with Wo_r_local
  PRE_bwd:  A2A+GEMM (serial A2A-inv + matmul) — inverse of GEMM+A2A forward

Weight grads computed serially.
"""

import torch
import torch.distributed as dist
from .base import UlyssesBase


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
        # POST forward: GEMM+RS (A=attn_local[local_m, local_hidden], B=Wo_r_local[local_N, local_hidden])
        self.sym_gemm_rs = get_symm_buffer_for_gemm_rs(self.group, local_m, local_N)
        # POST backward: AG+GEMM
        # A_local = grad_y [local_m, local_N], all-gather on token dim → [full_m, local_N]
        # B = Wo_r_local [local_hidden, local_N] (NT layout)
        # d = A_gathered @ B^T = [full_m, local_hidden]
        # hidden = local_N (the gather input feature width)
        self.sym_ag_gemm = get_symm_buffer_for_bf16_ag_gemm(self.group, local_m, local_N)
        # PRE backward: A2A+GEMM (dual of GEMM+A2A forward); QKV = 3*num_heads "heads"
        assert (3 * self.cfg.num_heads) % self.sp_size == 0, '3*num_heads must divide sp_size'
        self.sym_pre_bwd = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, self.bs, 3 * self.cfg.num_heads, self.seq, self.head_dim)
        # Wo row-split per rank
        rank = self.group.rank()
        Wo = self.model.o_proj.weight
        self.Wo_r_local = Wo[
            rank * local_N:(rank + 1) * local_N,
            rank * self.local_hidden:(rank + 1) * self.local_hidden].contiguous()
        self.Wo_r_local_t = self.Wo_r_local.t().contiguous()
        self._wo_sharded = True  # Wo is row-split → weight grad is local, no all-reduce needed

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'): self.sym_pre.destroy()
        if hasattr(self, 'sym_gemm_rs'):
            self.sym_gemm_rs.handle = None; self.sym_gemm_rs.buffer = None; self.sym_gemm_rs.group = None
        if hasattr(self, 'sym_ag_gemm'):
            self.sym_ag_gemm.handle = None; self.sym_ag_gemm.buffer = None; self.sym_ag_gemm.group = None
        if hasattr(self, 'sym_pre_bwd'): self.sym_pre_bwd.destroy()

    def _pre_forward(self, x_local, llseq):
        from deep_gemm import bf16_gemm_a2a_transpose_nt
        qkv = bf16_gemm_a2a_transpose_nt(x_local, self.Wqkv, self.sym_pre, llseq)
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        if self.layout == 'THD': qkv = qkv[:lbs, :lseq, :]
        return qkv

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """GEMM+RS: attn_local @ Wo_r_local^T → reduce-scatter → y[local_m, local_N]."""
        from deep_gemm.gemm_rs import bf16_gemm_rs_nt
        local_N = self.cfg.dim // self.sp_size
        lm_actual = self.bs * (self.seq // self.sp_size)
        attn_local = o.reshape(self.bs * self.seq, self.local_hidden).contiguous()
        y = torch.empty((lm_actual, local_N), dtype=torch.bfloat16, device=o.device)
        bf16_gemm_rs_nt(y, attn_local, self.Wo_r_local, self.sym_gemm_rs, lm_actual)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid, **kw):
        """POST BWD: AG+GEMM (bf16_ag_gemm_nt).

        Forward: attn_local @ Wo_r_local^T → RS → y[local_m, local_N]
        Backward: grad_y[local_m, local_N] → AG → grad_y_full[bs*seq, local_N]
                  → GEMM(grad_y_full, Wo_r_local^T) → grad_attn[bs*seq, local_hidden]

        bf16_ag_gemm_nt does: d = all_gather(A_local) @ B^T
          A_local = grad_y [local_m, local_N]
          B = Wo_r_local [local_hidden, local_N]  (NT layout, so B^T = Wo_r_local^T)
          d = grad_attn [bs*seq, local_hidden]
        """
        from deep_gemm.ag_gemm import bf16_ag_gemm_nt
        sp = self.sp_size
        local_N = self.cfg.dim // sp
        local_hidden = self.local_hidden
        o = cache  # [bs, seq, local_nh, hd]
        full_m = self.bs * self.seq

        # Fused activation grad: AG+GEMM
        # Copy grad_y into sym buffer's x [local_m, local_N]
        self.sym_ag_gemm.x[:lm, :local_N].copy_(grad_y)
        grad_attn_flat = torch.empty((full_m, local_hidden), dtype=torch.bfloat16, device=grad_y.device)
        bf16_ag_gemm_nt(grad_attn_flat, self.Wo_r_local, self.sym_ag_gemm, lm)
        # grad_attn_flat = [full_m, local_hidden] → [bs, seq, local_nh, hd]
        grad_attn = grad_attn_flat.reshape(self.bs, self.seq, self.local_nh, self.head_dim)

        # Weight grad (serial): grad_Wo_r_local = grad_y_full^T @ attn_local
        # Need grad_y_full: recompute from all-gather
        gy_list = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(gy_list, grad_y, group=self.group)
        grad_y_full = torch.cat(gy_list, dim=0)  # [full_m, local_N]
        attn_local = o.reshape(full_m, local_hidden).contiguous()
        grad_Wo_r_local = torch.matmul(grad_y_full.t(), attn_local)

        return grad_attn, grad_Wo_r_local

    def _pre_backward(self, grad_qkv, lbs, lseq, llseq, lm, x_local, **kw):
        """PRE BWD: A2A+GEMM fused (dual of GEMM+A2A forward).

        Forward was: x_local → GEMM(x, Wqkv^T) → A2A-scatter(heads) → qkv
        Backward:    grad_qkv → A2A-gather(heads) → grad_local[lm, n_qkv]
                     → GEMM(grad_local @ Wqkv_t^T) → grad_X
        Reuses the POST-forward A2A+GEMM kernel (bf16_a2a_transpose_gemm_nt, M0):
        A2A-inv gathers QKV heads (3*local_nh per rank) into grad_local, then the
        NT GEMM with Wqkv_t (=[hidden, n_qkv]) yields grad_X = grad_local @ Wqkv.
        Weight grad grad_Wqkv = grad_local^T @ x_local (serial, reuses gathered).
        """
        from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt
        n_qkv = self.cfg.n_qkv
        local_nh_qkv = 3 * self.local_nh
        # grad_qkv [lbs, lseq, local_nqkv] → BSHD [lbs, lseq, 3*local_nh, hd] → BHSD [lbs, 3*local_nh, lseq, hd]
        gqkv_bhsd = grad_qkv.view(lbs, lseq, local_nh_qkv, self.head_dim).transpose(1, 2).contiguous()
        self.sym_pre_bwd.x.copy_(gqkv_bhsd)
        grad_X = torch.empty((lm, self.cfg.dim), dtype=torch.bfloat16, device=grad_qkv.device)
        bf16_a2a_transpose_gemm_nt(grad_X, self.Wqkv_t, self.sym_pre_bwd)
        # Weight grad (serial): grad_Wqkv = grad_local^T @ x_local (grad_local = gathered)
        grad_local = self.sym_pre_bwd.gathered[:lm, :n_qkv]
        grad_Wqkv = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_Wqkv
