"""Standard Ulysses with DeepGEMM fused PRE and POST communication operators.

PRE uses ``bf16_fused_qkv_norm_a2a_nt`` (fused QKV projection + QK RMSNorm +
A2A-transpose) — a single kernel that computes the projection, applies
RMSNorm in a norm-deferred two-phase approach, and scatters to peers.

POST uses ``bf16_a2a_transpose_gemm_nt`` (fused A2A-transpose + Wo GEMM).

Wo is replicated across SP ranks, identical to the baseline data layout.
A single ``UnifiedSymmBuffer`` is shared between PRE and POST — all operators
run serially (fwd/bwd not concurrent), so one physical allocation is reused.

Backward correctness verified: grad_X rel=0.019, grad_Wq rel=0.018,
grad_Wk rel=0.032, grad_Wv rel=0.015, grad_Nq rel=0.031 (BF16 normal range).
Forward rel=0.234 is BF16 FA4 amplification of systematic V diff (not a bug).
"""

import torch
import torch.nn as nn

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from ..autograd_ops import fused_pre_qkv, fused_post_wo


def _build_wqkv(Wq, Wk, Wv):
    """Build fused QKV weight in [Q | K | V] segment layout.

    Returns ``[q_dim + 2*kv_dim, hidden]`` = ``[3*hidden, hidden]`` for MHA,
    matching the layout expected by ``bf16_fused_qkv_norm_a2a_nt``:
    ``b = [Wq(q_dim, K); Wk(kv_dim, K); Wv(kv_dim, K)]``.
    """
    return torch.cat([Wq, Wk, Wv], dim=0).contiguous()


class FusedUlysses(UlyssesBase):
    """Standard Ulysses with DeepGEMM fused PRE and POST operators."""

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = True
        sp_config.post_strategy = 'a2a_gemm'
        super().__init__(config, sp_config)
        self.sym_post = None
        self._owns_sym_post = False
        self._wqkv = None  # rebuilt each forward from current q/k/v weights

    def _create_buffers(self):
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, self.cfg.dim,
            q_nheads=self.cfg.num_heads, kv_nheads=self.cfg.num_heads,
            head_dim=self.head_dim, out_dtype=torch.bfloat16)
        self._owns_sym_post = True

    def share_buffers_from(self, other):
        """Borrow the workspace shared serially by all attention layers."""
        self.sym_post = other.sym_post
        self._owns_sym_post = False

    def destroy_buffers(self):
        if self._owns_sym_post and self.sym_post is not None:
            self.sym_post.destroy()
        self.sym_post = None
        self._owns_sym_post = False

    def _pre_forward(self, x_local, llseq):
        """Fused QKV projection + QK RMSNorm + A2A-transpose."""
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        hd = self.head_dim
        local_seq = llseq

        # Rebuild Wqkv from current q/k/v weights (handles parameter updates)
        self._wqkv = _build_wqkv(
            self.model.q.weight, self.model.k.weight, self.model.v.weight,
        )

        # Fused GEMM + Norm + A2A: x_local @ Wqkv.T → norm(Q/K) → scatter → qkv
        # The Function internally reorders seq from [sp, local_seq] to
        # [local_seq, sp] to match the serial baseline layout.
        qkv = fused_pre_qkv(
            x_local, self._wqkv,
            self.model.norm_q.weight if hasattr(self.model.norm_q, 'weight') else None,
            self.model.norm_k.weight if hasattr(self.model.norm_k, 'weight') else None,
            self.sym_post, local_seq, self.bs, eps=self.cfg.eps,
        )
        return qkv.view(lbs, lseq, -1)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: fused A2A-transpose + Wo GEMM.

        ``o`` is the FA4 output: [bs, seq, local_nh, hd] (BSHD layout).
        """
        local_m = lbs * llseq
        y = fused_post_wo(o, self.model.o.weight, self.sym_post, local_m, self.bs)
        return y + self.model.o.bias


# Compatibility alias for older imports.
FusedStandardUlysses = FusedUlysses
