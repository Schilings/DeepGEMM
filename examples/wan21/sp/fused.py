"""Standard Ulysses with DeepGEMM fused PRE and POST communication operators.

PRE uses ``bf16_fused_qkv_norm_a2a_nt`` (fused QKV projection + QK RMSNorm +
A2A-transpose) — a single kernel that computes the projection, applies
RMSNorm in a norm-deferred two-phase approach, and scatters to peers.

POST uses ``bf16_a2a_transpose_gemm_nt`` (fused A2A-transpose + Wo GEMM).

Wo is replicated across SP ranks, identical to the baseline data layout.
A single ``UnifiedSymmBuffer`` is shared between PRE and POST — all operators
run serially (fwd/bwd not concurrent), so one physical allocation is reused.
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
        """PRE: inherited from serial baseline.

        ``FusedPreQKVFunction`` (using ``bf16_fused_qkv_norm_a2a_nt``) has been
        verified correct in forward (Q/K/V rel=0.014 vs serial), but its
        backward (inverse A2A + RMSNorm backward + GEMM) has a bug causing
        large gradient errors.  PRE is temporarily falling back to serial
        until the backward is fixed.  See ``FusedPreQKVFunction`` in
        ``autograd_ops.py`` for the WIP implementation.
        """
        return super()._pre_forward(x_local, llseq)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: fused A2A-transpose + Wo GEMM.

        ``o`` is the FA4 output: [bs, seq, local_nh, hd] (BSHD layout).
        """
        local_m = lbs * llseq
        y = fused_post_wo(o, self.model.o.weight, self.sym_post, local_m, self.bs)
        return y + self.model.o.bias


# Compatibility alias for older imports.
FusedStandardUlysses = FusedUlysses
