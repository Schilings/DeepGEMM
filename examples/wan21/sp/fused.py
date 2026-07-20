"""Standard Ulysses with DeepGEMM fused POST communication operator.

PRE (QKV projection + QK RMSNorm + A2A) is identical to the serial baseline,
because QK RMSNorm must be applied *before* the A2A head-scatter and the fused
``bf16_gemm_a2a_transpose_nt`` kernel does not support inserting a norm between
GEMM and A2A.  Using the dedicated ``fused_qkv_norm_a2a`` kernel would change
the PRE implementation and is left as a future optimization.

POST (A2A-transpose + Wo GEMM) is replaced by DeepGEMM
``bf16_a2a_transpose_gemm_nt``, which fuses the communication and GEMM using
symmetric memory.  Wo is replicated, identical to the baseline data layout.

A single ``UnifiedSymmBuffer`` is shared with the variant's GEMM-RS / AG-GEMM
operators — all operators run serially (fwd/bwd not concurrent), so one
physical allocation is reused across PRE/POST and forward/backward.
"""

import torch
import torch.nn as nn

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from ..autograd_ops import fused_post_wo


class FusedUlysses(UlyssesBase):
    """Standard Ulysses with DeepGEMM fused POST operator.

    PRE is inherited unchanged from :class:`UlyssesBase`.  Only POST uses
    the DeepGEMM fused A2A-transpose + Wo GEMM.
    """

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = True
        sp_config.post_strategy = 'a2a_gemm'
        super().__init__(config, sp_config)
        self.sym_post = None
        self._owns_sym_post = False

    def _create_buffers(self):
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, self.cfg.dim,
            q_nheads=self.cfg.num_heads, kv_nheads=self.cfg.num_heads,
            head_dim=self.head_dim, out_dtype=torch.bfloat16)
        self._owns_sym_post = True

    def share_buffers_from(self, other):
        """Borrow the POST workspace shared serially by all attention layers."""
        self.sym_post = other.sym_post
        self._owns_sym_post = False

    def destroy_buffers(self):
        if self._owns_sym_post and self.sym_post is not None:
            self.sym_post.destroy()
        self.sym_post = None
        self._owns_sym_post = False

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: fused A2A-transpose + Wo GEMM.

        ``o`` is the FA4 output: [bs, seq, local_nh, hd] (BSHD layout).
        It is passed to ``fused_post_wo`` which copies it into sym_post.x
        (transposed to BHSD) and runs the fused A2A+GEMM.
        """
        local_m = lbs * llseq
        y = fused_post_wo(o, self.model.o.weight, self.sym_post, local_m, self.bs)
        return y + self.model.o.bias


# Compatibility alias for older imports.
FusedStandardUlysses = FusedUlysses
