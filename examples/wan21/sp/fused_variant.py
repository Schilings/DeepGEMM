"""POST-only Ulysses variant using fused GEMM+RS / AG+GEMM.

PRE projection, Q/K normalization, synchronous A2A, RoPE and FA4 are inherited
unchanged from :class:`UlyssesBase`.  The only experimental variable
is POST attention:

  baseline: A2A(heads -> sequence) + replicated Wo linear
  variant:  column-sharded Wo linear + ReduceScatter

The fused POST backward is the exact dual: AllGather + GEMM for the activation
gradient, while the local Wo shard gradient is computed by torch autograd code.
"""

import torch
import torch.nn as nn

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from ..autograd_ops import fused_post_linear


class FusedVariantUlysses(UlyssesBase):
    """Ulysses with only its POST-attention stage replaced by GEMM+RS."""

    def __init__(self, config, sp_config):
        sp_config.post_strategy = 'gemm_rs'
        super().__init__(config, sp_config)
        self.sym_post = None
        self._owns_sym_post = False

    def _build_weights(self):
        """Keep only this rank's input-column shard of Wo.

        ``nn.Linear.weight`` is [out_features, in_features].  Attention heads are
        sharded along ``in_features``, so each rank owns [dim, dim / SP].  The
        original full ``model.o.weight`` is unregistered after slicing; retaining
        it would invalidate the memory ablation.
        """
        local_hidden = self.local_hidden
        rank = self.group.rank()
        full_weight = self.model.o.weight.detach()
        local_weight = full_weight[:, rank * local_hidden:(rank + 1) * local_hidden]
        self.Wo_r_local = nn.Parameter(local_weight.contiguous(), requires_grad=True)
        self.model.o.register_parameter('weight', None)
        self._wo_sharded = True

    def _create_buffers(self):
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, self.cfg.dim, out_dtype=torch.bfloat16)
        self._owns_sym_post = True

    def share_buffers_from(self, other):
        """Borrow one fixed workspace shared serially by all attention layers."""
        self.sym_post = other.sym_post
        self._owns_sym_post = False

    def destroy_buffers(self):
        if self._owns_sym_post and self.sym_post is not None:
            self.sym_post.destroy()
        self.sym_post = None
        self._owns_sym_post = False

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: local Wo shard GEMM fused with ReduceScatter, then shared bias."""
        full_m = lbs * lseq
        attn_local = o.reshape(full_m, self.local_hidden).contiguous()
        y = fused_post_linear(attn_local, self.Wo_r_local, self.sym_post, self.local_m)
        return y + self.model.o.bias
