"""POST-only Ulysses variant v2 + Wo deferred: same as v2 but also defers the
Wo weight gradient to the next layer's AG communication window.
"""

import torch
import torch.nn as nn

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from ..autograd_ops_v2 import (
    DeferredGradManager,
    deferred_linear,
    post_linear_v2_wo,
)


class FusedVariantV2WoUlysses(UlyssesBase):
    """Ulysses variant v2 + Wo deferred.

    Identical to FusedVariantV2Ulysses except POST backward also defers the
    Wo weight gradient (``grad_weight``) to the next layer's AG window.
    """

    def __init__(self, config, sp_config):
        sp_config.post_strategy = 'gemm_rs'
        super().__init__(config, sp_config)
        self.sym_post = None
        self._owns_sym_post = False
        self.grad_manager = None
        self._owns_grad_manager = False
        self._qkv_replaced = False

    def _build_weights(self):
        local_hidden = self.local_hidden
        rank = self.group.rank()
        full_weight = self.model.o.weight.detach()
        local_weight = full_weight[:, rank * local_hidden:(rank + 1) * local_hidden]
        self.Wo_r_local = nn.Parameter(local_weight.contiguous(), requires_grad=True)
        self.Wo_r_local._sp_sharded = True
        # Wo is SP-sharded — no cross-SP reduce, but still needs deferred grad
        self.model.o.register_parameter('weight', None)
        self._wo_sharded = True
        self._replace_qkv_with_deferred()

    def _replace_qkv_with_deferred(self):
        if self._qkv_replaced:
            return
        self._qkv_replaced = True
        for name in ('q', 'k', 'v'):
            linear = getattr(self.model, name)
            linear.weight._deferred_grad = True
            if linear.bias is not None:
                linear.bias._deferred_grad = True
            linear._v2_original_forward = linear.forward

            def _make_deferred_forward(lin):
                def _forward(x):
                    return deferred_linear(
                        x, lin.weight, lin.bias, self.grad_manager)
                return _forward

            linear.forward = _make_deferred_forward(linear)

    def _create_buffers(self):
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, self.cfg.dim, out_dtype=torch.bfloat16)
        self._owns_sym_post = True
        self.grad_manager = DeferredGradManager(self.group, self.sp_size)
        self._owns_grad_manager = True

    def share_buffers_from(self, other):
        self.sym_post = other.sym_post
        self._owns_sym_post = False
        if hasattr(other, 'grad_manager') and other.grad_manager is not None:
            self.grad_manager = other.grad_manager
            self._owns_grad_manager = False
            if self._qkv_replaced:
                for name in ('q', 'k', 'v'):
                    linear = getattr(self.model, name)

                    def _make_deferred_forward(lin):
                        def _forward(x):
                            return deferred_linear(
                                x, lin.weight, lin.bias, self.grad_manager)
                        return _forward

                    linear.forward = _make_deferred_forward(linear)

    def destroy_buffers(self):
        if self._owns_sym_post and self.sym_post is not None:
            self.sym_post.destroy()
        self.sym_post = None
        self._owns_sym_post = False
        self.grad_manager = None
        self._owns_grad_manager = False

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        full_m = lbs * lseq
        attn_local = o.reshape(full_m, self.local_hidden).contiguous()
        y = post_linear_v2_wo(
            attn_local, self.Wo_r_local, self.sym_post,
            self.local_m, self.grad_manager)
        return y + self.model.o.bias
