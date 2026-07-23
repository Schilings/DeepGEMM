"""POST-only Ulysses variant v2: native AG+GEMM backward with deferred
QKV weight-grad overlap.

Differences from v1 (``variant.py``):

* **Forward** — identical: GEMM+ReduceScatter (DeepGEMM fused op), Wo
  column-sharded.
* **POST backward** — native NCCL ``all_gather`` + native GEMM (same as the
  serial baseline) instead of the fused AG+GEMM operator.
* **QKV weight gradients** — deferred via ``DeferredLinearFunction`` and
  overlapped with the next layer's AG communication.

Everything else (PRE, RoPE, FA4, cross-attn, FFN, modulation) is unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from ..autograd_ops_v2 import (
    DeferredGradManager,
    deferred_linear,
    post_linear_v2,
)


class FusedVariantV2Ulysses(UlyssesBase):
    """Ulysses variant v2: GEMM+RS forward, native AG+GEMM backward with
    deferred QKV weight-grad overlap."""

    def __init__(self, config, sp_config):
        sp_config.post_strategy = 'gemm_rs'
        super().__init__(config, sp_config)
        self.sym_post = None
        self._owns_sym_post = False
        self.grad_manager = None
        self._owns_grad_manager = False
        self._qkv_replaced = False

    # ------------------------------------------------------------------
    # Weight layout — identical to v1 (Wo column-sharded)
    # ------------------------------------------------------------------

    def _build_weights(self):
        """Keep only this rank's input-column shard of Wo (same as v1)."""
        local_hidden = self.local_hidden
        rank = self.group.rank()
        full_weight = self.model.o.weight.detach()
        local_weight = full_weight[:, rank * local_hidden:(rank + 1) * local_hidden]
        self.Wo_r_local = nn.Parameter(local_weight.contiguous(), requires_grad=True)
        self.Wo_r_local._sp_sharded = True
        self.model.o.register_parameter('weight', None)
        self._wo_sharded = True

        # Replace Q/K/V nn.Linear internals with deferred-linear wrappers.
        # The weight/bias tensors are reused (not copied), so checkpoint
        # loading and weight copying still work before setup_shape.
        self._replace_qkv_with_deferred()

    def _replace_qkv_with_deferred(self):
        """Wrap Q/K/V linear forward to defer weight gradients.

        Called once during ``_build_weights`` (which runs inside
        ``setup_shape``).  The underlying ``nn.Linear.weight`` / ``.bias``
        Parameters are retained — only the ``forward`` method is redirected
        through ``deferred_linear``.

        A ``_deferred_grad`` flag is set on each QKV weight so that DDP can
        exclude it from automatic reduction (the grad is set later by the
        deferred-grad manager).
        """
        if self._qkv_replaced:
            return
        self._qkv_replaced = True
        for name in ('q', 'k', 'v'):
            linear = getattr(self.model, name)
            # Mark weight as deferred so DDP / manual-sync can handle it
            linear.weight._deferred_grad = True
            if linear.bias is not None:
                linear.bias._deferred_grad = True
            # Replace the forward method to route through deferred_linear
            linear._v2_original_forward = linear.forward

            def _make_deferred_forward(lin):
                def _forward(x):
                    return deferred_linear(
                        x, lin.weight, lin.bias, self.grad_manager)
                return _forward

            linear.forward = _make_deferred_forward(linear)

    # ------------------------------------------------------------------
    # Buffer / manager lifecycle — mirrors v1's sym_post sharing
    # ------------------------------------------------------------------

    def _create_buffers(self):
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, self.cfg.dim, out_dtype=torch.bfloat16)
        self._owns_sym_post = True
        self.grad_manager = DeferredGradManager(self.group, self.sp_size)
        self._owns_grad_manager = True

    def share_buffers_from(self, other):
        """Borrow the sym_post workspace and grad_manager from the owner layer."""
        self.sym_post = other.sym_post
        self._owns_sym_post = False
        if hasattr(other, 'grad_manager') and other.grad_manager is not None:
            self.grad_manager = other.grad_manager
            self._owns_grad_manager = False
            # Re-point QKV forward to the shared manager
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

    # ------------------------------------------------------------------
    # POST forward — GEMM+RS (identical to v1)
    # ------------------------------------------------------------------

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: local Wo shard GEMM fused with ReduceScatter, then shared bias."""
        full_m = lbs * lseq
        attn_local = o.reshape(full_m, self.local_hidden).contiguous()
        y = post_linear_v2(
            attn_local, self.Wo_r_local, self.sym_post,
            self.local_m, self.grad_manager)
        return y + self.model.o.bias
