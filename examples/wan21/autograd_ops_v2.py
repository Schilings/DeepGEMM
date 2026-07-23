"""Autograd wrappers for Ulysses variant v2: delayed weight-grad overlap.

Design
------
*Forward* is identical to v1 — GEMM+ReduceScatter via the DeepGEMM fused op.

*POST backward* switches from the fused AG+GEMM operator to **native NCCL
all_gather + native GEMM** (same as the serial baseline).  The AG is launched
on a dedicated comm stream so it can overlap with computation on the default
stream.

*QKV weight gradients* are **deferred**:  during each layer's PRE backward the
DeferredLinearFunction computes ``grad_x`` (needed immediately by the previous
layer) but pushes ``(x, grad_output, weight)`` onto a shared queue instead of
computing ``grad_W`` right away.

In the **next** layer's POST backward, while the AG runs on the comm stream,
the deferred weight-grad GEMMs are executed on the default stream — effectively
hiding the QKV weight-grad cost behind the AG latency.

The last layer in backward (layer 0) has no subsequent POST to overlap with, so
its deferred grads are flushed by an explicit ``finalize()`` call after
``loss.backward()``.

All state (comm stream, deferred queue, AG buffer) lives in a single
``DeferredGradManager`` shared across all attention layers, mirroring how the
``UnifiedSymmBuffer`` is shared in v1.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from deep_gemm import bf16_gemm_rs_nt


# ---------------------------------------------------------------------------
# Deferred-grad manager — one per model, shared across all layers
# ---------------------------------------------------------------------------

class DeferredGradManager:
    """Owns the comm stream, the deferred-grad queue, and the AG scratch buffer.

    A single instance is created by the first attention layer and borrowed by
    all subsequent layers (same pattern as ``UnifiedSymmBuffer``).
    """

    def __init__(self, group, sp_size: int):
        self.group = group
        self.sp_size = sp_size
        self.comm_stream = torch.cuda.Stream()
        self._queue: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._ag_buffer: torch.Tensor | None = None

    # -- AG scratch buffer ---------------------------------------------------

    def get_ag_buffer(self, full_m: int, hidden: int, dtype, device) -> torch.Tensor:
        """Return a reusable contiguous buffer for all_gather_into_tensor."""
        if (self._ag_buffer is None
                or self._ag_buffer.shape[0] != full_m
                or self._ag_buffer.shape[1] != hidden
                or self._ag_buffer.dtype != dtype
                or self._ag_buffer.device != device):
            self._ag_buffer = torch.empty(full_m, hidden, dtype=dtype, device=device)
        return self._ag_buffer

    # -- deferred queue ------------------------------------------------------

    def push(self, x: torch.Tensor, grad_output: torch.Tensor,
             weight: torch.Tensor) -> None:
        """Queue a weight-grad GEMM for later execution."""
        self._queue.append((x, grad_output, weight))

    def process(self) -> None:
        """Execute all queued weight-grad GEMMs on the current (default) stream.

        Called from the POST backward while the AG runs on *comm_stream*.
        """
        while self._queue:
            x, grad_output, weight = self._queue.pop(0)
            # Cast to a common dtype (weight's dtype, typically bf16) to avoid
            # matmul dtype mismatch when grad flows through float32 norm layers.
            w_dtype = weight.dtype
            if x.dtype != w_dtype:
                x = x.to(w_dtype)
            if grad_output.dtype != w_dtype:
                grad_output = grad_output.to(w_dtype)
            grad_w = torch.matmul(grad_output.t(), x)
            if weight.grad is not None:
                weight.grad.add_(grad_w)
            else:
                weight.grad = grad_w

    def finalize(self) -> None:
        """Flush any remaining deferred grads (last layer in backward)."""
        self.process()

    @property
    def pending(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Deferred-weight-grad Linear  (replaces Q/K/V nn.Linear in the PRE)
# ---------------------------------------------------------------------------

class DeferredLinearFunction(torch.autograd.Function):
    """``nn.Linear`` whose weight gradient is deferred.

    * Forward — identical to ``F.linear(x, weight, bias)``.
    * Backward — computes ``grad_x = grad_output @ weight`` immediately (the
      previous layer needs it) and pushes ``(x, grad_output, weight)`` onto the
      shared queue.  The weight gradient itself is computed later during the
      next POST backward's AG overlap window.

    Bias gradient is computed immediately (cheap ``sum`` — not worth deferring).
    """

    @staticmethod
    def forward(ctx, x, weight, bias, grad_manager):
        ctx.grad_manager = grad_manager
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight)
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # Cast grad_output to weight's dtype for matmul (handles float32→bf16
        # when grad flows through float32 norm/modulation layers).
        w_dtype = weight.dtype
        go = grad_output.to(w_dtype) if grad_output.dtype != w_dtype else grad_output
        # grad_x — needed by the previous layer's backward (do NOT defer)
        grad_x = go @ weight
        if grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
        # Defer the weight gradient
        ctx.grad_manager.push(x.detach(), go.detach(), weight)
        # Bias gradient is trivial (sum over batch); compute immediately
        grad_bias = grad_output.sum(dim=0) if ctx.has_bias else None
        return grad_x, None, grad_bias, None


def deferred_linear(x, weight, bias, grad_manager):
    """Drop-in replacement for ``nn.Linear.forward`` with deferred weight grad."""
    return DeferredLinearFunction.apply(x, weight, bias, grad_manager)


# ---------------------------------------------------------------------------
# POST linear — GEMM+RS forward, native AG+GEMM backward with overlap
# ---------------------------------------------------------------------------

class PostLinearV2Function(torch.autograd.Function):
    """Column-sharded output linear with fused ReduceScatter forward and
    **native** AG+GEMM backward.

    Forward: identical to v1 — ``bf16_gemm_rs_nt`` (DeepGEMM fused GEMM+RS).

    Backward:
      1. Launch ``dist.all_gather_into_tensor`` on *comm_stream*.
      2. Process deferred QKV weight grads on the *default* stream — overlaps
         with the AG communication.
      3. Wait for AG.
      4. ``grad_attn = grad_y_full @ weight``      [full_m, local_hidden]
      5. ``grad_weight = grad_y_full.T @ attn``     [hidden, local_hidden]
    """

    @staticmethod
    def forward(ctx, attn, weight, workspace, local_m, grad_manager):
        if attn.ndim != 2 or weight.ndim != 2:
            raise ValueError('attn and weight must be 2D tensors')
        if attn.shape[1] != weight.shape[1]:
            raise ValueError('attn K dimension must match the local Wo shard')
        sp = workspace.group.size()
        if attn.shape[0] != local_m * sp:
            raise ValueError('attn rows must equal local_m * sequence-parallel size')

        ctx.workspace = workspace
        ctx.local_m = local_m
        ctx.grad_manager = grad_manager
        ctx.save_for_backward(attn, weight)

        output = attn.new_empty((local_m, weight.shape[0]))
        bf16_gemm_rs_nt(output, attn, weight, workspace, local_m)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        attn, weight = ctx.saved_tensors
        local_m = ctx.local_m
        gm = ctx.grad_manager
        group = ctx.workspace.group
        sp = group.size()
        full_m = local_m * sp
        hidden = weight.shape[0]

        grad_output = grad_output.contiguous()
        # Cast to weight's dtype (bf16) — handles float32 grad_output from
        # float32 loss / norm layers.
        w_dtype = weight.dtype
        if grad_output.dtype != w_dtype:
            grad_output = grad_output.to(w_dtype)

        # 1 — Launch AG on the comm stream (overlaps with step 2)
        grad_y_full = gm.get_ag_buffer(full_m, hidden, w_dtype,
                                       grad_output.device)
        gm.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gm.comm_stream):
            dist.all_gather_into_tensor(grad_y_full, grad_output, group=group)

        # 2 — Process deferred QKV weight grads on the default stream
        #     (runs concurrently with the AG on comm_stream)
        gm.process()

        # 3 — Ensure AG has finished before reading grad_y_full
        torch.cuda.current_stream().wait_stream(gm.comm_stream)

        # 4 — grad_attn = grad_y_full @ weight   [full_m, local_hidden]
        grad_attn = torch.matmul(grad_y_full, weight)
        if grad_attn.dtype != attn.dtype:
            grad_attn = grad_attn.to(attn.dtype)

        # 5 — grad_weight = grad_y_full.T @ attn   [hidden, local_hidden]
        grad_weight = torch.matmul(grad_y_full.t(), attn)

        return grad_attn, grad_weight, None, None, None


def post_linear_v2(attn, weight, workspace, local_m, grad_manager):
    """Apply the v2 POST linear (GEMM+RS fwd, native AG+GEMM bwd with overlap)."""
    return PostLinearV2Function.apply(attn, weight, workspace, local_m, grad_manager)


# ---------------------------------------------------------------------------
# POST linear v2 + Wo deferred — also defers grad_weight to next layer's AG
# ---------------------------------------------------------------------------

class PostLinearV2WoFunction(torch.autograd.Function):
    """Same as PostLinearV2Function but **also defers the Wo weight gradient**.

    Backward:
      1. Launch ``dist.all_gather_into_tensor`` on *comm_stream*.
      2. Process deferred QKV (and previous layer's Wo) weight grads on the
         default stream — overlaps with AG.
      3. Wait for AG.
      4. ``grad_attn = grad_y_full @ weight``   [full_m, local_hidden]  (immediate)
      5. Push ``(attn, grad_y_full, weight)`` onto deferred queue for next layer.
    """

    @staticmethod
    def forward(ctx, attn, weight, workspace, local_m, grad_manager):
        if attn.ndim != 2 or weight.ndim != 2:
            raise ValueError('attn and weight must be 2D tensors')
        if attn.shape[1] != weight.shape[1]:
            raise ValueError('attn K dimension must match the local Wo shard')
        sp = workspace.group.size()
        if attn.shape[0] != local_m * sp:
            raise ValueError('attn rows must equal local_m * sequence-parallel size')

        ctx.workspace = workspace
        ctx.local_m = local_m
        ctx.grad_manager = grad_manager
        ctx.save_for_backward(attn, weight)

        output = attn.new_empty((local_m, weight.shape[0]))
        bf16_gemm_rs_nt(output, attn, weight, workspace, local_m)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        attn, weight = ctx.saved_tensors
        local_m = ctx.local_m
        gm = ctx.grad_manager
        group = ctx.workspace.group
        sp = group.size()
        full_m = local_m * sp
        hidden = weight.shape[0]

        grad_output = grad_output.contiguous()
        w_dtype = weight.dtype
        if grad_output.dtype != w_dtype:
            grad_output = grad_output.to(w_dtype)

        # 1 — Launch AG on comm stream
        grad_y_full = gm.get_ag_buffer(full_m, hidden, w_dtype,
                                       grad_output.device)
        gm.comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gm.comm_stream):
            dist.all_gather_into_tensor(grad_y_full, grad_output, group=group)

        # 2 — Process deferred QKV + previous Wo weight grads (overlap with AG)
        gm.process()

        # 3 — Wait for AG
        torch.cuda.current_stream().wait_stream(gm.comm_stream)

        # 4 — grad_attn = grad_y_full @ weight  (immediate — needed by prev layer)
        grad_attn = torch.matmul(grad_y_full, weight)
        if grad_attn.dtype != attn.dtype:
            grad_attn = grad_attn.to(attn.dtype)

        # 5 — Defer Wo weight grad: grad_weight = grad_y_full.T @ attn
        #     Pushed to queue; will be computed in next layer's AG window.
        gm.push(attn.detach(), grad_y_full.detach(), weight)

        return grad_attn, None, None, None, None


def post_linear_v2_wo(attn, weight, workspace, local_m, grad_manager):
    """v2 POST linear with **Wo weight grad also deferred**."""
    return PostLinearV2WoFunction.apply(attn, weight, workspace, local_m, grad_manager)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finalize_deferred_grads(model) -> None:
    """Flush remaining deferred weight grads after ``loss.backward()``.

    Call this *before* gradient synchronisation (DDP all-reduce or manual
    ``sync_replicated_grads``).  Safe to call on non-v2 models (no-op).
    """
    for module in model.modules():
        gm = getattr(module, 'grad_manager', None)
        if gm is not None:
            gm.finalize()
            break  # all layers share one manager


def sync_deferred_grads(model, group) -> None:
    """All-reduce deferred-grad parameters that were excluded from DDP.

    Only needed in ``--sync-mode ddp``; manual mode uses ``sync_replicated_grads``
    which already covers replicated parameters.
    """
    sp = group.size()
    for param in model.parameters():
        if getattr(param, '_deferred_grad', False) and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
            param.grad.div_(sp)
