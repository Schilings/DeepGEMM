"""Autograd wrapper for the POST-only Ulysses experiment.

The baseline contains no DeepGEMM operator.  This module exposes one Function
whose forward is GEMM+ReduceScatter and whose activation backward is the dual
AllGather+GEMM.  The long-lived symmetric workspace is owned by the strategy,
not by autograd, so one allocation can be reused across layers and iterations.
"""

import torch
from torch.autograd.function import once_differentiable

from deep_gemm import (
    bf16_ag_gemm_nt_with_input,
    bf16_gemm_rs_nt,
)


class FusedPostLinearFunction(torch.autograd.Function):
    """Column-sharded output linear with fused sequence ReduceScatter.

    Args:
        attn: Full-sequence local-head activation ``[full_m, local_hidden]``.
        weight: Local input-column shard of Wo ``[hidden, local_hidden]``.
        workspace: Reusable ``UnifiedSymmBuffer`` owned by the strategy.
        local_m: Number of output tokens owned by this rank.

    The C++ launchers own their stream/event and ready-state protocols.  In
    particular, GEMM-RS uses self-resetting barrier phases and AG+GEMM clears
    its ``slot_state`` on its communication stream.  Clearing the full workspace
    here would race peer signals, destroy overlap, and add hundreds of MB of
    memory traffic per call.
    """

    @staticmethod
    def forward(ctx, attn, weight, workspace, local_m):
        if attn.ndim != 2 or weight.ndim != 2:
            raise ValueError('attn and weight must be 2D tensors')
        if attn.shape[1] != weight.shape[1]:
            raise ValueError('attn K dimension must match the local Wo shard')
        if attn.shape[0] != local_m * workspace.group.size():
            raise ValueError('attn rows must equal local_m * sequence-parallel size')

        ctx.workspace = workspace
        ctx.local_m = local_m
        ctx.save_for_backward(attn, weight)

        output = attn.new_empty((local_m, weight.shape[0]))
        bf16_gemm_rs_nt(output, attn, weight, workspace, local_m)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        attn, weight = ctx.saved_tensors
        local_m = ctx.local_m
        grad_output = grad_output.contiguous()

        # AG+GEMM expects NT weight layout [local_hidden, hidden].  The helper
        # stages grad_output, launches the dual fused op, and returns the already
        # gathered grad_output view for the local weight-gradient GEMM.
        weight_t = weight.t().contiguous()
        grad_attn = attn.new_empty(attn.shape)
        grad_output_full = bf16_ag_gemm_nt_with_input(
            grad_attn, grad_output, weight_t, ctx.workspace, local_m)

        # dW = grad_output_full.T @ attn.  This orientation avoids forcing a
        # contiguous transpose of the much larger [full_m, hidden] tensor.
        grad_weight = torch.matmul(attn.t(), grad_output_full).t().contiguous()
        return grad_attn, grad_weight, None, None


def fused_post_linear(attn, weight, workspace, local_m):
    """Apply the POST-only fused linear while reusing ``workspace``."""
    return FusedPostLinearFunction.apply(attn, weight, workspace, local_m)


# Compatibility alias for older imports.
GemmRSFunction = FusedPostLinearFunction

def gemm_rs(attn, weight, sym_buffer, layout_info):
    return fused_post_linear(attn, weight, sym_buffer, layout_info['local_m'])
