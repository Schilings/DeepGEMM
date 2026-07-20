"""Autograd wrappers for Ulysses SP fused operators.

Two Function families are exposed:

- ``FusedPostLinearFunction`` (variant): forward GEMM+ReduceScatter, backward
  AllGather+GEMM.  Wo is input-column sharded; one ``UnifiedSymmBuffer`` is
  shared between forward and backward.

- ``FusedPostWoFunction`` (fused): forward A2A-transpose+GEMM (gather heads +
  Wo projection), backward GEMM+A2A-transpose (the dual).  Wo is replicated,
  identical to the standard Ulysses POST but with DeepGEMM fused communication.

All symmetric workspaces are owned by the strategy, not by autograd, so one
allocation can be reused across layers and iterations.
"""

import torch
from torch.autograd.function import once_differentiable

from deep_gemm import (
    bf16_ag_gemm_nt_with_input,
    bf16_gemm_rs_nt,
)
from deep_gemm.a2a_transpose_gemm import (
    bf16_a2a_transpose_gemm_nt,
)


# ---------------------------------------------------------------------------
# Variant: GEMM+RS forward, AG+GEMM backward (Wo input-column sharded)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fused standard Ulysses: A2A+GEMM POST (Wo replicated)
# ---------------------------------------------------------------------------

class FusedPostWoFunction(torch.autograd.Function):
    """Fused A2A-transpose + Wo GEMM (Ulysses SP post-attention).

    Forward:  A2A-transpose attn output →  Wo GEMM  →  y [bs*local_seq, hidden]
    Backward: GEMM grad →  A2A-transpose  →  grad_attn (dual)

    Wo is replicated (standard Ulysses).  The attention output is passed as
    ``attn_output`` and copied into ``sym_post.x`` inside forward; this keeps
    ``attn_output`` in the autograd graph so backward can return its gradient.

    Args:
        attn_output: [bs, seq, local_nh, hd] — FlashAttention output (BSHD).
        wo_weight:   [hidden, hidden] — replicated Wo weight (NT layout).
        sym_post:    ``BF16A2ATransposeGemmSymmBuffer`` owned by the strategy.
        local_m:     bs * local_seq (output token count on this rank).
        bs:          Batch size.
    """

    @staticmethod
    def forward(ctx, attn_output, wo_weight, sym_post, local_m, bs):
        ctx.sym_post = sym_post
        ctx.local_m = local_m
        ctx.bs = bs
        ctx.save_for_backward(wo_weight)

        # Copy attention output into the POST symmetric buffer.
        # attn_output is FA4 BSHD [bs, seq, local_nh, hd]; sym_post.x is BHSD
        # [bs, local_nh, seq, hd].  The comm kernel (seq_major=False) reads BHSD.
        sym_post.x.copy_(attn_output.transpose(1, 2))

        hidden = wo_weight.shape[0]
        y = wo_weight.new_empty((local_m, hidden))
        bf16_a2a_transpose_gemm_nt(y, wo_weight, sym_post)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y):
        (wo_weight,) = ctx.saved_tensors
        sym_post = ctx.sym_post
        local_m = ctx.local_m
        bs = ctx.bs
        sp = sym_post.group.size()
        hidden = wo_weight.shape[0]
        local_seq = sym_post.local_seq
        local_nh = sym_post.nheads // sp
        hd = sym_post.head_dim

        grad_y = grad_y.contiguous()  # [local_m, hidden]

        # Backward of A2A+GEMM is GEMM+A2A:
        #   Forward: y = gathered @ wo_weight.T  (NT layout)
        #     gathered = A2A_transpose(attn_output)  [bs*local_seq, hidden]
        #   1. grad_gathered = grad_y @ wo_weight   [local_m, hidden]
        #   2. grad_wo = grad_y.T @ gathered         [hidden, hidden]
        #   3. inverse A2A-transpose on grad_gathered → grad_attn_output

        gathered = sym_post.gathered  # [bs*local_seq, hidden]

        # Step 1: grad w.r.t. gathered input
        grad_gathered = torch.matmul(grad_y, wo_weight)  # [local_m, hidden]

        # Step 2: grad w.r.t. wo_weight
        grad_wo = torch.matmul(grad_y.t(), gathered)  # [hidden, hidden]

        # Step 3: inverse A2A-transpose
        # Forward: BHSD [bs, local_nh, seq, hd] → A2A → [bs*local_seq, hidden]
        # Inverse: scatter seq shards back → BHSD, then transpose to BSHD to
        # match attn_output's layout.
        grad_gathered = grad_gathered.view(bs, local_seq, sp, local_nh, hd)
        send = grad_gathered.permute(2, 0, 1, 3, 4).contiguous()
        import torch.distributed as dist
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=sym_post.group)
        # recv: [sp, bs, local_seq, local_nh, hd] → BHSD [bs, local_nh, seq, hd]
        grad_bhsd = recv.permute(1, 3, 0, 2, 4).reshape(bs, local_nh, sp * local_seq, hd)
        # Transpose BHSD → BSHD to match attn_output
        grad_attn = grad_bhsd.transpose(1, 2).contiguous()
        return grad_attn, grad_wo, None, None, None


def fused_post_wo(attn_output, wo_weight, sym_post, local_m, bs):
    """Apply fused A2A-transpose + Wo GEMM."""
    return FusedPostWoFunction.apply(attn_output, wo_weight, sym_post, local_m, bs)


# ---------------------------------------------------------------------------
# Compatibility aliases for older imports.
# ---------------------------------------------------------------------------

GemmRSFunction = FusedPostLinearFunction

def gemm_rs(attn, weight, sym_buffer, layout_info):
    return fused_post_linear(attn, weight, sym_buffer, layout_info['local_m'])
