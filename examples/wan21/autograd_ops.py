"""Autograd wrappers for Ulysses SP fused operators.

Three Function families are exposed:

- ``FusedPostLinearFunction`` (variant): forward GEMM+ReduceScatter, backward
  AllGather+GEMM.  Wo is input-column sharded; one ``UnifiedSymmBuffer`` is
  shared between forward and backward.

- ``FusedPreQKVFunction`` (fused): forward GEMM+RMSNorm+A2A-transpose (QKV
  projection + QK norm + head/sequence scatter), backward inverse A2A + RMSNorm
  backward + GEMM (the dual, using PyTorch autograd for the norm backward).

- ``FusedPostWoFunction`` (fused): forward A2A-transpose+GEMM (gather heads +
  Wo projection), backward GEMM+A2A-transpose (the dual).  Wo is replicated,
  identical to the standard Ulysses POST but with DeepGEMM fused communication.

All symmetric workspaces are owned by the strategy, not by autograd, so one
allocation can be reused across layers and iterations.
"""

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from deep_gemm import (
    bf16_ag_gemm_nt_with_input,
    bf16_gemm_rs_nt,
)
from deep_gemm.a2a_transpose_gemm import (
    bf16_a2a_transpose_gemm_nt,
)
from deep_gemm.fused_qkv_norm_a2a import (
    bf16_fused_qkv_norm_a2a_nt,
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


def _rmsnorm_backward(grad_y, x, rms, w, dim, eps):
    """Analytical RMSNorm backward.

    y = x * rms * w,  rms = rsqrt(mean(x²) + eps)
    grad_x = grad_y * rms * w - x * (rms³ / dim) * sum(grad_y * x * w, -1, keepdim)

    All inputs are float32 for numerical stability.
    """
    grad_yxw = grad_y * x * w  # [M, dim]
    s = grad_yxw.sum(-1, keepdim=True)  # [M, 1]
    grad_x = grad_y * rms * w - x * (rms.pow(3) / dim) * s
    return grad_x


# ---------------------------------------------------------------------------
# Fused standard Ulysses: GEMM+Norm+A2A PRE (Wo replicated)
# ---------------------------------------------------------------------------

class FusedPreQKVFunction(torch.autograd.Function):
    """Fused QKV projection + QK RMSNorm + A2A-transpose (Ulysses SP pre-attention).

    Forward (fused kernel):
      X_local @ Wqkv.T → RMSNorm(Q/K) → A2A-transpose → qkv [bs, seq, local_nqkv]
    Backward (analytical, per docs/FUSED_QKV_NORM_A2A.md §对偶 backward):
      inverse A2A → Norm-inverse → GEMM

    Norm-inverse formula (docs line 97):
      grad_x = grad_y * rms * w - x * (rms³ / dim) * sum(grad_y * x * w, -1, keepdim)

    The pre-norm projection is recomputed from saved (x, wqkv) via a single
    GEMM — cheaper than PyTorch autograd's retain_graph + backward overhead.
    """

    @staticmethod
    def forward(ctx, x_local, wqkv, norm_q_weight, norm_k_weight,
                sym_pre, local_seq, bs, eps):
        ctx.sym_pre = sym_pre
        ctx.local_seq = local_seq
        ctx.bs = bs
        ctx.eps = eps
        ctx.save_for_backward(x_local, wqkv, norm_q_weight, norm_k_weight)

        q_nheads = sym_pre.q_nheads
        kv_nheads = sym_pre.kv_nheads
        head_dim = sym_pre.head_dim

        x_bf16 = x_local.to(torch.bfloat16) if x_local.dtype != torch.bfloat16 else x_local
        w_bf16 = wqkv.to(torch.bfloat16) if wqkv.dtype != torch.bfloat16 else wqkv
        nq_bf16 = norm_q_weight.to(torch.bfloat16) if (norm_q_weight is not None and norm_q_weight.dtype != torch.bfloat16) else norm_q_weight
        nk_bf16 = norm_k_weight.to(torch.bfloat16) if (norm_k_weight is not None and norm_k_weight.dtype != torch.bfloat16) else norm_k_weight

        out, rms = bf16_fused_qkv_norm_a2a_nt(
            x_bf16, w_bf16, sym_pre, local_seq,
            q_nheads, kv_nheads, head_dim, eps,
            norm_q_weight=nq_bf16,
            norm_k_weight=nk_bf16,
        )
        ctx.rms = rms  # [bs, seq, 2] fp32
        sp = sym_pre.group.size()
        out = out.view(bs, sp, local_seq, -1).transpose(1, 2).reshape(bs, sp * local_seq, -1)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x_local, wqkv, norm_q_weight, norm_k_weight = ctx.saved_tensors
        sym_pre = ctx.sym_pre
        local_seq = ctx.local_seq
        bs = ctx.bs
        eps = ctx.eps
        rms = ctx.rms

        sp = sym_pre.group.size()
        hidden = wqkv.shape[1]
        q_nheads = sym_pre.q_nheads
        kv_nheads = sym_pre.kv_nheads
        head_dim = sym_pre.head_dim
        q_dim = q_nheads * head_dim
        kv_dim = kv_nheads * head_dim
        n_total = q_dim + 2 * kv_dim
        local_q_n = (q_nheads // sp) * head_dim
        local_kv_n = (kv_nheads // sp) * head_dim
        local_n = local_q_n + 2 * local_kv_n
        local_hidden = local_q_n  # MHA: == local_kv_n

        # --- 1. Inverse A2A: grad_out [local_seq, sp] → grad_proj_normed [bs*local_seq, n_total] ---
        grad_out = grad_out.contiguous()
        send = grad_out.view(bs, local_seq, sp, local_n).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=sym_pre.group)
        recv = recv.view(sp, bs, local_seq, 3, local_hidden)
        recv = recv.permute(1, 2, 3, 0, 4)  # [bs, ls, 3, sp, lh]
        grad_proj_normed = recv.reshape(bs * local_seq, n_total)

        # --- 2. Extract rms for our tokens ---
        r = sym_pre.group.rank()
        rms_local = rms[:, r*local_seq:(r+1)*local_seq, :]  # [bs, local_seq, 2]
        rms_q = rms_local[:, :, 0:1].reshape(bs * local_seq, 1).float()  # [bs*local_seq, 1]
        rms_k = rms_local[:, :, 1:2].reshape(bs * local_seq, 1).float()

        # --- 3. Recompute pre-norm proj (single GEMM) ---
        x_bf16 = x_local.to(torch.bfloat16)
        w_bf16 = wqkv.to(torch.bfloat16)
        proj = torch.matmul(x_bf16, w_bf16.t())  # [bs*local_seq, n_total]
        proj_q = proj[:, :q_dim].float()
        proj_k = proj[:, q_dim:q_dim+kv_dim].float()
        proj_v = proj[:, q_dim+kv_dim:]  # bf16, no norm

        # --- 4. Norm-inverse (analytical, no autograd) ---
        # y = x * rms * w  →  grad_x = grad_y * rms * w - x * (rms³/dim) * sum(grad_y * x * w, -1, keepdim)
        grad_q_normed = grad_proj_normed[:, :q_dim].float()
        grad_k_normed = grad_proj_normed[:, q_dim:q_dim+kv_dim].float()
        grad_v = grad_proj_normed[:, q_dim+kv_dim:]  # bf16

        if norm_q_weight is not None:
            nq = norm_q_weight.float()
            grad_proj_q = _rmsnorm_backward(grad_q_normed, proj_q, rms_q, nq, q_dim, eps)
            grad_nq = (grad_q_normed * proj_q * rms_q).sum(0)
        else:
            grad_proj_q = grad_q_normed
            grad_nq = None

        if norm_k_weight is not None:
            nk = norm_k_weight.float()
            grad_proj_k = _rmsnorm_backward(grad_k_normed, proj_k, rms_k, nk, kv_dim, eps)
            grad_nk = (grad_k_normed * proj_k * rms_k).sum(0)
        else:
            grad_proj_k = grad_k_normed
            grad_nk = None

        # Assemble grad_proj [bs*local_seq, n_total]
        grad_proj = torch.empty_like(proj)
        grad_proj[:, :q_dim] = grad_proj_q.to(torch.bfloat16)
        grad_proj[:, q_dim:q_dim+kv_dim] = grad_proj_k.to(torch.bfloat16)
        grad_proj[:, q_dim+kv_dim:] = grad_v

        # --- 5. GEMM backward ---
        grad_x = torch.matmul(grad_proj, w_bf16)
        grad_wqkv = torch.matmul(grad_proj.t(), x_bf16)

        return grad_x, grad_wqkv, grad_nq, grad_nk, None, None, None, None, None


def fused_pre_qkv(x_local, wqkv, norm_q_weight, norm_k_weight,
                  sym_pre, local_seq, bs, eps=1e-6):
    """Apply fused QKV projection + RMSNorm + A2A-transpose."""
    return FusedPreQKVFunction.apply(x_local, wqkv, norm_q_weight, norm_k_weight,
                                     sym_pre, local_seq, bs, eps)


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
        # attn_output is FA4 BSHD [bs, seq, local_nh, hd]; views.x is BHSD
        # [bs, local_nh, seq, hd].  The comm kernel (seq_major=False) reads BHSD.
        sym_post.a2a_gemm.x.copy_(attn_output.transpose(1, 2))

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

        gathered = sym_post.a2a_gemm.gathered  # [bs*local_seq, hidden]

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
