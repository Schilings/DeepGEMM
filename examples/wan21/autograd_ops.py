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


# ---------------------------------------------------------------------------
# Fused standard Ulysses: GEMM+Norm+A2A PRE (Wo replicated)
# ---------------------------------------------------------------------------

class FusedPreQKVFunction(torch.autograd.Function):
    """Fused QKV projection + QK RMSNorm + A2A-transpose (Ulysses SP pre-attention).

    Forward (fused kernel):
      X_local @ Wqkv.T → RMSNorm(Q/K) → A2A-transpose → qkv [bs, seq, local_nqkv]
    Backward (PyTorch native, the dual):
      inverse A2A → RMSNorm backward → GEMM → grad_X_local

    The fused kernel computes the projection, norm and scatter in one launch.
    The backward path does NOT use a fused kernel: it re-computes the projection
    from saved (x, wqkv) to get the pre-norm activations needed by RMSNorm
    backward, then runs the GEMM gradient.  This is acceptable because PRE
    backward is not the performance bottleneck (POST backward AG+GEMM is).

    Args:
        x_local:        [bs*local_seq, hidden] — seq-sharded input.
        wqkv:           [3*hidden, hidden] — rank-major fused QKV weight (NT).
        norm_q_weight:  [hidden] fp32 — Q RMSNorm scale (or None).
        norm_k_weight:  [hidden] fp32 — K RMSNorm scale (or None).
        sym_pre:        ``UnifiedSymmBuffer`` with attention params.
        local_seq:      Per-rank sequence length (must be 128-aligned).
        bs:             Batch size.
        eps:            RMSNorm epsilon.
    """

    @staticmethod
    def forward(ctx, x_local, wqkv, norm_q_weight, norm_k_weight,
                sym_pre, local_seq, bs, eps):
        ctx.sym_pre = sym_pre
        ctx.local_seq = local_seq
        ctx.bs = bs
        ctx.eps = eps
        ctx.save_for_backward(x_local, wqkv, norm_q_weight, norm_k_weight)

        sp = sym_pre.group.size()
        q_nheads = sym_pre.q_nheads
        kv_nheads = sym_pre.kv_nheads
        head_dim = sym_pre.head_dim

        out, rms = bf16_fused_qkv_norm_a2a_nt(
            x_local, wqkv, sym_pre, local_seq,
            q_nheads, kv_nheads, head_dim, eps,
            norm_q_weight=norm_q_weight,
            norm_k_weight=norm_k_weight,
        )
        ctx.rms = rms  # [bs, seq, 2] fp32 — saved for backward
        return out  # [bs, seq, local_nqkv] bf16

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        x_local, wqkv, norm_q_weight, norm_k_weight = ctx.saved_tensors
        sym_pre = ctx.sym_pre
        local_seq = ctx.local_seq
        bs = ctx.bs
        eps = ctx.eps
        rms = ctx.rms  # [bs, seq, 2] fp32

        sp = sym_pre.group.size()
        hidden = wqkv.shape[1]  # K = hidden
        q_nheads = sym_pre.q_nheads
        kv_nheads = sym_pre.kv_nheads
        head_dim = sym_pre.head_dim
        q_dim = q_nheads * head_dim
        kv_dim = kv_nheads * head_dim
        n_total = q_dim + 2 * kv_dim  # 3 * hidden for MHA
        local_q_n = (q_nheads // sp) * head_dim
        local_kv_n = (kv_nheads // sp) * head_dim
        local_n = local_q_n + 2 * local_kv_n
        local_m = bs * local_seq
        seq = local_seq * sp

        # grad_out: [bs, seq, local_n] — inverse A2A to get grad_proj_normed
        # [bs*local_seq, n_total]
        grad_out = grad_out.contiguous()
        # Reshape: [bs, seq, local_n] → [bs, seq, sp, local_n/sp_item]
        # Actually local_n = local_q_n + 2*local_kv_n, and the A2A scatter
        # split Q/K/V each by head group.  Inverse: gather head groups back.
        #
        # Forward scatter: rank r's GEMM output [bs*local_seq, n_total] is
        # split into sp head groups, each group sent to a different dst.
        # Each dst receives [bs, local_seq, local_n] (its head group for Q/K/V).
        #
        # Inverse: each rank sends its [bs, local_seq, local_n] pieces back,
        # concatenating along the head dimension to reconstruct [bs*local_seq, n_total].
        #
        # The scatter is per-QKV-segment (Q, K, V each split independently).
        # For MHA (q_nheads == kv_nheads), local_q_n == local_kv_n, so the
        # inverse is a simple all-to-all on the flattened [bs, seq, local_n]
        # treated as [bs, sp, local_seq, local_n_per_rank].

        # Simplify: for MHA, local_n = 3 * local_q_n.
        # grad_out [bs, seq, local_n] = [bs, sp*local_seq, 3*local_q_n]
        # The scatter was: rank r writes to dst's [bs, r*local_seq:(r+1)*local_seq, :]
        # So inverse A2A: send [bs, seq, local_n] split by seq → recv by head
        # Actually the scatter is by head group (N dim), not seq.
        # Let me re-derive from the design doc:
        #   Forward: GEMM output D[global_m, n], n in [0, n_total)
        #     Q: dst_rank = n / local_q_n, base_n = n % local_q_n
        #     K: dst_rank = (n-q_dim) / local_kv_n, base_n = n%local_kv_n + local_q_n
        #     V: similar
        #   So the output [bs, seq, local_n] has local_n = local_q_n + 2*local_kv_n
        #   and the seq dim is the FULL seq (each rank has full seq, local heads).

        # Inverse A2A: reconstruct [bs*local_seq, n_total] from [bs, seq, local_n]
        # For MHA (q_nheads == kv_nheads): local_q_n == local_kv_n == hidden // sp
        # local_n = 3 * hidden // sp
        # The forward scatter split each of Q/K/V into sp groups by head.
        # Inverse: gather sp groups back for each of Q/K/V.
        #
        # Reshape grad_out to [bs, seq, 3, local_q_n] (Q/K/V segments)
        # then for each segment, treat as [bs, seq, sp, local_q_n_per_rank]
        # and all_to_all to gather.

        # For MHA: local_q_n == local_kv_n == local_hidden
        local_hidden = local_q_n  # == local_kv_n for MHA
        grad_qkv = grad_out.view(bs, seq, 3, local_hidden)  # [bs, seq, 3, local_hidden]

        # Forward: rank r's Q segment [q_dim] was split into sp groups,
        # group r sent to dst's seq offset r*local_seq.
        # Wait, I need to re-read the scatter index more carefully.
        #
        # From GEMM_A2A_TRANSPOSE_DESIGN.md:
        #   out_dst[ b*seq + r*local_seq + s_local , n_local ] = D[global_m, n]
        # So the output is [bs*seq, local_N], row = b*seq + r*local_seq + s_local.
        # This means rank r writes to dst's rows [r*local_seq, (r+1)*local_seq)
        # (within each batch), and columns [0, local_N).
        #
        # So the A2A is along the SEQ dimension (M), not the N dimension!
        # Each rank has FULL heads (all of Q/K/V for its head group),
        # but only LOCAL_SEQ tokens.
        #
        # Wait, that's for gemm_a2a_transpose (PRE without norm).
        # For fused_qkv_norm_a2a, the scatter is different because Q/K/V
        # have different head counts in GQA.  But for MHA they're the same.
        #
        # Let me check: the kernel output is [bs, seq, local_n_total]
        # where local_n_total = (local_q_nheads + 2*local_kv_nheads) * head_dim
        # and seq = local_seq * sp (FULL seq).
        #
        # So each rank has FULL seq, LOCAL heads. The A2A scatter is along
        # the HEAD dimension (N), sending head groups to different ranks.
        # Each rank's GEMM output [bs*local_seq, n_total] is split by N
        # into sp groups, each group goes to a different dst.
        #
        # But the output layout is [bs, seq, local_n] = [bs, full_seq, local_heads*hd]
        # where seq = local_seq * sp.  This means the output has:
        # - rows: all seq tokens (from all ranks, each rank contributes local_seq)
        # - cols: local head group (local_q_n + 2*local_kv_n)
        #
        # So the A2A is BOTH seq-scatter AND head-gather:
        # - Each rank sends its local_seq tokens to all other ranks (seq scatter)
        # - Each rank receives head groups from all other ranks (head gather)
        #
        # This is exactly the transpose: [bs, local_seq, all_heads] → [bs, all_seq, local_heads]
        #
        # Inverse: [bs, all_seq, local_heads] → [bs, local_seq, all_heads]
        # = inverse A2A transpose

        # For MHA, the Q/K/V segments are contiguous in local_n:
        # local_n = [local_q_n | local_kv_n | local_kv_n]
        # We can treat the whole thing as [bs, seq, local_n] and do inverse A2A
        # on the seq dimension, scattering local_n back to full n_total.

        # Actually, looking at the output shape [bs, seq, local_n]:
        # - seq = local_seq * sp (full seq, each rank has all tokens)
        # - local_n = local_q_n + 2*local_kv_n (local head group)
        #
        # The forward A2A was: rank r produces [bs, local_seq, n_total]
        # → scatter to sp ranks: each dst gets [bs, local_seq, local_n] at seq offset r
        # → dst has [bs, sp*local_seq, local_n] = [bs, seq, local_n]
        #
        # Inverse: rank r has [bs, seq, local_n]
        # → for each src rank s, extract [bs, s*local_seq:(s+1)*local_seq, local_n]
        # → all_to_all: send these slices to src ranks
        # → each rank receives sp slices of [bs, local_seq, local_n]
        # → concatenate along N to get [bs, local_seq, sp*local_n] = [bs, local_seq, n_total]

        # grad_out was permuted in forward to [local_seq, sp] ordering
        # (matching serial baseline).  Undo that permutation before inverse A2A:
        # view as [bs, local_seq, sp, local_n] → permute to [sp, bs, local_seq, local_n]
        send = grad_out.view(bs, local_seq, sp, local_n).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=sym_pre.group)
        # recv: [sp, bs, local_seq, local_n] → cat along sp (dim 0) to get
        # [bs, local_seq, sp*local_n] = [bs, local_seq, n_total]
        grad_proj_normed = recv.permute(1, 2, 0, 3).reshape(bs * local_seq, n_total)

        # Now grad_proj_normed is [bs*local_seq, n_total] with Q/K/V segments.
        # But this is POST-norm gradient. We need to go through norm backward
        # to get grad_proj (pre-norm).
        #
        # RMSNorm backward: y = x * rms * w, rms = rsqrt(mean(x²) + eps)
        # grad_x = grad_y * rms * w - x * (rms³ / dim) * sum(grad_y * x * w)
        # grad_w = sum(grad_y * x)  (per-feature)
        #
        # But the norm was done on the FULL dim (before A2A scatter), so
        # rms is [bs, local_seq, 2] (one for Q, one for K), computed from
        # the pre-norm projection [bs*local_seq, q_dim] and [bs*local_seq, kv_dim].
        #
        # The fused kernel returned rms = [bs, seq, 2], but seq = local_seq * sp.
        # Each rank only needs its own local_seq portion: rms[:, r*local_seq:(r+1)*local_seq, :]
        # Wait, rms is computed per-token (per row of the GEMM), and each rank
        # computed it for its own local_seq tokens. The kernel scatters rms too.
        # So rms[:, r*local_seq:(r+1)*local_seq, :] is rank r's contribution.
        # But we need the rms for OUR local_seq tokens, which are at
        # rms[:, :local_seq, :] (our rank's tokens are at the beginning? No...)
        #
        # Actually, the kernel computes rms for the GEMM rows = bs*local_seq
        # (our local seq shard). Then it scatters the un-normed data + rms to
        # peers. The rms that the kernel returns is for OUR local_seq tokens.
        # But the output `out` has shape [bs, seq, local_n] = [bs, sp*local_seq, local_n]
        # and rms has shape [bs, seq, 2] = [bs, sp*local_seq, 2].
        #
        # Hmm, that doesn't match. Let me re-read the kernel API:
        # out: [bs, seq, local_n_total] bf16 (un-normed QKV, scattered by head)
        # rms: [bs, seq, 2] fp32
        # seq = local_seq * num_ranks
        #
        # So rms has full seq dimension. But rms is per-token, and each token's
        # rms was computed by the rank that owns that token (before A2A).
        # After A2A, each rank has all tokens but only local heads.
        # The rms for token (b, s) was computed by rank (s // local_seq).
        #
        # For backward, we need the rms for OUR local_seq tokens (the ones we
        # originally projected). Those are at rows [0, local_seq) in the
        # original GEMM space, which after A2A are at seq offsets
        # [r*local_seq, (r+1)*local_seq) where r = our rank.
        #
        # Wait, no. The GEMM input is x_local [bs*local_seq, hidden] (our seq shard).
        # GEMM output is [bs*local_seq, n_total]. RMSNorm is applied per-row.
        # Then A2A scatters by head group: each rank sends to all dsts.
        # The output [bs, seq, local_n] has:
        # - row b*seq + r*local_seq + s_local: from rank r, token s_local
        # - col: local head group
        #
        # So OUR tokens are at rows [r*local_seq, (r+1)*local_seq) within each batch.
        # r = sym_pre.group.rank()
        r = sym_pre.group.rank()
        rms_local = rms[:, r*local_seq:(r+1)*local_seq, :]  # [bs, local_seq, 2]

        # Similarly, grad_proj_normed we reconstructed is [bs*local_seq, n_total]
        # which is OUR local seq shard's gradient (all heads). This is correct
        # because the inverse A2A gathered all head groups back for our tokens.

        # RMSNorm backward for Q segment:
        grad_q_normed = grad_proj_normed[:, :q_dim]  # [bs*local_seq, q_dim]
        grad_k_normed = grad_proj_normed[:, q_dim:q_dim+kv_dim]
        # grad_v = grad_proj_normed[:, q_dim+kv_dim:]  # V has no norm

        # Re-compute pre-norm projection for norm backward.
        # saved_tensors strip requires_grad, so we re-enable it to build a
        # fresh autograd graph for the norm backward.
        with torch.enable_grad():
            x_local_g = x_local.detach().requires_grad_(True)
            wqkv_g = wqkv.detach().requires_grad_(True)
            proj = torch.matmul(x_local_g, wqkv_g.t())  # [bs*local_seq, n_total]
            proj.retain_grad()  # need grad_proj for GEMM backward
            proj_q = proj[:, :q_dim]  # [bs*local_seq, q_dim]
            proj_k = proj[:, q_dim:q_dim+kv_dim]

            # RMSNorm forward (recompute, keeping grad)
            if norm_q_weight is not None:
                nq_g = norm_q_weight.detach().requires_grad_(True)
                rms_q = torch.rsqrt(proj_q.float().pow(2).mean(-1, keepdim=True) + eps)
                q_normed = proj_q * rms_q.to(proj_q.dtype) * nq_g.to(proj_q.dtype)
            else:
                nq_g = None
                q_normed = proj_q

            if norm_k_weight is not None:
                nk_g = norm_k_weight.detach().requires_grad_(True)
                rms_k = torch.rsqrt(proj_k.float().pow(2).mean(-1, keepdim=True) + eps)
                k_normed = proj_k * rms_k.to(proj_k.dtype) * nk_g.to(proj_k.dtype)
            else:
                nk_g = None
                k_normed = proj_k

            # Backward through norm (single call, Q and K share the proj graph)
            torch.autograd.backward([q_normed, k_normed], [grad_q_normed, grad_k_normed])

        grad_proj = proj.grad  # [bs*local_seq, n_total]
        grad_x = torch.matmul(grad_proj, wqkv)
        grad_wqkv = torch.matmul(grad_proj.t(), x_local)

        grad_nq = nq_g.grad if (nq_g is not None) else None
        grad_nk = nk_g.grad if (nk_g is not None) else None

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
