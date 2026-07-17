"""torch.autograd.Function wrappers for fused communication-GEMM operators.

Each fused op is wrapped as an autograd.Function so it integrates with PyTorch's
autograd graph. This enables:
  1. Automatic backward via the dual operator (no manual backward needed)
  2. FSDP2 (fully_shard) integration — weight grads auto reduce-scatter via hooks
  3. Clean training framework integration (loss.backward() just works)

Dual relationships (forward ↔ backward):
  GEMM+A2A (pre-attn fwd)  ↔  A2A+GEMM (pre-attn bwd)  [gather heads]
  A2A+GEMM (post-attn fwd) ↔  GEMM+A2A (post-attn bwd)  [scatter heads]
  GEMM+RS  (post-attn var) ↔  AG+GEMM  (post-attn bwd)  [gather tokens]

Weight gradients: manually computed in backward (standard matmul, no overlap).
The fused kernel only computes the activation gradient; the weight grad is a
standard matmul that the Function's backward computes explicitly.
"""

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm import bf16_gemm_a2a_transpose_nt
from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt, bf16_a2a_transpose_gemm_nt_fused
from deep_gemm.gemm_rs import bf16_gemm_rs_nt
from deep_gemm.ag_gemm import bf16_ag_gemm_nt


# ════════════════════════════════════════════════════════════════════════════
# 1. GEMM + A2A-transpose (PRE-attn forward)
#    Forward:  x_local[local_m, K] @ Wqkv^T[N, K] → A2A-scatter(heads) → qkv[bs, seq, local_n]
#    Backward: grad_qkv → A2A-gather(heads) → grad_local[local_m, N] → grad_X = grad_local @ Wqkv
#              grad_Wqkv = grad_local^T @ x_local  (weight grad, standard matmul)
# ════════════════════════════════════════════════════════════════════════════

class GemmA2ATransposeFunction(torch.autograd.Function):
    """GEMM + A2A-transpose (Ulysses pre-attn). Dual backward = A2A + GEMM."""

    @staticmethod
    def forward(ctx, x_local, weight, sym_buffer, sym_buffer_bwd, local_seq, n_qkv, sp_size, group, layout_info):
        ctx.sym_buffer = sym_buffer
        ctx.sym_buffer_bwd = sym_buffer_bwd
        ctx.local_seq = local_seq
        ctx.n_qkv = n_qkv
        ctx.sp_size = sp_size
        ctx.group = group
        ctx.layout_info = layout_info
        ctx.save_for_backward(x_local, weight)
        qkv = bf16_gemm_a2a_transpose_nt(x_local, weight, sym_buffer, local_seq)
        lbs = layout_info['lbs']; lseq = layout_info['lseq']
        return qkv[:lbs, :lseq, :] if layout_info.get('thd') else qkv

    @staticmethod
    def backward(ctx, grad_qkv):
        x_local, weight = ctx.saved_tensors
        li = ctx.layout_info
        lbs = li['lbs']; lseq = li['lseq']; llseq = li['llseq']; lm = li['lm']
        local_nh_qkv = 3 * (li['nheads'] // ctx.sp_size)
        head_dim = li['head_dim']
        gqkv_bhsd = grad_qkv.view(lbs, lseq, local_nh_qkv, head_dim).transpose(1, 2).contiguous()
        ctx.sym_buffer_bwd.x.copy_(gqkv_bhsd)
        grad_X = torch.empty((lm, li['hidden']), dtype=torch.bfloat16, device=grad_qkv.device)
        bf16_a2a_transpose_gemm_nt(grad_X, weight.t(), ctx.sym_buffer_bwd)
        grad_local = ctx.sym_buffer_bwd.gathered[:lm, :ctx.n_qkv]
        grad_weight = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_weight, None, None, None, None, None, None, None


def gemm_a2a_transpose(x_local, weight, sym_buffer, sym_buffer_bwd, local_seq, n_qkv, sp_size, group, layout_info):
    return GemmA2ATransposeFunction.apply(x_local, weight, sym_buffer, sym_buffer_bwd, local_seq, n_qkv, sp_size, group, layout_info)


# ════════════════════════════════════════════════════════════════════════════
# 2. A2A + GEMM (POST-attn forward, serial/fused_std)
#    Forward:  o[bs, local_nh, seq, hd] → A2A-transpose → gathered[local_m, hidden] → @ Wo^T → y[local_m, N]
#    Backward: grad_y → grad_y @ Wo → grad_gathered → A2A-inv → grad_attn[bs, seq, local_nh, hd]
#              grad_Wo = grad_y^T @ gathered  (weight grad)
# ════════════════════════════════════════════════════════════════════════════

class A2ATransposeGemmFunction(torch.autograd.Function):
    """A2A-transpose + GEMM (Ulysses post-attn). Dual backward = GEMM + A2A."""

    @staticmethod
    def forward(ctx, o, weight, sym_buffer, layout_info):
        ctx.sym_buffer = sym_buffer
        ctx.layout_info = layout_info
        ctx.save_for_backward(o, weight)
        li = layout_info
        lm = li['local_m']
        sym_buffer.x.copy_(o.transpose(1, 2).contiguous())
        y = torch.empty((lm, li['hidden']), dtype=torch.bfloat16, device=o.device)
        bf16_a2a_transpose_gemm_nt_fused(y, weight, sym_buffer)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        o, weight = ctx.saved_tensors
        li = ctx.layout_info
        sp = li['sp_size']; hd = li['hd']; hidden = li['hidden']
        lbs = li['lbs']; lseq = li['lseq']; llseq = li['llseq']; lm = li['local_m']
        local_nh = li['local_nh']
        group = li['group']
        grad_gathered = torch.matmul(grad_y, weight)
        send_bwd = grad_gathered.view(lbs, llseq, sp, local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=group)
        grad_o = recv_bwd.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, local_nh, hd)
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        grad_weight = torch.matmul(grad_y.t(), gathered)
        return grad_o, grad_weight, None, None


def a2a_transpose_gemm(o, weight, sym_buffer, layout_info):
    return A2ATransposeGemmFunction.apply(o, weight, sym_buffer, layout_info)


# ════════════════════════════════════════════════════════════════════════════
# 3. GEMM + RS (POST-attn forward, fused_var variant)
#    Forward:  attn[local_m, local_hidden] @ Wo_r^T[local_N, local_hidden] → RS → y[local_m, local_N]
#    Backward: grad_y → AG → grad_y_full[full_m, local_N] → @ Wo_r → grad_attn[full_m, local_hidden]
#              grad_Wo_r = grad_y_full^T @ attn  (weight grad)
# ════════════════════════════════════════════════════════════════════════════

class GemmRSFunction(torch.autograd.Function):
    """GEMM + Reduce-Scatter (Ulysses post-attn variant). Dual backward = AG + GEMM.

    Uses a SINGLE UnifiedSymmBuffer (deep_gemm.get_unified_symm_buffer) shared
    across all layers and reused for both passes:
      - forward  (GEMM+RS)  uses the unified buffer's GEMM-RS workspace (partial@32)
      - backward (AG+GEMM) uses the unified buffer's AG views (.ag_x / .ag_slots_x)
    The two passes never run concurrently, so the one physical buffer is safe to reuse.

    Forward:  attn[full_m, local_hidden] @ Wo_r[dim, local_hidden].t()  →  RS  →  y[local_m, dim]
        bf16_gemm_rs_nt(y, a=attn, b=Wo_r, sym_buffer, local_m)
            a: [total_M, K] = [full_m, local_hidden]
            b: [N, K]       = [dim, local_hidden]      (Wo_r_local)
            y: [tokens_per_rank, N] = [local_m, dim]

    Backward: grad_y[local_m, dim]  →  AG  →  grad_y_full[full_m, dim]  →  @ Wo_r  →  grad_attn[full_m, local_hidden]
        bf16_ag_gemm_nt(d=grad_attn, b=Wo_r.t(), sym_buffer, local_m)
            x = grad_y [tokens_per_rank, K] = [local_m, dim]  (sym_buffer.ag_x)
            b: [N, K] = [local_hidden, dim]              (Wo_r_local.t() — NT layout)
            d: [full_M, N] = [full_m, local_hidden]
        grad_y_full = sym_buffer.ag_slots_x[:sp, :local_m, :local_N].reshape(full_m, local_N)
    """

    @staticmethod
    def forward(ctx, attn, weight, sym_buffer, layout_info):
        ctx.sym_buffer = sym_buffer
        ctx.layout_info = layout_info
        ctx.save_for_backward(attn, weight)
        li = layout_info
        local_m = li['local_m']
        local_N = li['local_N']  # = dim
        group = li['group']
        # Reset the shared unified buffer's signal/barrier region [0,32) (and
        # slot_state) before each GEMM-RS. The buffer is reused across ALL
        # layers' fwd GEMM-RS and the bwd AG-GEMM, so stale signal/slot
        # bytes from a previous op would corrupt this kernel's +1/-1 handshake.
        # The trailing synchronize() forces the symm_mem comm stream to flush
        # before we memset, avoiding a cross-stream race on the signal region.
        # (Mirrors tests/comm/test_unified_buffer.py Test 6, hardened.)
        group.barrier()
        torch.cuda.synchronize()
        sym_buffer.buffer.zero_()
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()
        y = torch.empty((local_m, local_N), dtype=torch.bfloat16, device=attn.device)
        bf16_gemm_rs_nt(y, attn, weight, sym_buffer, local_m)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        attn, weight = ctx.saved_tensors
        li = ctx.layout_info
        sp = li['sp_size']
        local_N = li['local_N']      # = dim
        local_hidden = li['local_hidden']
        full_m = li['full_m']
        local_m = li['local_m']
        group = li['group']

        # The forward GEMM-RS (and possibly other layers' ops on the shared buffer)
        # left the barrier/signal region [0,32) and slot_state dirty. AG-GEMM
        # needs a clean barrier to start its own +1/-1 protocol — otherwise the
        # gather/AG handshake reads stale signal values and produces wrong (often
        # non-deterministic) grad_attn. Mirror tests/comm/test_unified_buffer.py
        # (Test 6), hardened with an extra trailing synchronize() so the symm_mem
        # comm stream is fully flushed before we memset the signal region.
        group.barrier()
        torch.cuda.synchronize()
        ctx.sym_buffer.buffer.zero_()
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

        # Copy grad_y into the unified buffer's AG local_x view (reused across fwd/bwd)
        ctx.sym_buffer.ag_x[:local_m, :local_N].copy_(grad_y)

        # AG + GEMM: grad_attn = AG(grad_y) @ Wo_r_local
        #   b for bf16_ag_gemm_nt = [N, K] = [local_hidden, dim] = weight.t()
        weight_t = weight.t().contiguous()  # [local_hidden, dim]
        grad_attn = torch.empty((full_m, local_hidden), dtype=torch.bfloat16, device=grad_y.device)
        bf16_ag_gemm_nt(grad_attn, weight_t, ctx.sym_buffer, local_m)

        # Weight grad: reuse gathered grad_y from the unified buffer's AG slots_x
        # (already populated by the AG+GEMM comm kernel).
        # grad_W = grad_y_full.T @ attn  = [dim, full_m] @ [full_m, local_hidden] = [dim, local_hidden]
        # To avoid matmul internal copy of grad_y_full.T (335MB), use:
        #   grad_W.T = attn.T @ grad_y_full  = [local_hidden, full_m] @ [full_m, dim] = [local_hidden, dim]
        #   then grad_W = (attn.T @ grad_y_full).T
        grad_y_full = ctx.sym_buffer.ag_slots_x[:sp, :local_m, :local_N].reshape(full_m, local_N)
        grad_weight = torch.matmul(attn.t(), grad_y_full).t().contiguous()

        return grad_attn, grad_weight, None, None


def gemm_rs(attn, weight, sym_buffer, layout_info):
    return GemmRSFunction.apply(attn, weight, sym_buffer, layout_info)
