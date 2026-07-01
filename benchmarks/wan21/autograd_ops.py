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

Weight gradients: PyTorch autograd computes them automatically from the GEMM's
input gradient chain (since weights are nn.Parameter leaf nodes with requires_grad).
The fused kernel only computes the activation gradient; the weight grad is a
standard matmul that autograd derives from the forward GEMM's backward.

NOTE on weight grads: The fused comm-GEMM kernels write the output directly (not
via a standard GEMM that autograd can differentiate). So for weight grads we must
manually compute them in the Function's backward (standard matmul, no overlap).
The fused kernel's epilogue communication only applies to the activation path.
"""

import torch
import torch.distributed as dist


# ════════════════════════════════════════════════════════════════════════════
# 1. GEMM + A2A-transpose (PRE-attn forward)
#    Forward:  x_local[local_m, K] @ Wqkv^T[N, K] → A2A-scatter(heads) → qkv[bs, seq, local_n]
#    Backward: grad_qkv → A2A-gather(heads) → grad_local[local_m, N] → grad_X = grad_local @ Wqkv
#              grad_Wqkv = grad_local^T @ x_local  (weight grad, standard matmul)
# ════════════════════════════════════════════════════════════════════════════

class GemmA2ATransposeFunction(torch.autograd.Function):
    """GEMM + A2A-transpose (Ulysses pre-attn). Dual backward = A2A + GEMM."""

    @staticmethod
    def forward(ctx, x_local, weight, sym_buffer, local_seq, n_qkv, sp_size, group, layout_info):
        """x_local [local_m, K] @ weight[N, K]^T → A2A-scatter → qkv [bs, seq, local_n].

        Args:
            x_local: [local_m, K] bf16, requires_grad=True
            weight: [N, K] bf16 (Wqkv, NT layout), nn.Parameter
            sym_buffer: GemmA2ATransposeSymmBuffer
            local_seq: this rank's seq length
            n_qkv: full N = 3 * nheads * head_dim
            sp_size: SP degree
            group: process group
            layout_info: dict with bs, lseq, lbs, llseq for backward reshaping
        """
        from deep_gemm import bf16_gemm_a2a_transpose_nt
        ctx.sym_buffer = sym_buffer
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
        """grad_qkv → A2A-gather(heads) → grad_local → grad_X = grad_local @ Wqkv, grad_Wqkv = grad_local^T @ x."""
        from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt
        x_local, weight = ctx.saved_tensors
        li = ctx.layout_info
        lbs = li['lbs']; lseq = li['lseq']; llseq = li['llseq']; lm = li['lm']
        local_nh_qkv = 3 * (li['nheads'] // ctx.sp_size)
        head_dim = li['head_dim']
        # grad_qkv [lbs, lseq, local_nqkv] → BHSD [lbs, 3*local_nh, lseq, hd]
        gqkv_bhsd = grad_qkv.view(lbs, lseq, local_nh_qkv, head_dim).transpose(1, 2).contiguous()
        ctx.sym_buffer.x.copy_(gqkv_bhsd)  # sym_pre_bwd buffer
        grad_X = torch.empty((lm, li['hidden']), dtype=torch.bfloat16, device=grad_qkv.device)
        # weight is [N, K] (Wqkv), we need [K, N]^T for NT → use weight.t() (Wqkv_t)
        bf16_a2a_transpose_gemm_nt(grad_X, weight.t(), ctx.sym_buffer)
        # Weight grad: grad_Wqkv = grad_local^T @ x_local
        grad_local = ctx.sym_buffer.gathered[:lm, :ctx.n_qkv]
        grad_weight = torch.matmul(grad_local.t(), x_local)
        return grad_X, grad_weight, None, None, None, None, None, None


def gemm_a2a_transpose(x_local, weight, sym_buffer, local_seq, n_qkv, sp_size, group, layout_info):
    """Functional wrapper for GemmA2ATransposeFunction."""
    return GemmA2ATransposeFunction.apply(x_local, weight, sym_buffer, local_seq, n_qkv, sp_size, group, layout_info)


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
        """o [bs, seq, local_nh, hd] → A2A-transpose → gathered → @ weight^T → y [local_m, N].

        Args:
            o: attention output [bs, seq, local_nh, hd] bf16, requires_grad=True
            weight: Wo [N, hidden] bf16, nn.Parameter (NT layout)
            sym_buffer: BF16A2ATransposeGemmSymmBuffer
            layout_info: dict with bs, seq, local_nh, hd, local_m, hidden, sp_size, group
        """
        from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt_fused
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
        """grad_y → grad_y @ Wo → A2A-inv → grad_o. grad_Wo = grad_y^T @ gathered."""
        o, weight = ctx.saved_tensors
        li = ctx.layout_info
        sp = li['sp_size']; hd = li['hd']; hidden = li['hidden']
        lbs = li['lbs']; lseq = li['lseq']; llseq = li['llseq']; lm = li['local_m']
        local_nh = li['local_nh']
        group = li['group']
        # grad_gathered = grad_y @ Wo (standard matmul, autograd-compatible)
        grad_gathered = torch.matmul(grad_y, weight)
        # A2A-inverse: [lm, hidden] → [lbs, llseq, sp, local_nh, hd] → permute → A2A → reshape
        send_bwd = grad_gathered.view(lbs, llseq, sp, local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=group)
        grad_o = recv_bwd.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, local_nh, hd)
        # Weight grad: grad_Wo = grad_y^T @ gathered (need to recompute gathered from o)
        x_bhsd = o.transpose(1, 2)
        send = x_bhsd.view(lbs, local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)
        grad_weight = torch.matmul(grad_y.t(), gathered)
        return grad_o, grad_weight, None, None


def a2a_transpose_gemm(o, weight, sym_buffer, layout_info):
    """Functional wrapper for A2ATransposeGemmFunction."""
    return A2ATransposeGemmFunction.apply(o, weight, sym_buffer, layout_info)


# ════════════════════════════════════════════════════════════════════════════
# 3. GEMM + RS (POST-attn forward, fused_var variant)
#    Forward:  attn[local_m, local_hidden] @ Wo_r^T[local_N, local_hidden] → RS → y[local_m, local_N]
#    Backward: grad_y → AG → grad_y_full[full_m, local_N] → @ Wo_r → grad_attn[full_m, local_hidden]
#              grad_Wo_r = grad_y_full^T @ attn  (weight grad)
# ════════════════════════════════════════════════════════════════════════════

class GemmRSFunction(torch.autograd.Function):
    """GEMM + Reduce-Scatter (Ulysses post-attn variant). Dual backward = AG + GEMM."""

    @staticmethod
    def forward(ctx, attn, weight, sym_buffer, layout_info):
        """attn [local_m, local_hidden] @ weight[local_N, local_hidden]^T → RS → y [local_m, local_N].

        Args:
            attn: [local_m, local_hidden] bf16 (local attention output, flattened), requires_grad=True
            weight: Wo_r_local [local_N, local_hidden] bf16, nn.Parameter (NT layout)
            sym_buffer: GemmRSSymmBuffer
            layout_info: dict with local_m, local_N, local_hidden, full_m, sp_size, group, bs, seq
        """
        from deep_gemm.gemm_rs import bf16_gemm_rs_nt
        ctx.sym_buffer = sym_buffer
        ctx.layout_info = layout_info
        ctx.save_for_backward(attn, weight)
        li = layout_info
        local_m = li['local_m']; local_N = li['local_N']
        y = torch.empty((local_m, local_N), dtype=torch.bfloat16, device=attn.device)
        bf16_gemm_rs_nt(y, attn, weight, sym_buffer, local_m)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """grad_y → AG → grad_y_full → @ Wo_r → grad_attn. grad_Wo_r = grad_y_full^T @ attn."""
        from deep_gemm.ag_gemm import bf16_ag_gemm_nt
        attn, weight = ctx.saved_tensors
        li = ctx.layout_info
        sp = li['sp_size']; local_N = li['local_N']; local_hidden = li['local_hidden']
        full_m = li['full_m']; local_m = li['local_m']; group = li['group']
        # Fused AG+GEMM: grad_y → all-gather → @ Wo_r^T → grad_attn
        ctx.sym_buffer.x[:local_m, :local_N].copy_(grad_y)  # sym_ag_gemm buffer
        grad_attn = torch.empty((full_m, local_hidden), dtype=torch.bfloat16, device=grad_y.device)
        bf16_ag_gemm_nt(grad_attn, weight, ctx.sym_buffer, local_m)
        # Weight grad: grad_Wo_r = grad_y_full^T @ attn (need grad_y_full from all-gather)
        gy_list = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(gy_list, grad_y, group=group)
        grad_y_full = torch.cat(gy_list, dim=0)
        grad_weight = torch.matmul(grad_y_full.t(), attn)
        return grad_attn, grad_weight, None, None


def gemm_rs(attn, weight, sym_buffer, layout_info):
    """Functional wrapper for GemmRSFunction."""
    return GemmRSFunction.apply(attn, weight, sym_buffer, layout_info)
