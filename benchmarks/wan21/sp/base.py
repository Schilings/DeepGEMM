"""Abstract base for SP attention strategies.

Defines the interface that all strategies (standard, variant, torch-baseline) implement.
This interface is designed to be framework-friendly:
  - forward(x_local, grid_sizes) → y_local
  - backward(grad_y, cache) → grad_X_local, weight_grads (auto-synced via FSDP2 or manual)

The strategy wraps WanSelfAttention (model layer) and adds:
  - PRE: GEMM + A2A-transpose (sequence scatter → head gather)
  - POST: communication + Wo GEMM (strategy-specific)

Subclasses implement _pre_forward, _post_forward, _pre_backward, _post_backward.
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist

from ..config import Wan21Config, SPConfig
from ..model import WanSelfAttention, build_wqkv_rankmajor
from ..rope import build_wan21_freqs, rope_apply, rope_inverse
from ..norm import WanRMSNorm


class UlyssesSPBase(nn.Module):
    """Base class for Ulysses SP attention strategies.

    Holds:
      - WanSelfAttention model (the WHAT)
      - SP config (the HOW: sp_size, layout, fused ops, post strategy)
      - Symm buffers (created lazily per shape)

    Subclasses override:
      _post_forward(o_local, ...)  → y_local
      _post_backward(grad_y, ...)  → grad_attn_local, grad_Wo
    """

    def __init__(self, config: Wan21Config, sp_config: SPConfig):
        super().__init__()
        self.config = config
        self.sp = sp_config
        self.model = WanSelfAttention(config)
        self.scale = config.scale
        self.sp_size = sp_config.sp_size
        self.group = sp_config.group
        self.layout = sp_config.layout
        self.use_fused = sp_config.use_fused_ops

        # Derived dims (set per-shape in setup_shape)
        self._shape_set = False
        self.local_nh = None
        self.local_n = None
        self.local_hidden = None
        self.local_nqkv = None
        self.bs = None
        self.seq = None
        self.local_seq = None
        self.local_m = None

        # Fused QKV weight (rank-major, built from model's q/k/v projections)
        self._Wqkv_built = False

    def setup_shape(self, bs, seq, nheads, head_dim):
        """Called before forward to set shape-dependent params + create symm buffers."""
        sp = self.sp_size
        assert nheads % sp == 0, f"nheads {nheads} not divisible by sp {sp}"
        assert seq % sp == 0, f"seq {seq} not divisible by sp {sp}"
        local_seq = seq // sp
        assert local_seq % 128 == 0, f"local_seq {local_seq} not multiple of 128"

        self.bs = bs
        self.seq = seq
        self.local_nh = nheads // sp
        self.head_dim = head_dim
        self.local_n = self.local_nh * head_dim
        self.local_hidden = self.local_n  # = hidden / sp
        self.local_nqkv = 3 * self.local_n  # = 3 * local_nh * hd
        self.local_seq = local_seq
        self.local_m = bs * local_seq
        self._shape_set = True

        self._build_fused_weights()
        self._create_symm_buffers()

    def _build_fused_weights(self):
        """Build rank-major Wqkv from model's q/k/v projection weights."""
        Wq = self.model.q_proj.weight  # [dim, dim]
        Wk = self.model.k_proj.weight
        Wv = self.model.v_proj.weight
        Wo = self.model.o_proj.weight  # [dim, dim]
        sp = self.sp_size
        lnh = self.local_nh
        hd = self.head_dim

        self.Wqkv = build_wqkv_rankmajor(Wq, Wk, Wv, sp, lnh, hd)  # [3*dim, dim]
        self.Wqkv_t = self.Wqkv.t().contiguous()
        self.Wo = Wo
        self.Wo_t = Wo.t().contiguous()
        self._Wqkv_built = True

    def _create_symm_buffers(self):
        """Override in subclass to create strategy-specific symm buffers."""
        pass

    def destroy_buffers(self):
        """Override in subclass to destroy symm buffers."""
        pass

    # ── Shared PRE forward (GEMM + A2A-transpose) ──────────────────────────

    def _pre_forward(self, X_local, llseq):
        """PRE: GEMM(QKV) + A2A-transpose → qkv_local [bs, seq, local_nqkv].

        Uses fused bf16_gemm_a2a_transpose_nt if use_fused, else torch matmul + NCCL A2A.
        """
        if self.use_fused:
            import deep_gemm
            qkv = deep_gemm.bf16_gemm_a2a_transpose_nt(
                X_local, self.Wqkv, self.sym_pre, llseq)
        else:
            lbs = self.bs if self.layout == 'BSHD' else 1
            lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
            d = torch.matmul(X_local, self.Wqkv_t).view(lbs, llseq, self.sp_size, self.local_nqkv)
            send = d.permute(2, 0, 1, 3).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            qkv = recv.permute(1, 2, 0, 3).reshape(lbs, lseq, self.local_nqkv)
        return qkv

    # ── Shared attention forward (QK norm + RoPE + FA4) ──────────────────

    def _attn_forward(self, qkv, grid_sizes, lbs, lseq):
        """Split qkv → q/k/v, apply norm + RoPE, run FA4. Returns o [lbs, lseq, local_nh, hd] + cache."""
        ln = self.local_n
        q = qkv[:, :, :ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()
        k = qkv[:, :, ln:2*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()
        v = qkv[:, :, 2*ln:3*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous()

        # QK norm (preserves bf16)
        q = self.model.norm_q(q.reshape(-1, self.config.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        k = self.model.norm_k(k.reshape(-1, self.config.dim)).view(lbs, lseq, self.local_nh, self.head_dim)

        # RoPE
        q = rope_apply(q, grid_sizes, self.model.freqs)
        k = rope_apply(k, grid_sizes, self.model.freqs)

        # FA4 attention
        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=self.config.causal)
        o = o[0] if isinstance(o, tuple) else o
        return o

    # ── Shared attention backward (FA4 bwd via autograd) ──────────────────

    def _attn_backward(self, grad_attn, X_local, Wqkv_t, grid_sizes, lbs, llseq, lseq):
        """Recompute q/k/v (with requires_grad), run FA4 backward via autograd.

        Returns: grad_q_pre_norm, grad_k_pre_norm, grad_v (all [lbs, lseq, local_nh, hd]).
        """
        local_nqkv = self.local_nqkv
        ln = self.local_n

        # Recompute qkv (torch path — A2A doesn't support autograd)
        qkv_local = torch.matmul(X_local, Wqkv_t).view(lbs, llseq, self.sp_size, local_nqkv)
        send = qkv_local.permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        qkv = recv.permute(1, 2, 0, 3).reshape(lbs, lseq, local_nqkv)

        q_leaf = qkv[:, :, :ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)
        k_leaf = qkv[:, :, ln:2*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)
        v_leaf = qkv[:, :, 2*ln:3*ln].view(lbs, lseq, self.local_nh, self.head_dim).contiguous().requires_grad_(True)

        q = self.model.norm_q(q_leaf.reshape(-1, self.config.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        k = self.model.norm_k(k_leaf.reshape(-1, self.config.dim)).view(lbs, lseq, self.local_nh, self.head_dim)
        q = rope_apply(q, grid_sizes, self.model.freqs)
        k = rope_apply(k, grid_sizes, self.model.freqs)

        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v_leaf, softmax_scale=self.scale, causal=self.config.causal)
        o = o[0] if isinstance(o, tuple) else o

        # Get gradients w.r.t. LEAF tensors (pre-norm) — autograd handles norm + RoPE backward
        grad_q, grad_k, grad_v = torch.autograd.grad(o, [q_leaf, k_leaf, v_leaf], grad_attn)

        return grad_q, grad_k, grad_v

    # ── Shared PRE backward (A2A-inv + GEMM bwd) ──────────────────────────

    def _pre_backward(self, grad_qkv, lbs, llseq, lseq, X_local, lm):
        """PRE backward: A2A-inverse (seq-scatter + head-gather) + GEMM backward.

        grad_qkv: [lbs, lseq, local_nqkv] (full seq, local QKV head group)
        Returns: grad_X_local [lm, dim], grad_Wqkv [3*dim, dim]

        Inverse of _pre_forward:
          fwd: [lbs, llseq, sp, local_nqkv] → permute(2,0,1,3) → A2A → permute(1,2,0,3) → reshape
          bwd: reshape → permute(2,0,1,3) → A2A → permute(1,0,2,3) → reshape
          (A2A is self-inverse; the permutes swap which axis is the "rank" axis)
        """
        n_qkv = self.config.n_qkv
        sp = self.sp_size
        local_nqkv = self.local_nqkv

        # [lbs, lseq, local_nqkv] → [lbs, llseq, sp, local_nqkv] → [sp, lbs, llseq, local_nqkv]
        send = grad_qkv.view(lbs, llseq, sp, local_nqkv).permute(2, 0, 1, 3).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        # [sp, lbs, llseq, local_nqkv] → [lbs, llseq, sp, local_nqkv] → [lm, n_qkv]
        grad_local = recv.permute(1, 2, 0, 3).reshape(lm, n_qkv)

        grad_X = torch.matmul(grad_local, self.Wqkv)        # [lm, dim]
        grad_Wqkv = torch.matmul(grad_local.t(), X_local)    # [3*dim, dim]
        return grad_X, grad_Wqkv

    # ── Abstract POST forward/backward ────────────────────────────────────

    def _post_forward(self, o, **kwargs):
        """POST: attention output → y_local. Override in subclass."""
        raise NotImplementedError

    def _post_backward(self, grad_y, cache, **kwargs):
        """POST backward. Override in subclass."""
        raise NotImplementedError

    # ── Full forward ──────────────────────────────────────────────────────

    def forward(self, x_local, grid_sizes, llseq=None):
        """Full forward: x_local [lm, dim] → y_local [lm, ...].

        llseq: local_seq for this layout (BSHD: seq//sp, THD: bs*seq//sp).
        """
        assert self._shape_set, "Call setup_shape() before forward()"
        if llseq is None:
            llseq = self.local_seq
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq

        qkv = self._pre_forward(x_local, llseq)
        o = self._attn_forward(qkv, grid_sizes, lbs, lseq)
        y = self._post_forward(o, lbs=lbs, lseq=lseq, llseq=llseq, grid_sizes=grid_sizes)
        return y

    def backward(self, grad_y, x_local, grid_sizes, cache=None, llseq=None):
        """Full backward — all-gather approach for correctness.

        Strategy: all-gather X_local + grad_y across ranks → each rank has full data →
        compute gradients locally → split grad_X to local shard. Weight grads are partial,
        need all-reduce (FSDP2 or manual).

        This is the simplest correct approach. A2A-inverse is mathematically equivalent
        but error-prone to implement. For perf, the A2A-inverse avoids the all-gather
        bandwidth cost, but for correctness verification this is the reference.
        """
        assert self._shape_set
        if llseq is None:
            llseq = self.local_seq
        sp = self.sp_size
        ln = self.local_n
        hd = self.head_dim
        hidden = self.config.dim
        local_nqkv = self.local_nqkv
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        lm = lbs * llseq

        # All-gather X_local and grad_y → each rank has full [bs*seq, ...]
        X_full_list = [torch.empty_like(x_local) for _ in range(sp)]
        dist.all_gather(X_full_list, x_local, group=self.group)
        X_full_local = torch.cat(X_full_list, dim=0)  # [bs*seq, dim]

        # grad_y: each rank has [lm, dim] (standard) or [lm, local_N] (variant)
        # For standard: all-gather along seq → [bs*seq, dim]
        if self.sp.post_strategy == 'a2a_gemm':
            grad_y_list = [torch.empty_like(grad_y) for _ in range(sp)]
            dist.all_gather(grad_y_list, grad_y, group=self.group)
            grad_y_full = torch.cat(grad_y_list, dim=0)  # [bs*seq, dim]
        else:
            # variant: output is N-sharded, need different handling
            grad_y_list = [torch.empty_like(grad_y) for _ in range(sp)]
            dist.all_gather(grad_y_list, grad_y, group=self.group)
            grad_y_full = torch.cat(grad_y_list, dim=0)  # [bs*seq, local_N*sp]?

        # Now compute gradients with autograd (single-GPU style, but on full data)
        X_g = X_full_local.detach().clone().requires_grad_(True)
        Wq_g = self.model.q_proj.weight.detach().clone().requires_grad_(True)
        Wk_g = self.model.k_proj.weight.detach().clone().requires_grad_(True)
        Wv_g = self.model.v_proj.weight.detach().clone().requires_grad_(True)
        Wo_g = self.Wo.detach().clone().requires_grad_(True)
        nheads = self.config.num_heads

        # QKV projection (separate, not rank-major — for full-data backward)
        q = self.model.norm_q(torch.matmul(X_g, Wq_g.t())).view(lbs, lseq, nheads, hd)
        k = self.model.norm_k(torch.matmul(X_g, Wk_g.t())).view(lbs, lseq, nheads, hd)
        v = torch.matmul(X_g, Wv_g.t()).view(lbs, lseq, nheads, hd)

        # RoPE
        q = rope_apply(q, grid_sizes, self.model.freqs)
        k = rope_apply(k, grid_sizes, self.model.freqs)

        # FA4 attention
        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=self.config.causal)
        o = o[0] if isinstance(o, tuple) else o

        # Wo projection
        y = torch.matmul(o.reshape(lbs * lseq, hidden), Wo_g.t())  # [bs*seq, dim]

        # Backward
        grad_X_full, grad_Wq, grad_Wk, grad_Wv, grad_Wo = torch.autograd.grad(
            y, [X_g, Wq_g, Wk_g, Wv_g, Wo_g], grad_y_full)

        # Assemble grad_Wqkv (rank-major, matching forward Wqkv)
        grad_Wqkv = build_wqkv_rankmajor(
            grad_Wq, grad_Wk, grad_Wv, sp, self.local_nh, hd)

        # Split grad_X to this rank's local seq shard
        rank = self.group.rank()
        grad_X = grad_X_full[rank * lm:(rank + 1) * lm]

        return grad_X, grad_Wqkv, grad_Wo
