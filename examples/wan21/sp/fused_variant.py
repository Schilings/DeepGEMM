"""Fused Variant Ulysses: Q/K/V separate → A2A → attn → Wo GEMM+RS (column-split Wo).

Wo column-split (in Y=XW, W row-split): Wo_i [dim, local_hidden] per rank.
Attention output [full_m, local_hidden] @ Wo_i^T → [full_m, dim] partial → RS → [local_m, dim].

POST uses fused GEMM+RS kernel (bf16_gemm_rs_nt) — no materialized y_partial.
POST backward uses fused AG+GEMM kernel (bf16_ag_gemm_nt) — no materialized grad_y_full.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from deep_gemm import get_unified_symm_buffer

from .base import UlyssesBase
from .serial import NCCLAllToAll
from ..autograd_ops import gemm_rs


class FusedVariantUlysses(UlyssesBase):
    """Q/K/V separate + Wo column-split + fused GEMM+RS (Ulysses variant)."""

    def __init__(self, config, sp_config):
        super().__init__(config, sp_config)

    def _build_weights(self):
        """Standard [Q_all, K_all, V_all] + Wo column-split (for GEMM+RS)."""
        Wq = self.model.q.weight
        Wk = self.model.k.weight
        Wv = self.model.v.weight
        self.Wqkv = nn.Parameter(torch.cat([Wq, Wk, Wv], dim=0).clone(), requires_grad=True)
        self.Wqkv_t = nn.Parameter(self.Wqkv.data.t().contiguous(), requires_grad=True)
        # Wo column-split: Wo_i = Wo[:, rank*local_hidden:(rank+1)*local_hidden] = [dim, local_hidden]
        local_hidden = self.local_nh * self.head_dim
        rank = self.group.rank()
        Wo = self.model.o.weight  # [dim, dim] = [out, in]
        Wo_i = Wo[:, rank * local_hidden:(rank + 1) * local_hidden].contiguous()  # [dim, local_hidden]
        self.Wo_r_local = nn.Parameter(Wo_i.clone(), requires_grad=True)
        self._wo_sharded = True  # Wo grad is local (no FSDP2 sync)

    def _create_buffers(self):
        """Create ONE unified symmetric buffer shared by GEMM+RS (fwd) and AG+GEMM (bwd).

        Uses `deep_gemm.get_unified_symm_buffer`: a single fixed-size symmetric
        allocation sized to the largest operator's needs. GEMM+RS (fwd) and
        AG+GEMM (bwd) reinterpret the same physical data region via dual views
        (RS `partial` vs AG `ag_x`/`ag_slots_x`). They never run concurrently
        (fwd pass fully completes before bwd), so the one buffer is safely
        reused across all layers and both passes — no extra overhead or memory.
        See deep_gemm/unified_buffer/__init__.py and docs/SYM_BUF_SHARING_ANALYSIS.md.
        """
        dim = self.cfg.dim
        # num_max_tokens_per_rank = bs * (seq // sp) == self.local_m, so the
        # buffer is sized exactly for the gemm_rs call's token-per-rank.
        self.sym_post = get_unified_symm_buffer(
            self.group, self.bs, self.seq, dim,
            out_dtype=torch.bfloat16,
        )

    def share_buffers_from(self, other):
        """Reuse the single unified buffer from another strategy instance (multi-layer)."""
        self.sym_post = other.sym_post

    def destroy_buffers(self):
        if self.sym_post is not None:
            self.sym_post.destroy()
        self.sym_post = None

    def _pre_forward(self, x_local, llseq):
        """PRE: GEMM → split Q/K/V → norm → A2A scatter heads → gather seq."""
        dim = self.cfg.dim
        sp = self.sp_size
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        hd = self.head_dim
        bias = torch.cat([self.model.q.bias, self.model.k.bias, self.model.v.bias])
        d = torch.matmul(x_local, self.Wqkv.t()) + bias
        q = self.model.norm_q(d[:, :dim]).view(lbs, llseq, sp, self.local_nh, hd)
        k = self.model.norm_k(d[:, dim:2*dim]).view(lbs, llseq, sp, self.local_nh, hd)
        v = d[:, 2*dim:].view(lbs, llseq, sp, self.local_nh, hd)
        def a2a_scatter(t):
            send = t.permute(2, 0, 1, 3, 4).contiguous()
            recv = NCCLAllToAll.apply(send, self.group)
            return recv.permute(1, 2, 0, 3, 4).reshape(lbs, lseq, self.local_nh, hd)
        q = a2a_scatter(q)
        k = a2a_scatter(k)
        v = a2a_scatter(v)
        return torch.cat([q, k, v], dim=2).reshape(lbs, lseq, -1)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: Wo column-split GEMM + RS (fused) + bias.

        o [lbs, lseq, local_nh, hd] → flatten [full_m, local_hidden]
        → fused GEMM+RS: attn @ Wo_r_local.t() → RS → y [local_m, dim]
        → + bias
        """
        full_m = lbs * lseq  # full seq
        local_hidden = self.local_hidden
        # Flatten attention output to [full_m, local_hidden]
        attn_local = o.reshape(full_m, local_hidden).contiguous()

        # Build layout_info for GemmRSFunction
        layout_info = {
            'local_m': self.local_m,       # tokens per rank (output rows)
            'local_N': self.cfg.dim,       # dim (output cols)
            'full_m': full_m,              # full tokens (input rows)
            'local_hidden': local_hidden,  # K dim of GEMM
            'sp_size': self.sp_size,
            'group': self.group,
        }

        # Fused GEMM + RS: y = RS(attn @ Wo_r.t()) → [local_m, dim]
        # Uses the single unified buffer (fwd GEMM+RS view).
        y = gemm_rs(attn_local, self.Wo_r_local, self.sym_post, layout_info)
        # Add bias (after RS, each rank has local_m tokens)
        return y + self.model.o.bias
