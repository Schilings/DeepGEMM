"""Ulysses Variant: PRE = GEMM+A2A-transpose, POST = Wo row-split + GEMM+RS.

Variant idea: keep sequence splitting + pre-attn A2A, but replace post-attn A2A-transpose+Wo
with Wo row-split + GEMM+RS. This forms a "weight-splitting Ulysses" where:
  - Wo is split along N (output) dimension: each rank owns Wo_r = Wo[r*N/sp:(r+1)*N/sp, :]
  - Wo_r is further split along hidden dim: Wo_r_local = Wo_r[:, r*hidden/sp:(r+1)*hidden/sp]
  - Each rank computes: partial = attn_local @ Wo_r_local^T, then GEMM+RS reduces+scatters
  - Output y is [lm, N/sp] (N-sharded per rank)

POST backward = All-Gather + GEMM (inverse of GEMM+RS):
  grad_y [lm, N/sp] → all-gather → grad_y_full [bs*seq, N/sp] → grad_attn = grad_y_full @ Wo_r_local
"""

import torch
import torch.distributed as dist

from .base import UlyssesSPBase
from ..model import build_wqkv_rankmajor
from ..rope import rope_apply


class UlyssesVariantAttention(UlyssesSPBase):
    """Ulysses variant with GEMM+RS for POST (weight-splitting Ulysses)."""

    def _create_symm_buffers(self):
        import deep_gemm
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.gemm_rs import get_symm_buffer_for_gemm_rs

        n_qkv = self.config.n_qkv
        bs, seq = self.bs, self.seq
        local_m = bs * (seq // self.sp_size)
        local_N = self.config.dim // self.sp_size  # N is sharded

        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, bs, seq, n_qkv)
        # GEMM+RS: num_tokens_per_rank=local_m, hidden=local_N
        self.sym_gemm_rs = get_symm_buffer_for_gemm_rs(
            self.group, local_m, local_N)

        # Wo row-split per rank: Wo_r_local [local_N, local_hidden]
        Wo = self.model.o_proj.weight  # [dim, dim]
        rank = self.group.rank()
        self.Wo_r_local = Wo[
            rank * local_N:(rank + 1) * local_N,
            rank * self.local_hidden:(rank + 1) * self.local_hidden
        ].contiguous()
        self.Wo_r_local_t = self.Wo_r_local.t().contiguous()

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'):
            self.sym_pre.destroy()
        if hasattr(self, 'sym_gemm_rs'):
            self.sym_gemm_rs.handle = None
            self.sym_gemm_rs.buffer = None
            self.sym_gemm_rs.group = None

    def _post_forward(self, o, lbs, lseq, llseq, grid_sizes, **kw):
        """POST: Wo row-split + GEMM+RS → y [lm, local_N]."""
        lm = lbs * llseq if 'lbs' in kw else self.local_m
        lm = kw.get('lm', self.bs * llseq)
        local_N = self.config.dim // self.sp_size
        local_hidden = self.local_hidden

        if self.use_fused:
            from deep_gemm.gemm_rs import bf16_gemm_rs_nt
            attn_local = o.reshape(self.bs * self.seq, local_hidden).contiguous()
            y = torch.empty((lm, local_N), dtype=torch.bfloat16, device=o.device)
            bf16_gemm_rs_nt(y, attn_local, self.Wo_r_local, self.sym_gemm_rs, lm)
        else:
            # torch: matmul + reduce_scatter
            attn_local = o.reshape(lbs * lseq, local_hidden).contiguous()
            partial = torch.matmul(attn_local, self.Wo_r_local_t)  # [bs*seq, local_N]
            partial_chunks = partial.view(self.sp_size, lm, local_N)
            y = torch.empty((lm, local_N), dtype=torch.bfloat16, device=o.device)
            dist.reduce_scatter_tensor(y, partial_chunks.contiguous(), group=self.group)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid_sizes, **kw):
        """POST backward: all-gather grad_y → GEMM → grad_attn + grad_Wo_r_local."""
        sp = self.sp_size
        local_N = self.config.dim // sp
        local_hidden = self.local_hidden

        # All-gather grad_y → grad_y_full [bs*seq, local_N]
        grad_y_full = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(grad_y_full, grad_y, group=self.group)
        grad_y_cat = torch.cat(grad_y_full, dim=0)  # [bs*seq, local_N]

        # Recompute attn_local from cache
        o = cache  # [bs, seq, local_nh, hd]
        attn_local = o.reshape(self.bs * self.seq, local_hidden).contiguous()

        # grad_attn_local = grad_y_full @ Wo_r_local → [bs*seq, local_hidden]
        grad_attn_local = torch.matmul(grad_y_cat, self.Wo_r_local)
        grad_Wo_r_local = torch.matmul(grad_y_cat.t(), attn_local)  # [local_N, local_hidden]

        # grad_attn → [bs, seq, local_nh, hd]
        grad_attn = grad_attn_local.reshape(self.bs, self.seq, self.local_nh, self.head_dim)
        return grad_attn, grad_Wo_r_local

    def get_cache(self, o):
        return o

    def backward(self, grad_y, x_local, grid_sizes, cache=None, llseq=None):
        """Variant backward: all-gather approach (output is N-sharded)."""
        assert self._shape_set
        if llseq is None:
            llseq = self.local_seq
        sp = self.sp_size
        ln = self.local_n
        hd = self.head_dim
        hidden = self.config.dim
        local_N = hidden // sp  # variant output is N-sharded
        lbs = self.bs if self.layout == 'BSHD' else 1
        lseq = self.seq if self.layout == 'BSHD' else self.bs * self.seq
        lm = lbs * llseq

        # All-gather X_local across ranks → full [bs*seq, dim]
        X_list = [torch.empty_like(x_local) for _ in range(sp)]
        dist.all_gather(X_list, x_local, group=self.group)
        X_full = torch.cat(X_list, dim=0)  # [bs*seq, dim]

        # All-gather grad_y across N-shards → full [bs*seq, dim]
        # grad_y: [lm, local_N] per rank. All-gather along N → [lm, local_N*sp] = [lm, dim]
        # But all_gather gives [sp, lm, local_N] → need to permute to [lm, sp*local_N]
        gy_list = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(gy_list, grad_y, group=self.group)
        # Each gy_list[r] = [lm, local_N] for rank r's N-shard
        grad_y_full = torch.cat(gy_list, dim=1)  # [lm, sp*local_N] = [lm, dim] -- but only local seq!
        # Need full seq: all-gather grad_y along seq too... 
        # Actually grad_y is [lm, local_N] (local seq, local N). 
        # We need [bs*seq, dim] for the full backward.
        # First all-gather along seq → [sp*lm, local_N], then along N → [sp*lm, sp*local_N]
        gy_seq_list = [torch.empty_like(grad_y) for _ in range(sp)]
        dist.all_gather(gy_seq_list, grad_y, group=self.group)
        grad_y_seq = torch.cat(gy_seq_list, dim=0)  # [bs*seq, local_N]
        # Now all-gather along N (across ranks, each has different N-shard)
        # Actually all ranks already have the same grad_y_seq after the first all-gather!
        # We need to gather N-shards. But grad_y_seq has only local_N per rank.
        # Use all_gather on the N dimension:
        gy_n_list = [torch.empty_like(grad_y_seq) for _ in range(sp)]
        dist.all_gather(gy_n_list, grad_y_seq, group=self.group)
        grad_y_full = torch.cat(gy_n_list, dim=1)  # [bs*seq, dim]

        # Now compute full backward with autograd
        X_g = X_full.detach().clone().requires_grad_(True)
        Wq_g = self.model.q_proj.weight.detach().clone().requires_grad_(True)
        Wk_g = self.model.k_proj.weight.detach().clone().requires_grad_(True)
        Wv_g = self.model.v_proj.weight.detach().clone().requires_grad_(True)
        Wo_g = self.model.o_proj.weight.detach().clone().requires_grad_(True)
        nheads = self.config.num_heads

        q = self.model.norm_q(torch.matmul(X_g, Wq_g.t())).view(lbs, lseq, nheads, hd)
        k = self.model.norm_k(torch.matmul(X_g, Wk_g.t())).view(lbs, lseq, nheads, hd)
        v = torch.matmul(X_g, Wv_g.t()).view(lbs, lseq, nheads, hd)
        q = rope_apply(q, grid_sizes, self.model.freqs)
        k = rope_apply(k, grid_sizes, self.model.freqs)

        from flash_attn.cute import flash_attn_func
        o = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=self.config.causal)
        o = o[0] if isinstance(o, tuple) else o
        y = torch.matmul(o.reshape(lbs * lseq, hidden), Wo_g.t())

        grad_X_full, grad_Wq, grad_Wk, grad_Wv, grad_Wo = torch.autograd.grad(
            y, [X_g, Wq_g, Wk_g, Wv_g, Wo_g], grad_y_full)

        grad_Wqkv = build_wqkv_rankmajor(grad_Wq, grad_Wk, grad_Wv, sp, self.local_nh, hd)
        rank = self.group.rank()
        grad_X = grad_X_full[rank * lm:(rank + 1) * lm]
        return grad_X, grad_Wqkv, grad_Wo
