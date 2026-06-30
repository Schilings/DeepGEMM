"""Standard Ulysses SP: PRE = GEMM+A2A-transpose, POST = A2A-transpose+GEMM.

POST flow:
  attn_out [bs, seq, local_nh, hd] → BHSD → A2A-transpose → gathered [lm, hidden] → Wo GEMM → y [lm, dim]

POST backward:
  grad_y → grad_gathered = grad_y @ Wo → A2A-inverse → grad_attn
  grad_Wo = grad_y^T @ gathered
"""

import torch
import torch.distributed as dist

from .base import UlyssesSPBase


class UlyssesStandardAttention(UlyssesSPBase):
    """Standard Ulysses SP attention with fused A2A-transpose + GEMM."""

    def _create_symm_buffers(self):
        import deep_gemm
        from deep_gemm import get_symm_buffer_for_gemm_a2a_transpose
        from deep_gemm.a2a_transpose_gemm import get_symm_buffer_for_a2a_transpose_gemm

        n_qkv = self.config.n_qkv
        bs, seq, nheads, hd = self.bs, self.seq, self.config.num_heads, self.head_dim
        self.sym_pre = get_symm_buffer_for_gemm_a2a_transpose(
            self.group, bs, seq, n_qkv)
        self.sym_post = get_symm_buffer_for_a2a_transpose_gemm(
            self.group, bs, nheads, seq, hd)

    def destroy_buffers(self):
        if hasattr(self, 'sym_pre'):
            self.sym_pre.destroy()
        if hasattr(self, 'sym_post'):
            self.sym_post.destroy()

    def _post_forward(self, o, lbs, lseq, llseq, grid_sizes, **kw):
        """POST: A2A-transpose + Wo GEMM (fused)."""
        lm = lbs * llseq
        N = self.config.dim

        if self.use_fused:
            from deep_gemm.a2a_transpose_gemm import bf16_a2a_transpose_gemm_nt_fused
            # o: [bs, seq, local_nh, hd] → BHSD [bs, local_nh, seq, hd] for POST
            self.sym_post.x.copy_(o.transpose(1, 2).contiguous())
            y = torch.empty((lm, N), dtype=torch.bfloat16, device=o.device)
            bf16_a2a_transpose_gemm_nt_fused(y, self.Wo, self.sym_post)
        else:
            # torch: A2A + matmul
            sp = self.sp_size
            ln = self.local_n
            hd = self.head_dim
            x_bhsd = o.transpose(1, 2)  # [bs, local_nh, seq, hd]
            send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, self.config.dim)
            y = torch.matmul(gathered, self.Wo_t)
        return y

    def _post_backward(self, grad_y, cache, lbs, lseq, llseq, lm, grid_sizes, **kw):
        """POST backward: grad_y → grad_attn + grad_Wo.

        Returns: grad_attn [lbs, lseq, local_nh, hd], grad_Wo [dim, dim]
        """
        Wo = self.Wo
        Wo_t = self.Wo_t
        sp = self.sp_size
        ln = self.local_n
        hd = self.head_dim
        hidden = self.config.dim

        # grad_gathered = grad_y @ Wo → [lm, hidden]
        grad_gathered = torch.matmul(grad_y, Wo)

        # Recompute gathered from cache (attn output)
        o = cache  # [lbs, lseq, local_nh, hd]
        x_bhsd = o.transpose(1, 2)  # [lbs, local_nh, lseq, hd]
        send = x_bhsd.view(lbs, self.local_nh, sp, llseq, hd).permute(2, 0, 3, 1, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(lm, hidden)

        # grad_Wo = grad_y^T @ gathered
        grad_Wo = torch.matmul(grad_y.t(), gathered)

        # grad_attn = A2A-inverse(grad_gathered)
        # grad_gathered [lm, hidden] → [lbs, llseq, sp, local_nh, hd] → permute → A2A → reshape
        send_bwd = grad_gathered.view(lbs, llseq, sp, self.local_nh, hd).permute(2, 0, 1, 3, 4).contiguous()
        recv_bwd = torch.empty_like(send_bwd)
        dist.all_to_all_single(recv_bwd, send_bwd, group=self.group)
        grad_attn = recv_bwd.permute(1, 3, 2, 4, 0).reshape(lbs, lseq, self.local_nh, hd)

        return grad_attn, grad_Wo

    def get_cache(self, o):
        """Return cache for backward (the attention output o)."""
        return o
