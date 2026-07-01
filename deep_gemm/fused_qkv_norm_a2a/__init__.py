"""
Fused QKV GEMM + RMSNorm + A2A-transpose (Ulysses SP pre-attn).

Combines:
  1. Q/K/V projection GEMM (with optional bias)
  2. RMSNorm on Q and/or K (optional, before A2A scatter)
  3. A2A-transpose scatter by head groups (GQA-aware)

When norm is disabled, this degenerates to a biased GEMM + A2A-transpose.

Usage (Ulysses SP, seq-sharded input):
  sym = get_symm_buffer_for_fused_qkv_norm_a2a(group, bs, seq, q_nheads, kv_nheads, head_dim)
  out = bf16_fused_qkv_norm_a2a_transpose_nt(
      x_local, Wqkv, sym, local_seq,
      q_nheads, kv_nheads, head_dim,
      eps=1e-6,
      norm_q_weight=norm_q,  # or None to skip
      norm_k_weight=norm_k,  # or None to skip
      bias=bias,             # or None
  )
  # out: [bs, seq, local_n_total] bf16, ready for RoPE + attention
"""

import torch
from typing import Optional

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load fused QKV+Norm+A2A kernels: {exception}')

from .. import _C
from ..utils.math import align


class FusedQKVNormA2ASymmBuffer:
    """Symmetric buffer for the Fused QKV GEMM + RMSNorm + A2A-transpose operator.

    The buffer holds ONLY the scattered OUTPUT region [bs, seq, local_n_total] (plus a 32B
    barrier header). local_n_total = local_q_n + 2*local_kv_n (GQA-aware).
    """
    def __init__(self, group: dist.ProcessGroup,
                 bs: int,
                 seq: int,
                 q_nheads: int,
                 kv_nheads: int,
                 head_dim: int,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.group = group
        self.world_size = group.size()
        self.bs = bs
        self.seq = seq
        self.q_nheads = q_nheads
        self.kv_nheads = kv_nheads
        self.head_dim = head_dim
        self.out_dtype = out_dtype

        self.local_q_nheads = q_nheads // self.world_size
        self.local_kv_nheads = kv_nheads // self.world_size
        self.local_q_n = self.local_q_nheads * head_dim
        self.local_kv_n = self.local_kv_nheads * head_dim
        self.local_n = self.local_q_n + 2 * self.local_kv_n  # local_n_total

        # Buffer size: barrier(32B) + output(bs*seq*local_n*elem_size)
        elem_size = 4 if out_dtype == torch.float32 else 2
        num_bytes = 32 + bs * seq * self.local_n * elem_size
        num_bytes = (num_bytes + 15) // 16 * 16  # align to 16

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    @property
    def out(self) -> torch.Tensor:
        """Output view [bs, seq, local_n] of the symmetric buffer."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        out_offset = 32  # kNumBarrierSignalBytes
        out_ptr = self.buffer.data_ptr() + out_offset
        return torch.frombuffer(
            self.buffer.cpu(),  # need to access via device pointer
            dtype=self.out_dtype,
            count=self.bs * self.seq * self.local_n,
        ).reshape(self.bs, self.seq, self.local_n) if False else \
            torch.empty((self.bs, self.seq, self.local_n), dtype=self.out_dtype, device='cuda').set_(
                self.buffer, out_offset, (self.bs, self.seq, self.local_n))

    def get_out_view(self) -> torch.Tensor:
        """Get a [bs, seq, local_n] view of the output region in the symmetric buffer."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        out_offset = 32
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype,
            device=self.buffer.device
        ).set_(self.buffer, out_offset // elem_size, (self.bs, self.seq, self.local_n))

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def get_symm_buffer_for_fused_qkv_norm_a2a(group: dist.ProcessGroup,
                                            bs: int,
                                            seq: int,
                                            q_nheads: int,
                                            kv_nheads: int,
                                            head_dim: int,
                                            out_dtype: torch.dtype = torch.bfloat16
                                            ) -> FusedQKVNormA2ASymmBuffer:
    """Create a symmetric buffer for the Fused QKV+Norm+A2A operator."""
    assert q_nheads % group.size() == 0, 'q_nheads must be divisible by sp_size'
    assert kv_nheads % group.size() == 0, 'kv_nheads must be divisible by sp_size'
    assert seq % group.size() == 0, 'seq must be divisible by sp_size'
    return FusedQKVNormA2ASymmBuffer(group, bs, seq, q_nheads, kv_nheads, head_dim, out_dtype)


def bf16_fused_qkv_norm_a2a_transpose_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    sym_buffer: FusedQKVNormA2ASymmBuffer,
    local_seq: int,
    q_nheads: int,
    kv_nheads: int,
    head_dim: int,
    eps: float = 1e-6,
    norm_q_weight: Optional[torch.Tensor] = None,
    norm_k_weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 Fused QKV GEMM + RMSNorm (optional) + A2A-transpose (Ulysses SP pre-attn).

    Args:
        a: Local activations [bs*local_seq, K], bf16 (seq-sharded, full hidden K)
        b: Projection weights [N_total, K] (NT layout), bf16
           N_total = (q_nheads + 2*kv_nheads) * head_dim
           Layout: [Wq(q_dim, K); Wk(kv_dim, K); Wv(kv_dim, K)]
        sym_buffer: Symmetric buffer from get_symm_buffer_for_fused_qkv_norm_a2a
        local_seq: This rank's local sequence length
        q_nheads: Total Q heads (>= kv_nheads for GQA, == for MHA)
        kv_nheads: Total K/V heads
        head_dim: Head dimension (typically 128)
        eps: RMSNorm epsilon
        norm_q_weight: [q_dim] fp32, RMSNorm weight for Q. None = skip norm on Q.
        norm_k_weight: [kv_dim] fp32, RMSNorm weight for K. None = skip norm on K.
        bias: [N_total] bf16, optional bias added after GEMM

    Returns:
        out: [bs, seq, local_n_total] bf16, where
             local_n_total = local_q_n + 2*local_kv_n
    """
    assert torch.cuda.get_device_capability()[0] == 10, \
        'bf16_fused_qkv_norm_a2a_transpose_nt is for SM100/B-series GPUs'
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16

    num_ranks = sym_buffer.world_size
    bs = sym_buffer.bs
    seq = local_seq * num_ranks
    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim
    local_m = bs * local_seq

    assert a.shape[0] == local_m
    assert b.shape[0] == n_total
    assert a.shape[1] == b.shape[1]  # K matches
    assert local_seq % 128 == 0, 'local_seq must be 128-aligned'

    # ════════════════════════════════════════════════════════════════
    #  v2b: Norm-deferred approach
    #  1. GEMM → local buffer (with bias)
    #  2. Compute x² sum per row (Q/K) → rms = rsqrt(sum/dim + eps)
    #  3. A2A scatter: x (un-normed) + rms (scalar per row) to peers
    #  4. Peer applies: out = x * rms * weight (elementwise, deferred norm)
    # ════════════════════════════════════════════════════════════════

    # Step 1: GEMM → local buffer (with bias)
    d_local = torch.matmul(a, b.t())  # [local_m, n_total]
    if bias is not None:
        d_local = d_local + bias

    # Step 2: Compute x² sum → rms (precomputed rsqrt, sent to peers)
    # rms_q[row] = rsqrt(sum(d_local[row, :q_dim]²) / q_dim + eps)
    # rms_k[row] = rsqrt(sum(d_local[row, q_dim:q_dim+kv_dim]²) / kv_dim + eps)
    do_norm_q = norm_q_weight is not None
    do_norm_k = norm_k_weight is not None

    rms_q = None
    rms_k = None
    if do_norm_q:
        qf = d_local[:, :q_dim].float()
        rms_q = torch.rsqrt(qf.pow(2).sum(-1) / q_dim + eps)  # [local_m]
    if do_norm_k:
        kf = d_local[:, q_dim:q_dim + kv_dim].float()
        rms_k = torch.rsqrt(kf.pow(2).sum(-1) / kv_dim + eps)  # [local_m]

    # Step 3: A2A scatter (x un-normed + rms)
    local_q_n = (q_nheads // num_ranks) * head_dim
    local_kv_n = (kv_nheads // num_ranks) * head_dim
    local_n = local_q_n + 2 * local_kv_n

    # Scatter x (un-normed) by head groups
    q_view = d_local[:, :q_dim].view(bs, local_seq, num_ranks, local_q_n)
    k_view = d_local[:, q_dim:q_dim+kv_dim].view(bs, local_seq, num_ranks, local_kv_n)
    v_view = d_local[:, q_dim+kv_dim:].view(bs, local_seq, num_ranks, local_kv_n)

    send = torch.cat([q_view, k_view, v_view], dim=-1).permute(2, 0, 1, 3).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=sym_buffer.group)
    scattered_x = recv.permute(1, 0, 2, 3).reshape(bs, seq, local_n).contiguous()
    # scattered_x layout: [bs, seq, local_q_n | local_kv_n | local_kv_n] (un-normed)

    # Scatter rms (tiny: [bs*local_seq, 2] → split by seq)
    # rms is per-src-rank-row, need to send each rank's rms to the right dst
    # Actually rms is per-row in the src's seq shard → dst gets it at the right seq offset
    if do_norm_q or do_norm_k:
        rms_send = torch.stack([rms_q if do_norm_q else torch.zeros(local_m, device=a.device, dtype=torch.float32),
                                rms_k if do_norm_k else torch.zeros(local_m, device=a.device, dtype=torch.float32)], dim=-1)
        # [bs, local_seq, 2] → [num_ranks, bs, local_seq, 2] for A2A
        rms_send = rms_send.view(bs, local_seq, 2).unsqueeze(0).expand(num_ranks, -1, -1, -1).contiguous()
        rms_recv = torch.empty_like(rms_send)
        dist.all_to_all_single(rms_recv, rms_send, group=sym_buffer.group)
        # [num_ranks, bs, local_seq, 2] → [bs, seq, 2]
        rms_gathered = rms_recv.permute(1, 0, 2, 3).reshape(bs, seq, 2).contiguous()

    # Step 4: Apply deferred norm on the receiving side
    out = scattered_x
    if do_norm_q:
        # Q segment: [bs, seq, local_q_n]
        q_out = out[:, :, :local_q_n].float()
        rms_q_recv = rms_gathered[:, :, 0:1]  # [bs, seq, 1]
        # norm weight: this rank's head group of Q
        local_norm_q = norm_q_weight.view(num_ranks, local_q_n)[sym_buffer.group.rank()]
        q_out = (q_out * rms_q_recv * local_norm_q.to(q_out.device)).to(out.dtype)
        out[:, :, :local_q_n] = q_out
    if do_norm_k:
        # K segment: [bs, seq, local_kv_n]
        k_out = out[:, :, local_q_n:local_q_n+local_kv_n].float()
        rms_k_recv = rms_gathered[:, :, 1:2]
        local_norm_k = norm_k_weight.view(num_ranks, local_kv_n)[sym_buffer.group.rank()]
        k_out = (k_out * rms_k_recv * local_norm_k.to(k_out.device)).to(out.dtype)
        out[:, :, local_q_n:local_q_n+local_kv_n] = k_out
    # V segment: no norm

    # Copy to sym_buffer's output region
    out_view = sym_buffer.get_out_view()
    out_view.copy_(out)

    return out_view.clone()
