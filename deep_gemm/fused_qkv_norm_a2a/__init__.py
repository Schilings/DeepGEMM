"""
Fused QKV GEMM + RMSNorm(x²sum) + A2A-transpose (Ulysses SP pre-attn).

Single-kernel approach (norm-deferred):
  1. GEMM epilogue: x² partial sum (atomic add) + P2P TMA scatter x (un-normed)
  2. Post-barrier: compute rms = rsqrt(sum/dim+eps) + scatter rms to peers
  3. Peer applies: out = x * rms * weight (elementwise, open to any downstream op)

The kernel returns:
  - out: [bs, seq, local_n_total] bf16 (un-normed QKV, scattered by head group)
  - rms: [bs, seq, 2] fp32 (Q rms at [:,:,0], K rms at [:,:,1]; zeros if norm disabled)

When norm is disabled (do_norm_q=False, do_norm_k=False), this degenerates to
GEMM + A2A scatter (like bf16_gemm_a2a_transpose_nt but with GQA support).

Usage:
  sym = get_symm_buffer_for_fused_qkv_norm_a2a(group, bs, seq, q_nheads, kv_nheads, head_dim)
  out, rms = bf16_fused_qkv_norm_a2a_nt(x, Wqkv, sym, local_seq, ...)
  # Downstream: q = out[:,:,:local_q_n] * rms[:,:,0:1] * norm_q_weight
"""

import torch
from typing import Optional, Tuple

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load fused QKV+Norm+A2A kernels: {exception}')

from .. import _C
from ..utils.math import align


class FusedQKVNormA2ASymmBuffer:
    """Symmetric buffer: [barrier(32B) | rms(bs*seq*2*4) | out(bs*seq*local_n*2)]"""
    def __init__(self, group, bs, seq, q_nheads, kv_nheads, head_dim,
                 out_dtype=torch.bfloat16):
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
        self.local_n = self.local_q_n + 2 * self.local_kv_n

        elem_size = 4 if out_dtype == torch.float32 else 2
        rms_bytes = bs * seq * 2 * 4  # [bs*seq, 2] float32
        out_bytes = bs * seq * self.local_n * elem_size
        num_bytes = 32 + rms_bytes + out_bytes
        num_bytes = (num_bytes + 15) // 16 * 16

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

    def get_out_view(self):
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        rms_bytes = self.bs * self.seq * 2 * 4
        out_offset = 32 + rms_bytes
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, out_offset // elem_size, (self.bs, self.seq, self.local_n))

    def get_rms_view(self):
        """[bs, seq, 2] float32 — Q rms at [:,:,0], K rms at [:,:,1]"""
        rms_offset = 32  # right after barrier region
        return torch.empty(
            (self.bs, self.seq, 2),
            dtype=torch.float32, device=self.buffer.device
        ).set_(self.buffer, rms_offset // 4, (self.bs, self.seq, 2))

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None


def get_symm_buffer_for_fused_qkv_norm_a2a(group, bs, seq, q_nheads, kv_nheads, head_dim,
                                            out_dtype=torch.bfloat16):
    assert q_nheads % group.size() == 0
    assert kv_nheads % group.size() == 0
    assert seq % group.size() == 0
    return FusedQKVNormA2ASymmBuffer(group, bs, seq, q_nheads, kv_nheads, head_dim, out_dtype)


def bf16_fused_qkv_norm_a2a_nt(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out, rms):
      out: [bs, seq, local_n_total] bf16 — un-normed QKV scattered by head
      rms: [bs, seq, 2] fp32 — Q rms at [:,:,0], K rms at [:,:,1]
    Downstream applies: q = out[:,:,:local_q_n] * rms[:,:,0:1] * norm_q_weight
    """
    assert torch.cuda.get_device_capability()[0] == 10
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16

    group = sym_buffer.group
    num_ranks = sym_buffer.world_size
    bs = sym_buffer.bs
    seq = local_seq * num_ranks
    q_dim = q_nheads * head_dim
    kv_dim = kv_nheads * head_dim
    n_total = q_dim + 2 * kv_dim
    local_m = bs * local_seq

    assert a.shape[0] == local_m
    assert b.shape[0] == n_total
    assert local_seq % 128 == 0

    do_norm_q = norm_q_weight is not None
    do_norm_k = norm_k_weight is not None

    # Allocate sum_buffer: [local_m, 2] (indexed by GEMM M row, zeroed by host)
    sum_buffer = torch.zeros(local_m, 2, dtype=torch.float32, device=a.device)

    # Call CUDA kernel
    _C.sm100_bf16_fused_qkv_norm_a2a_nt(
        a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sum_buffer,
        group.rank(),
        bs, local_seq,
        q_nheads, kv_nheads, head_dim,
        eps, do_norm_q, do_norm_k)

    # Sync: kernel has 3 nvlink barriers internally (init + tiles + rms).
    # After kernel returns, data is globally visible. Only need group barrier
    # to ensure all ranks finished before we read sym buffer.
    group.barrier()

    # Get output views (operate in-place on sym buffer, clone only at end)
    out = sym_buffer.get_out_view()
    rms = sym_buffer.get_rms_view().clone()  # small: [bs, seq, 2] fp32

    # Apply bias + norm in-place (minimal kernel launches)
    # Strategy: precompute rms*weight, then single fused add+mul per segment
    local_q_n = (q_nheads // num_ranks) * head_dim
    local_kv_n = (kv_nheads // num_ranks) * head_dim

    if do_norm_q:
        # Precompute rms_q * norm_q_weight (broadcast over seq)
        local_norm_q = norm_q_weight.view(num_ranks, local_q_n)[group.rank()]
        rms_q_scaled = rms[:, :, 0:1] * local_norm_q.float()  # [bs, seq, 1] * [local_q_n]
        q_slice = out[:, :, :local_q_n]
        if bias is not None:
            q_slice.add_(bias[:q_dim].view(num_ranks, local_q_n)[group.rank()].to(out.dtype))
        q_slice.mul_(rms_q_scaled.to(out.dtype))  # [bs, seq, local_q_n]
    elif bias is not None:
        out[:, :, :local_q_n].add_(bias[:q_dim].view(num_ranks, local_q_n)[group.rank()].to(out.dtype))

    if do_norm_k:
        local_norm_k = norm_k_weight.view(num_ranks, local_kv_n)[group.rank()]
        rms_k_scaled = rms[:, :, 1:2] * local_norm_k.float()
        k_slice = out[:, :, local_q_n:local_q_n+local_kv_n]
        if bias is not None:
            k_slice.add_(bias[q_dim:q_dim+kv_dim].view(num_ranks, local_kv_n)[group.rank()].to(out.dtype))
        k_slice.mul_(rms_k_scaled.to(out.dtype))
    elif bias is not None:
        out[:, :, local_q_n:local_q_n+local_kv_n].add_(
            bias[q_dim:q_dim+kv_dim].view(num_ranks, local_kv_n)[group.rank()].to(out.dtype))

    if bias is not None:
        out[:, :, local_q_n+local_kv_n:].add_(
            bias[q_dim+kv_dim:].view(num_ranks, local_kv_n)[group.rank()].to(out.dtype))

    return out.clone(), rms
