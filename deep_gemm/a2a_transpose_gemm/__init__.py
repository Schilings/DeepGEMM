"""
BF16 Ulysses-SP post-attention All2All-transpose + Wo GEMM.

Correct dataflow (unlike the M-axis token-A2A `a2a_gemm`):
  input per rank r: x_r[bs, local_nheads, seq, head_dim]  (rank r owns heads
      [r*local_nheads:(r+1)*local_nheads], FULL seq)
  A2A-transpose (gather along hidden + seq<->head transpose):
      xt_r[bs, local_seq, hidden]  (rank r keeps its seq shard, gathers ALL heads)
  Wo GEMM: y_r[bs*local_seq, N] = xt_r.reshape(bs*local_seq, hidden) @ Wo.t()

Usage:
  sym = get_symm_buffer_for_a2a_transpose_gemm(group, bs, nheads, seq, head_dim)
  sym.x.copy_(x_r)                       # [bs, local_nheads, seq, head_dim]
  d = torch.empty((bs*local_seq, N), dtype=torch.bfloat16, device='cuda')
  bf16_a2a_transpose_gemm_nt(d, Wo, sym)  # Wo: [N, hidden]

M0 (correctness-first): comm-then-GEMM with an SP-group barrier (no overlap yet).
"""

import torch
from typing import Optional

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load A2A-transpose+GEMM kernels, check your PyTorch version: {exception}')

from .. import _C


class BF16A2ATransposeGemmSymmBuffer:
    """Symmetric buffer for the A2A-transpose + Wo GEMM.

    Fields:
      x       : [bs, local_nheads, seq, head_dim] — write this rank's attention output here.
      gathered: [bs*local_seq, hidden]            — A matrix for the Wo GEMM (filled by comm).
    """
    def __init__(self, group: dist.ProcessGroup, bs: int, nheads: int, seq: int, head_dim: int):
        self.group = group
        self.world_size = group.size()
        assert nheads % self.world_size == 0, 'nheads must be divisible by sp_size'
        assert seq % self.world_size == 0, 'seq must be divisible by sp_size'
        assert head_dim % 8 == 0, 'head_dim must be a multiple of 8'
        self.bs, self.nheads, self.seq, self.head_dim = bs, nheads, seq, head_dim
        self.local_seq = seq // self.world_size
        self.hidden = nheads * head_dim

        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_bf16_a2a_transpose_gemm(
            self.world_size, bs, nheads, seq, head_dim)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()
        self.x, self.gathered = slice_buffers(self.buffer)

    def _barrier_bytes(self) -> int:
        tiles_per_seq = (self.local_seq + 127) // 128
        num_m_tiles = self.bs * tiles_per_seq
        nbytes = (num_m_tiles + 1) * 4
        return (nbytes + 127) // 128 * 128

    def reset_barriers(self):
        """Zero the per-M-tile barrier region (offset 0). Caller must SP-group-sync around this
        so all ranks reset before any peer's comm writes (fused path reuse across calls)."""
        self.buffer[:self._barrier_bytes()].zero_()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.gathered = None


def get_symm_buffer_for_a2a_transpose_gemm(group: dist.ProcessGroup,
                                           bs: int, nheads: int, seq: int, head_dim: int
                                           ) -> BF16A2ATransposeGemmSymmBuffer:
    return BF16A2ATransposeGemmSymmBuffer(group, bs, nheads, seq, head_dim)


def bf16_a2a_transpose(sym_buffer: BF16A2ATransposeGemmSymmBuffer) -> torch.Tensor:
    """Run the transpose-scatter comm + SP-group barrier. Returns gathered [bs*local_seq, hidden]."""
    _C.bf16_a2a_transpose_comm(
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.bs, sym_buffer.nheads, sym_buffer.seq, sym_buffer.head_dim)
    # M0: ensure all peers finished writing this rank's gathered region before reading it.
    torch.cuda.synchronize()
    sym_buffer.group.barrier()
    torch.cuda.synchronize()
    return sym_buffer.gathered


def bf16_a2a_transpose_gemm_nt_m0(d: torch.Tensor,
                                  b: torch.Tensor,
                                  sym_buffer: BF16A2ATransposeGemmSymmBuffer):
    """M0 fallback: A2A-transpose comm + SP-group barrier + standard Wo GEMM (no overlap)."""
    import deep_gemm
    a = bf16_a2a_transpose(sym_buffer)            # [bs*local_seq, hidden]
    deep_gemm.bf16_gemm_nt(a, b, d)


def bf16_a2a_transpose_gemm_nt(d: torch.Tensor,
                               b: torch.Tensor,
                               sym_buffer: BF16A2ATransposeGemmSymmBuffer):
    """M1 (fused): transpose-scatter comm overlapped with the Wo GEMM via per-M-tile barrier.

    Args:
      d: output [bs*local_seq, N], bf16.
      b: Wo weight [N, hidden], bf16 (NT layout).
      sym_buffer: with this rank's attention output already in sym_buffer.x.

    NOTE: the per-M-tile barriers must be 0 on entry. They are zeroed at buffer creation; for
    repeated calls on the same buffer, reset via sym_buffer.reset_barriers() with an SP-group
    sync in between (see M2).
    """
    _C.bf16_a2a_transpose_gemm_nt(
        d, sym_buffer.gathered, b, sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.bs, sym_buffer.nheads,
        sym_buffer.seq, sym_buffer.head_dim)
