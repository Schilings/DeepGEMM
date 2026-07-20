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

Entry points:
  bf16_a2a_transpose_gemm_nt        — DEFAULT (M0): comm(all SMs) + SP barrier + GEMM(all SMs).
                                      Fastest/most stable on a single node.
  bf16_a2a_transpose_gemm_nt_fused  — M1 (opt-in): comm/GEMM overlap via per-M-tile barrier.
                                      ~parity on a single node; for comm<<gemm or multi-node.
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


def bf16_a2a_transpose(sym_buffer,
                       seq_major: bool = False) -> torch.Tensor:
    """Run the transpose-scatter comm + SP-group barrier. Returns gathered [bs*local_seq, hidden].

    Accepts either ``BF16A2ATransposeGemmSymmBuffer`` or ``UnifiedSymmBuffer``
    (with attention params).  For ``UnifiedSymmBuffer``, views are accessed via
    ``sym_buffer.a2a_gemm.x`` / ``sym_buffer.a2a_gemm.gathered``.

    seq_major=True consumes FlashAttention-native BSHD input. Default False expects BHSD.
    """
    # UnifiedSymmBuffer stores views in a2a_gemm NamedTuple; legacy buffer uses flat attrs
    views = getattr(sym_buffer, 'a2a_gemm', None)
    x = views.x if views is not None else sym_buffer.x
    gathered = views.gathered if views is not None else sym_buffer.gathered

    _C.bf16_a2a_transpose_comm(
        sym_buffer.buffer, sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        sym_buffer.bs, sym_buffer.nheads, sym_buffer.seq, sym_buffer.head_dim, seq_major)
    torch.cuda.synchronize()
    sym_buffer.group.barrier()
    torch.cuda.synchronize()
    return gathered


def bf16_a2a_transpose_gemm_nt(d: torch.Tensor,
                               b: torch.Tensor,
                               sym_buffer: BF16A2ATransposeGemmSymmBuffer,
                               seq_major: bool = False):
    """DEFAULT (M0): A2A-transpose comm (all SMs, ring-rotated) + SP-group barrier + standard Wo GEMM.

    This is the fastest, most stable path on a SINGLE node: comm and GEMM each get all SMs, and the
    rotated transpose-scatter comm is ~3-4x faster than NCCL all_to_all + torch transposes. Fusing
    the overlap (see `..._fused`) is ~parity / net-negative on one node because the comm's SM
    carveout slows the GEMM by more than the comm it hides; the fused path pays off only when
    comm << gemm or across nodes.

    Args:
      d: output [bs*local_seq, N], bf16.
      b: Wo weight [N, hidden], bf16 (NT layout).
      sym_buffer: with this rank's attention output already in sym_buffer.x.
      seq_major: if True, sym.x holds FlashAttention-native BSHD [bs, seq, local_nheads, head_dim]
        and the comm consumes it directly (no .permute to BHSD). With FA-based attention this avoids
        a full-HBM transpose pass and speeds the whole post-attn op ~1.25-1.45x. Default False expects
        BHSD [bs, local_nheads, seq, head_dim].
    """
    import deep_gemm
    a = bf16_a2a_transpose(sym_buffer, seq_major=seq_major)   # [bs*local_seq, hidden]
    deep_gemm.bf16_gemm_nt(a, b, d)


def bf16_a2a_transpose_gemm_nt_fused(d: torch.Tensor,
                                     b: torch.Tensor,
                                     sym_buffer: BF16A2ATransposeGemmSymmBuffer):
    """M1 (fused, opt-in): transpose-scatter comm overlapped with the Wo GEMM via per-M-tile barrier.

    On a single node this is ~parity / slightly slower than the default M0 (the comm SM carveout
    costs more than the comm it hides). Prefer this only for comm<<gemm or multi-node setups.

    Args:
      d: output [bs*local_seq, N], bf16.
      b: Wo weight [N, hidden], bf16 (NT layout).
      sym_buffer: with this rank's attention output already in sym_buffer.x.

    NOTE: the per-M-tile barriers must be 0 on entry. They are zeroed at buffer creation; for
    repeated calls on the same buffer, reset via sym_buffer.reset_barriers() with an SP-group
    sync in between.
    """
    views = getattr(sym_buffer, 'a2a_gemm', None)
    gathered = views.gathered if views is not None else sym_buffer.gathered
    _C.bf16_a2a_transpose_gemm_nt(
        d, gathered, b, sym_buffer.buffer, sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(), sym_buffer.bs, sym_buffer.nheads,
        sym_buffer.seq, sym_buffer.head_dim)
