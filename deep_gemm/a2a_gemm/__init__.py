"""
BF16 All-to-All + GEMM fusion for Ulysses Sequence Parallelism.

Usage (Ulysses SP: after Attention, before Wo projection):
  1. Create sym_buffer = get_symm_buffer_for_a2a_gemm(group, M_per_rank, K)
  2. Copy input data into sym_buffer.x (shape [num_ranks, M_per_rank, K])
     - sym_buffer.x[j] = the chunk to send to rank j
  3. Call bf16_a2a_gemm_nt(d, b, sym_buffer, num_tokens)
     - This fuses All2All communication + GEMM computation
  4. Output d has shape [num_ranks * M_per_rank, N]
"""

import torch
from typing import Optional

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load A2A+GEMM kernels, please check your PyTorch version: {exception}')

from .. import _C
from ..utils.math import align


class BF16A2AGemmSymmBuffer:
    """Symmetric buffer for BF16 All2All + GEMM fusion.

    Memory layout:
      - x: [num_ranks, M_per_rank, K] — input data to scatter
        x[j] = chunk to be sent to rank j
      - slots_x: [num_slots, M_per_rank, K] — receive buffers
        slots_x[j] = data received FROM rank j
    """
    def __init__(self, group: dist.ProcessGroup,
                 num_max_tokens_per_rank: int,
                 hidden: int,
                 num_slots: Optional[int] = None):
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_slots = group.size() if num_slots is None else num_slots

        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_bf16_a2a_gemm(
            group.size(), num_max_tokens_per_rank, hidden, self.num_slots)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        self.x, self.slots_x = slice_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.slots_x = None


def get_symm_buffer_for_a2a_gemm(group: dist.ProcessGroup,
                                 num_max_tokens_per_rank: int,
                                 hidden: int,
                                 num_slots: Optional[int] = None) -> BF16A2AGemmSymmBuffer:
    """Create symmetric buffer for A2A+GEMM.

    Args:
        group: Process group
        num_max_tokens_per_rank: Max tokens per rank (will be aligned to 128)
        hidden: K dimension (heads/tp * head_dim for Ulysses)
        num_slots: Number of receive slots (default = group.size())
    """
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_a2a_gemm())
    return BF16A2AGemmSymmBuffer(group, num_max_tokens_per_rank, hidden, num_slots)


def bf16_a2a_gemm_nt(d: torch.Tensor,
                     b: torch.Tensor,
                     sym_buffer: BF16A2AGemmSymmBuffer,
                     num_tokens: int,
                     compiled_dims: str = 'nk'):
    """BF16 All2All + GEMM fusion (NT layout).

    Fuses All-to-All communication with GEMM computation:
    1. A2A scatter: each rank sends sym_buffer.x[j] to rank j
    2. GEMM: concatenated received data × b^T → d

    Before calling:
      Copy your input chunks into sym_buffer.x:
        sym_buffer.x[j].copy_(chunk_for_rank_j)

    Args:
        d: Output [num_ranks * M_per_rank, N], bf16 or fp32
        b: Weight matrix [N, K], bf16 (NT layout)
        sym_buffer: Symmetric buffer with input data in sym_buffer.x
        num_tokens: Actual tokens per rank for this call
        compiled_dims: JIT compilation string
    """
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_a2a_gemm_nt is for SM100/B-series GPUs'
    _C.bf16_a2a_gemm_nt(
        d,
        sym_buffer.buffer,
        b,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens,
        sym_buffer.num_slots,
        compiled_dims,
    )
