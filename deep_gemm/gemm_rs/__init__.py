import os
import torch
from typing import Tuple

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load GEMM+RS kernels, please check your PyTorch version: {exception}')

from .. import _C
from ..utils.math import align


class GemmRSSymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_max_tokens_per_rank: int,
                 hidden: int,
                 out_dtype: torch.dtype = torch.bfloat16,
                 comm_dtype: torch.dtype = None):
        """
        Args:
            group: Process group for distributed communication
            num_max_tokens_per_rank: Maximum tokens per rank (must be aligned)
            hidden: Hidden dimension (N)
            out_dtype: Output tensor dtype (bfloat16 or float32)
            comm_dtype: Communication dtype for partial buffer (bfloat16 or float32).
                        Controls the precision of NVLink data transfer.
                        - bfloat16 (default): saves 50% NVLink bandwidth, slight precision loss in reduce
                        - float32: full precision communication, 2x bandwidth cost
                        If None, defaults to bfloat16 (recommended for training).
        """
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.out_dtype = out_dtype
        self.comm_dtype = comm_dtype if comm_dtype is not None else torch.bfloat16
        self.use_fp32_comm = self.comm_dtype == torch.float32

        # Buffer size is determined by comm_dtype (what's stored in partial buffer)
        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_gemm_rs(
            group.size(), num_max_tokens_per_rank, hidden, self.use_fp32_comm)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        self.partial, self.ready = slice_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.partial = None
        self.ready = None


def get_symm_buffer_for_gemm_rs(group: dist.ProcessGroup,
                                num_max_tokens_per_rank: int,
                                hidden: int,
                                out_dtype: torch.dtype = torch.bfloat16,
                                comm_dtype: torch.dtype = None) -> GemmRSSymmBuffer:
    """
    Create a symmetric buffer for GEMM + Reduce-Scatter.

    Args:
        group: Process group
        num_max_tokens_per_rank: Maximum tokens per rank
        hidden: Hidden dimension (N)
        out_dtype: Output tensor dtype
        comm_dtype: Communication dtype for NVLink transfer (bfloat16 or float32).
                    - None/bfloat16: saves bandwidth, recommended for training
                    - float32: full precision, 2x bandwidth cost
    """
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_gemm_rs())
    return GemmRSSymmBuffer(group, num_max_tokens_per_rank, hidden, out_dtype, comm_dtype)


def bf16_gemm_rs_nt(y: torch.Tensor,
                    a: torch.Tensor,
                    b: torch.Tensor,
                    sym_buffer: GemmRSSymmBuffer,
                    num_tokens_per_rank: int,
                    compiled_dims: str = 'nk'):
    """
    BF16 GEMM + Reduce-Scatter 统一入口 —— 真·Flux pull 式 dual-kernel。

      - Kernel 1 (GEMM compute, 256T, 无 comm warps): epilogue 纯本地 scatter 写 slot[dst_rank] + 置本地 flag
      - Kernel 2 (RS reduce, pull): 从各远端 rank 的 scatter buffer 拉取并 FP32 reduce → output
      - compute_stream / comm_stream 流级 overlap + per-tile flag tile 级 overlap

    Args:
        y: Output tensor [tokens_per_rank, N], dtype bfloat16 or float32
        a: Input matrix [total_tokens, K], dtype bfloat16
        b: Weight matrix [N, K] (NT layout), dtype bfloat16
        sym_buffer: Symmetric buffer (created via get_symm_buffer_for_gemm_rs)
        num_tokens_per_rank: Actual tokens per rank for this call
        compiled_dims: JIT compilation dimension string
    """
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_gemm_rs_nt is for SM100/B-series GPUs'

    comm_dtype_str = 'fp32' if sym_buffer.use_fp32_comm else 'bf16'
    _C.bf16_gemm_rs_nt(
        y, a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens_per_rank,
        compiled_dims,
        comm_dtype_str,
    )
