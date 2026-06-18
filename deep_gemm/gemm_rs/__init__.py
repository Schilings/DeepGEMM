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
    BF16 GEMM + Reduce-Scatter 统一入口。

    默认实现采用 Flux 思路的 dual-kernel v3（计算/通信解耦 + tile-ready overlap）。
    可通过环境变量 `DG_GEMM_RS_IMPL` 切换：
      - `v3` / `flux` / `dual`（默认）
      - `single` / `legacy`（旧单 kernel 路径）

    Args:
        y: Output tensor [tokens_per_rank, N], dtype bfloat16 or float32
        a: Input matrix [total_tokens, K], dtype bfloat16
        b: Weight matrix [N, K] (NT layout), dtype bfloat16
        sym_buffer: Symmetric buffer (created via get_symm_buffer_for_gemm_rs)
        num_tokens_per_rank: Actual tokens per rank for this call
        compiled_dims: JIT compilation dimension string
    """
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_gemm_rs_nt is for SM100/B-series GPUs'

    impl = os.getenv('DG_GEMM_RS_IMPL', 'v3').strip().lower()
    if impl in ('v3', 'flux', 'dual'):
        return bf16_gemm_rs_nt_v3(y, a, b, sym_buffer, num_tokens_per_rank, compiled_dims)

    if impl in ('single', 'legacy'):
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
        return

    raise ValueError(
        f"Unsupported DG_GEMM_RS_IMPL={impl!r}, expected one of: v3/flux/dual/single/legacy"
    )


def bf16_gemm_rs_nt_v3(y: torch.Tensor,
                        a: torch.Tensor,
                        b: torch.Tensor,
                        sym_buffer: GemmRSSymmBuffer,
                        num_tokens_per_rank: int,
                        compiled_dims: str = 'nk'):
    """
    BF16 GEMM + Reduce-Scatter v3: Dual-kernel architecture with stream-level overlap.

    This implementation (Flux-inspired dual-kernel design):
    - Uses two separate kernels on different CUDA streams
    - Kernel 1 (GEMM Compute, 256T): No Comm Warps — pure GEMM + epilogue scatter write + flag signaling
    - Kernel 2 (RS Reduce, 256T/CTA): Per-tile polling — waits for flags, then reduces partial results
    - Natural tile-level pipeline overlap via per-tile ready flags

    Stream orchestration:
    - GEMM compute kernel runs on compute_stream
    - RS reduce kernel runs on comm_stream
    - RS reduce polls per-tile flags set by GEMM epilogue — natural tile-level overlap
    - Event synchronization ensures correct ordering

    Expected performance:
    - GEMM throughput should match standard 256T GEMM (~1100 TFLOPS on B300)
    - vs single-kernel 384T GEMM (~600 TFLOPS, register spilling bottleneck)
    - Overlap: while GEMM computes later tiles, RS reduce reduces earlier ones

    Args:
        y: Output tensor [tokens_per_rank, N], dtype bfloat16 or float32
        a: Input matrix [total_tokens, K], dtype bfloat16
        b: Weight matrix [N, K] (NT layout), dtype bfloat16
        sym_buffer: Symmetric buffer (created via get_symm_buffer_for_gemm_rs)
        num_tokens_per_rank: Actual tokens per rank for this call
        compiled_dims: JIT compilation dimension string
    """
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_gemm_rs_nt_v3 is for SM100/B-series GPUs'
    comm_dtype_str = 'fp32' if sym_buffer.use_fp32_comm else 'bf16'

    # Single C++ entry point handles both kernels + stream orchestration + event sync
    _C.bf16_gemm_rs_v3_nt(
        y, a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens_per_rank,
        compiled_dims,
        comm_dtype_str,
    )
