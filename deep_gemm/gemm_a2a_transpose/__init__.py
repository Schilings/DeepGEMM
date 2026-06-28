import torch
from typing import Optional

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load GEMM+A2A-transpose kernels, please check your PyTorch version: {exception}')

from .. import _C
from ..utils.math import align


class GemmA2ATransposeSymmBuffer:
    """Symmetric buffer for the Ulysses-SP pre-attn fused GEMM + All2All-transpose.

    The buffer holds ONLY the scattered OUTPUT region [bs, seq, local_n] (plus a 32B barrier
    header). Unlike GEMM-RS there are no partial slots and no ready flags: the A2A is a pure
    permutation, so the output region IS the result for this rank's head group after the kernel
    returns. `out` is a [bs, seq, local_n] view of that region (local_n = N / num_ranks =
    local_nheads * head_dim, i.e. BSHD [bs, seq, local_nheads, head_dim] flattened on the last 2 dims).
    """
    def __init__(self, group: dist.ProcessGroup,
                 bs: int,
                 max_seq: int,
                 n: int,
                 out_dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            group: Process group for distributed communication
            bs: Batch size (THD layout is the bs=1 special case)
            max_seq: Maximum full sequence length (= local_seq * num_ranks; must be aligned)
            n: Full projection width N = nheads * head_dim
            out_dtype: Output dtype (bfloat16 or float32)
        """
        self.group = group
        self.bs = bs
        self.max_seq = max_seq
        self.n = n
        self.out_dtype = out_dtype
        self.use_fp32_output = out_dtype == torch.float32

        num_bytes, slice_buffer = _C.get_symm_buffer_size_for_gemm_a2a_transpose(
            group.size(), bs, max_seq, n, self.use_fp32_output)
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        self.out = slice_buffer(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.out = None


def get_symm_buffer_for_gemm_a2a_transpose(group: dist.ProcessGroup,
                                           bs: int,
                                           max_seq: int,
                                           n: int,
                                           out_dtype: torch.dtype = torch.bfloat16) -> GemmA2ATransposeSymmBuffer:
    """Create a symmetric buffer for the pre-attn GEMM + All2All-transpose."""
    max_seq = align(max_seq, _C.get_seq_alignment_for_gemm_a2a_transpose())
    return GemmA2ATransposeSymmBuffer(group, bs, max_seq, n, out_dtype)


def bf16_gemm_a2a_transpose_nt(a: torch.Tensor,
                               b: torch.Tensor,
                               sym_buffer: GemmA2ATransposeSymmBuffer,
                               local_seq: int,
                               compiled_dims: str = 'nk') -> torch.Tensor:
    """BF16 pre-attn fused GEMM + All2All-transpose (Ulysses SP) — single kernel.

      QKV/Q projection on this rank's local seq shard, then a head-wise All2All-transpose so
      each rank ends up owning a head group with the FULL seq (BSHD), ready for FlashAttention.

      Dual of post-attn `bf16_a2a_transpose_gemm_nt` (which does a2a + Wo GEMM). Here the GEMM
      runs first and communication is pushed in the epilogue (no reduce, pure permutation).

    Args:
        a: Local activations [bs*local_seq, K], dtype bfloat16 (seq-sharded, full hidden K)
        b: Projection weights [N, K] (NT layout), dtype bfloat16, N = nheads*head_dim
        sym_buffer: Symmetric buffer (from get_symm_buffer_for_gemm_a2a_transpose)
        local_seq: This rank's local sequence length (= seq / num_ranks; must be 128-aligned)
        compiled_dims: JIT compilation dimension string

    Returns:
        out: [bs, seq, local_n] view of this rank's head-group output (seq = local_seq*num_ranks,
             local_n = N/num_ranks). This is BSHD [bs, seq, local_nheads, head_dim] flattened.
    """
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_gemm_a2a_transpose_nt is for SM100/B-series GPUs'

    comm_dtype_str = 'fp32' if sym_buffer.use_fp32_output else 'bf16'
    _C.bf16_gemm_a2a_transpose_nt(
        a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.bs,
        sym_buffer.max_seq,
        local_seq,
        compiled_dims,
        comm_dtype_str,
    )
    seq = local_seq * sym_buffer.group.size()
    return sym_buffer.out[:, :seq, :]
