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
                 out_dtype: torch.dtype = torch.bfloat16):
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.out_dtype = out_dtype
        self.use_fp32_output = out_dtype == torch.float32


        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_gemm_rs(
            group.size(), num_max_tokens_per_rank, hidden, self.use_fp32_output)
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
                                out_dtype: torch.dtype = torch.bfloat16) -> GemmRSSymmBuffer:
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_gemm_rs())
    return GemmRSSymmBuffer(group, num_max_tokens_per_rank, hidden, out_dtype)


def _fp8_gemm_nt_sm90(a: Tuple[torch.Tensor, torch.Tensor],
                      b: Tuple[torch.Tensor, torch.Tensor],
                      d: torch.Tensor,
                      recipe: Tuple[int, int, int] = (1, 128, 128),
                      compiled_dims: str = 'nk'):
    out = d if d.dtype == torch.float32 else torch.empty(d.shape, dtype=torch.float32, device=d.device)
    c = torch.zeros_like(out)
    _C.fp8_gemm_nt(a, b, out, c=c, recipe=recipe, compiled_dims=compiled_dims, disable_ue8m0_cast=True)
    if out is not d:
        d.copy_(out.to(d.dtype))


def fp8_gemm_rs_nt_hopper(y: torch.Tensor,
                          a: Tuple[torch.Tensor, torch.Tensor],
                          b: Tuple[torch.Tensor, torch.Tensor],
                          group: dist.ProcessGroup,
                          recipe: Tuple[int, int, int] = (1, 128, 128),
                          compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 9, 'fp8_gemm_rs_nt_hopper is for Hopper/H200 testing'
    assert a[1].dtype == torch.float32 and b[1].dtype == torch.float32, 'SM90 path expects FP32 scaling factors'
    world_size = group.size()
    assert a[0].shape[0] % world_size == 0
    tokens_per_rank = a[0].shape[0] // world_size
    partial = torch.empty((a[0].shape[0], b[0].shape[0]), dtype=torch.float32, device=y.device)
    _fp8_gemm_nt_sm90(a, b, partial, recipe=recipe, compiled_dims=compiled_dims)
    reduced = y if y.dtype == torch.float32 else torch.empty((tokens_per_rank, b[0].shape[0]), dtype=torch.float32, device=y.device)
    dist.reduce_scatter_tensor(reduced, partial.contiguous(), op=dist.ReduceOp.SUM, group=group)
    if reduced is not y:
        y.copy_(reduced.to(y.dtype))


def _bf16_gemm_nt(a: torch.Tensor,
                  b: torch.Tensor,
                  d: torch.Tensor,
                  compiled_dims: str = 'nk'):
    _C.bf16_gemm_nt(a, b, d, c=None, compiled_dims=compiled_dims)


def bf16_gemm_rs_nt_hopper(y: torch.Tensor,
                           a: torch.Tensor,
                           b: torch.Tensor,
                           group: dist.ProcessGroup,
                           compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 9, 'bf16_gemm_rs_nt_hopper is for Hopper/H200 testing'
    _bf16_gemm_rs_nt_impl(y, a, b, group, compiled_dims=compiled_dims)


def _bf16_gemm_rs_nt_impl(y: torch.Tensor,
                          a: torch.Tensor,
                          b: torch.Tensor,
                          group: dist.ProcessGroup,
                          compiled_dims: str = 'nk'):
    world_size = group.size()
    assert a.shape[0] % world_size == 0
    tokens_per_rank = a.shape[0] // world_size
    partial = torch.empty((a.shape[0], b.shape[0]), dtype=torch.float32, device=y.device)
    _bf16_gemm_nt(a, b, partial, compiled_dims=compiled_dims)
    reduced = y if y.dtype == torch.float32 else torch.empty((tokens_per_rank, b.shape[0]), dtype=torch.float32, device=y.device)
    dist.reduce_scatter_tensor(reduced, partial.contiguous(), op=dist.ReduceOp.SUM, group=group)
    if reduced is not y:
        y.copy_(reduced.to(y.dtype))


def bf16_gemm_rs_nt(y: torch.Tensor,
                    a: torch.Tensor,
                    b: torch.Tensor,
                    sym_buffer: GemmRSSymmBuffer,
                    num_tokens_per_rank: int,
                    compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_gemm_rs_nt is for SM100/B-series GPUs'
    _C.bf16_gemm_rs_nt(
        y, a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens_per_rank,
        compiled_dims,
    )


def fp8_gemm_rs_nt(y: torch.Tensor,



                   a: Tuple[torch.Tensor, torch.Tensor],
                   b: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: GemmRSSymmBuffer,
                   num_tokens_per_rank: int,
                   recipe: Tuple[int, int, int] = (1, 1, 32),
                   compiled_dims: str = 'nk',
                   disable_ue8m0_cast: bool = False):
    _C.fp8_gemm_rs_nt(
        y, a, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens_per_rank,
        recipe,
        compiled_dims,
        disable_ue8m0_cast,
    )

