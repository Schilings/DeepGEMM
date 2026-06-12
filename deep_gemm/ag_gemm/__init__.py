import torch
from typing import Optional, Tuple


try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load AG+GEMM kernels, please check your PyTorch version: {exception}')

from .. import _C
from ..utils.math import align


class BF16AGGemmSymmBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_max_tokens_per_rank: int,
                 hidden: int,
                 num_slots: Optional[int] = None):
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.num_slots = group.size() if num_slots is None else num_slots

        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_bf16_ag_gemm(
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


class AGGemmSymmBuffer:

    def __init__(self, group: dist.ProcessGroup,
                 num_max_tokens_per_rank: int,
                 hidden: int,
                 gran_k: int = 32,
                 num_slots: Optional[int] = None):
        self.group = group
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.hidden = hidden
        self.gran_k = gran_k
        self.num_slots = group.size() if num_slots is None else num_slots


        num_bytes, slice_buffers = _C.get_symm_buffer_size_for_ag_gemm(
            group.size(), num_max_tokens_per_rank, hidden, gran_k, self.num_slots)

        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        torch.cuda.synchronize()
        self.group.barrier()
        torch.cuda.synchronize()

        self.x, self.x_sf, self.slots_x, self.slots_x_sf = slice_buffers(self.buffer)

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.x = None
        self.x_sf = None
        self.slots_x = None
        self.slots_x_sf = None


def get_symm_buffer_for_ag_gemm(group: dist.ProcessGroup,
                                num_max_tokens_per_rank: int,
                                hidden: int,
                                gran_k: int = 32,
                                num_slots: Optional[int] = None) -> AGGemmSymmBuffer:
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_ag_gemm())
    return AGGemmSymmBuffer(group, num_max_tokens_per_rank, hidden, gran_k, num_slots)


def get_symm_buffer_for_bf16_ag_gemm(group: dist.ProcessGroup,
                                     num_max_tokens_per_rank: int,
                                     hidden: int,
                                     num_slots: Optional[int] = None) -> BF16AGGemmSymmBuffer:
    num_max_tokens_per_rank = align(num_max_tokens_per_rank, _C.get_token_alignment_for_ag_gemm())
    return BF16AGGemmSymmBuffer(group, num_max_tokens_per_rank, hidden, num_slots)




def _fp8_gemm_nt_sm90(a: Tuple[torch.Tensor, torch.Tensor],
                      b: Tuple[torch.Tensor, torch.Tensor],
                      d: torch.Tensor,
                      recipe: Tuple[int, int, int] = (1, 128, 128),
                      compiled_dims: str = 'nk'):
    # SM90 FP8 GEMM path accumulates into FP32 and requires an explicit C tensor.
    out = d if d.dtype == torch.float32 else torch.empty(d.shape, dtype=torch.float32, device=d.device)

    c = torch.zeros_like(out)
    _C.fp8_gemm_nt(a, b, out, c=c, recipe=recipe, compiled_dims=compiled_dims, disable_ue8m0_cast=True)
    if out is not d:
        d.copy_(out.to(d.dtype))


def fp8_ag_gemm_nt_hopper(d: torch.Tensor,
                          a: Tuple[torch.Tensor, torch.Tensor],
                          b: Tuple[torch.Tensor, torch.Tensor],
                          group: dist.ProcessGroup,
                          recipe: Tuple[int, int, int] = (1, 128, 128),
                          compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 9, 'fp8_ag_gemm_nt_hopper is for Hopper/H200 testing'
    assert a[1].dtype == torch.float32 and b[1].dtype == torch.float32, 'SM90 path expects FP32 scaling factors'

    world_size = group.size()
    gathered_x = [torch.empty_like(a[0]) for _ in range(world_size)]
    gathered_x_sf = [torch.empty_like(a[1]) for _ in range(world_size)]
    dist.all_gather(gathered_x, a[0], group=group)
    dist.all_gather(gathered_x_sf, a[1], group=group)
    full_a = (torch.cat(gathered_x, dim=0).contiguous(), torch.cat(gathered_x_sf, dim=0).contiguous())
    _fp8_gemm_nt_sm90(full_a, b, d, recipe=recipe, compiled_dims=compiled_dims)


def _bf16_gemm_nt(a: torch.Tensor,
                  b: torch.Tensor,
                  d: torch.Tensor,
                  compiled_dims: str = 'nk'):
    out = d if d.dtype in (torch.bfloat16, torch.float32) else torch.empty(d.shape, dtype=torch.bfloat16, device=d.device)
    _C.bf16_gemm_nt(a, b, out, c=None, compiled_dims=compiled_dims)
    if out is not d:
        d.copy_(out.to(d.dtype))


def bf16_ag_gemm_nt_hopper(d: torch.Tensor,
                           a: torch.Tensor,
                           b: torch.Tensor,
                           group: dist.ProcessGroup,
                           compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 9, 'bf16_ag_gemm_nt_hopper is for Hopper/H200 testing'
    world_size = group.size()
    gathered_x = [torch.empty_like(a) for _ in range(world_size)]
    dist.all_gather(gathered_x, a, group=group)
    full_a = torch.cat(gathered_x, dim=0).contiguous()
    _bf16_gemm_nt(full_a, b, d, compiled_dims=compiled_dims)



def bf16_ag_gemm_nt(d: torch.Tensor,
                    b: torch.Tensor,
                    sym_buffer: BF16AGGemmSymmBuffer,
                    num_tokens: int,
                    compiled_dims: str = 'nk'):
    assert torch.cuda.get_device_capability()[0] == 10, 'bf16_ag_gemm_nt is for SM100/B-series GPUs'
    _C.bf16_ag_gemm_nt(
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


def fp8_ag_gemm_nt(d: torch.Tensor,



                   b: Tuple[torch.Tensor, torch.Tensor],
                   sym_buffer: AGGemmSymmBuffer,
                   num_tokens: int,
                   recipe: Tuple[int, int, int] = (1, 1, 32),
                   compiled_dims: str = 'nk'):
    _C.fp8_ag_gemm_nt(
        d, b,
        sym_buffer.buffer,
        sym_buffer.handle.buffer_ptrs,
        sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank,
        num_tokens,
        sym_buffer.gran_k,
        sym_buffer.num_slots,
        recipe,
        compiled_dims,
    )

