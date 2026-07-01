import os, sys
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29556'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deep_gemm.ag_gemm import get_symm_buffer_for_bf16_ag_gemm

def worker(rank, ng):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(ng)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    buf = get_symm_buffer_for_bf16_ag_gemm(group, 1024, 5120)
    if rank == 0:
        print(f'buffer shape: {buf.buffer.shape}, dtype: {buf.buffer.dtype}')
        print(f'x shape: {buf.x.shape}, dtype: {buf.x.dtype}')
        print(f'slots_x type: {type(buf.slots_x)}')
        if isinstance(buf.slots_x, list):
            print(f'  len: {len(buf.slots_x)}')
            for i, s in enumerate(buf.slots_x):
                print(f'  [{i}] shape: {s.shape}, dtype: {s.dtype}')
        else:
            print(f'  shape: {buf.slots_x.shape}, dtype: {buf.slots_x.dtype}')
    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
