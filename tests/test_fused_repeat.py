"""
Test: Call bf16_gemm_rs_fused multiple times in succession to verify no barrier race.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run_test(local_rank, num_local_ranks, num_repeats):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    tpr, n, k = 256, 512, 1024
    total_m = tpr * num_ranks

    a = torch.randn((total_m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
    dist.broadcast(a, src=0)

    sym_buf = deep_gemm.get_symm_buffer_for_gemm_rs(group, tpr, n, out_dtype=torch.bfloat16)
    y = torch.zeros((tpr, n), dtype=torch.bfloat16, device='cuda')

    import sys
    if rank_idx == 0:
        print("  Before first kernel call...", flush=True)
    dist.barrier(group)
    
    for i in range(num_repeats):
        if rank_idx == 0:
            print(f"  Starting iteration {i+1}...", flush=True)
        deep_gemm.bf16_gemm_rs_fused(y, a, b, sym_buf, tpr, compiled_dims='nk')
        if rank_idx == 0:
            print(f"  Kernel launched, synchronizing...", flush=True)
        torch.cuda.synchronize()
        if rank_idx == 0:
            print(f"  Synchronized, barrier...", flush=True)
        dist.barrier(group)
        if rank_idx == 0:
            print(f"  Iteration {i+1}/{num_repeats} completed successfully", flush=True)

    sym_buf.destroy()

    if rank_idx == 0:
        print(f"\nAll {num_repeats} iterations PASSED!")

    dist.barrier(group)
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    num_repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Testing fused kernel repeated calls: {num_gpus} GPUs, {num_repeats} repeats...")
    mp.spawn(run_test, args=(num_gpus, num_repeats), nprocs=num_gpus, join=True)
