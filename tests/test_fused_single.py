"""
Minimal test: single call to bf16_gemm_rs_fused, check if all ranks complete.
"""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
import deep_gemm
from deep_gemm.utils.dist import init_dist

def run_test(local_rank, num_local_ranks):
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

    print(f"  [Rank {rank_idx}] Before kernel call", flush=True)
    dist.barrier(group)
    
    deep_gemm.bf16_gemm_rs_fused(y, a, b, sym_buf, tpr, compiled_dims='nk')
    
    print(f"  [Rank {rank_idx}] Kernel launched, synchronizing...", flush=True)
    torch.cuda.synchronize()
    print(f"  [Rank {rank_idx}] Synchronized!", flush=True)

    sym_buf.destroy()
    dist.barrier(group)
    if rank_idx == 0:
        print("ALL RANKS COMPLETED!", flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Single-call test: {num_gpus} GPUs")
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
