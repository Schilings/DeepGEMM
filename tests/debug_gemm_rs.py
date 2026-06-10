"""
Debug test for GEMM-RS to identify errors.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'

    tokens_per_rank = 256
    n_dim = 512
    k_dim = 1024
    total_m = tokens_per_rank * num_ranks

    try:
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        dist.broadcast(a, src=0)
        dist.broadcast(b, src=0)
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] Data created and broadcast. a={a.shape}, b={b.shape}')

        # Test reference GEMM first
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_nt(a, b, d_full)
        torch.cuda.synchronize(local_rank)
        if rank_idx == 0:
            print(f'[OK] bf16_gemm_nt works. d_full mean={d_full.float().mean().item():.4f}')

        dist.barrier()

        # Try to create symmetric buffer
        if rank_idx == 0:
            print(f'[...] Creating symmetric buffer...')

        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] Symmetric buffer created. buffer shape={sym_buffer.buffer.shape}')
            print(f'     partial={sym_buffer.partial.shape if sym_buffer.partial is not None else None}')
            print(f'     ready={sym_buffer.ready.shape if sym_buffer.ready is not None else None}')

        # Try fused kernel
        y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        if rank_idx == 0:
            print(f'[...] Running bf16_gemm_rs_nt...')

        deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] bf16_gemm_rs_nt succeeded! y mean={y.float().mean().item():.4f}, max={y.float().abs().max().item():.4f}')

        # Compute reference with reduce_scatter
        dist.reduce_scatter_tensor(
            torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device),
            d_full, op=dist.ReduceOp.SUM, group=group
        )
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] NCCL reduce_scatter_tensor works.')

        # Cleanup
        sym_buffer.destroy()
        dist.barrier()

        if rank_idx == 0:
            print(f'\n✅ ALL BASIC CHECKS PASSED!')

    except Exception as e:
        print(f'Rank {rank_idx}: ❌ ERROR: {e}')
        traceback.print_exc()

    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f'Running debug GEMM-RS test with {num_gpus} GPUs...')
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
