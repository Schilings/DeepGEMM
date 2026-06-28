"""
Minimal debug test for GEMM-RS with 2 GPUs first.
Uses smallest possible shape to minimize compute time and isolate comm issues.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback

os.environ['DG_JIT_DEBUG'] = '1'

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'

    # Use smallest valid shape: 
    # block_m will be 128 (or less), block_n=128, block_k=64
    # tokens_per_rank must be >= block_m and aligned
    tokens_per_rank = 128
    n_dim = 128
    k_dim = 64
    total_m = tokens_per_rank * num_ranks

    try:
        # Create deterministic data
        torch.manual_seed(42)
        a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
        # Ensure all ranks have same data
        dist.broadcast(a, src=0)
        dist.broadcast(b, src=0)
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] Data: a={a.shape}, b={b.shape}, total_m={total_m}')
            print(f'     tokens_per_rank={tokens_per_rank}, n={n_dim}, k={k_dim}')

        # Step 1: Reference with GEMM + NCCL reduce_scatter
        d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_nt(a, b, d_full)
        torch.cuda.synchronize(local_rank)
        
        y_ref = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        dist.reduce_scatter_tensor(y_ref, d_full, op=dist.ReduceOp.SUM, group=group)
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] Reference computed: y_ref mean={y_ref.float().mean().item():.4f}')

        # Step 2: Symmetric buffer creation
        if rank_idx == 0:
            print(f'[...] Creating symmetric buffer...')
        
        sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
            group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
        )
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            print(f'[OK] Symmetric buffer created. buffer={sym_buffer.buffer.shape}')
            print(f'     partial={sym_buffer.partial.shape}')
            print(f'     ready={sym_buffer.ready.shape}')
            # Check ready flags are zeroed
            print(f'     ready flags all zero: {(sym_buffer.ready == 0).all().item()}')

        # Step 3: Fused GEMM-RS
        y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        if rank_idx == 0:
            print(f'[...] Running bf16_gemm_rs_nt...')
        
        dist.barrier()
        deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
        torch.cuda.synchronize(local_rank)
        dist.barrier()

        if rank_idx == 0:
            max_diff = (y_fused.float() - y_ref.float()).abs().max().item()
            mean_diff = (y_fused.float() - y_ref.float()).abs().mean().item()
            print(f'[OK] bf16_gemm_rs_nt succeeded!')
            print(f'     y_fused mean={y_fused.float().mean().item():.4f}')
            print(f'     max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}')
            if max_diff < 2.0:
                print(f'     ✅ CORRECTNESS CHECK PASSED!')
            else:
                print(f'     ❌ CORRECTNESS CHECK FAILED!')

        sym_buffer.destroy()
        dist.barrier()

    except Exception as e:
        print(f'Rank {rank_idx}: ❌ ERROR: {e}')
        traceback.print_exc()

    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f'Running minimal GEMM-RS test with {num_gpus} GPUs...')
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
