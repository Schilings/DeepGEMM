"""
Debug: Check if the GEMM output in partial buffer matches standalone GEMM.
We use 1 GPU (no reduce) to isolate the GEMM+Epilogue path.
"""
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['DG_JIT_DEBUG'] = '0'

import deep_gemm
from deep_gemm.utils.dist import init_dist


def run(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'

    # Simple shape
    tokens_per_rank = 128
    n_dim = 128
    k_dim = 128
    total_m = tokens_per_rank * num_ranks

    # Deterministic
    torch.manual_seed(42)
    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
    dist.broadcast(a, src=0)
    dist.broadcast(b, src=0)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Reference: standard GEMM
    d_ref = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
    deep_gemm.bf16_gemm_nt(a, b, d_ref)
    torch.cuda.synchronize(local_rank)

    # GEMM-RS kernel
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
    )
    dist.barrier()

    y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
    deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # For 2 GPUs: each rank computes the full M×N GEMM, then writes tiles into partial buffer.
    # Rank 0's partial buffer, slot 0 should contain the GEMM output for:
    #   - tiles belonging to rank 0's chunk (rows 0:tokens_per_rank) — these are rank 0's own contribution
    # Rank 0's partial buffer, slot 1 doesn't exist conceptually here.
    #
    # Actually with M-swizzle:
    #   Rank 0 first computes rank 1's chunk (rows 128:256), writes to slot 0 at local positions 0:128
    #   Then computes its own chunk (rows 0:128), writes to slot 0 at local positions 0:128
    #
    # Wait... let me re-read the design. Each rank computes ALL tiles and writes each tile
    # to its own slot. The slot index is always rank_idx (itself).
    # The Comm warp on rank Y then pulls from all ranks' slot=src_rank, tile for rank Y's chunk.
    #
    # So rank 0's buffer slot 0 should contain:
    #   Row i, col j = GEMM result at (i, j) of the full matrix
    #   But which rows? The workspace has num_max_tokens_per_rank rows per slot.
    #   Each rank writes to slot=rank_idx, so rank 0 writes to slot 0.
    #   The Epilogue writes at (local_m_block_idx * BLOCK_M + row, base_col).
    #   local_m_block_idx = m_block_idx - dst_rank * num_m_blocks_per_rank
    #
    # For rank 0, when computing tile for rank 1 (m_block_idx = 1, dst_rank=1):
    #   local_m_block_idx = 1 - 1*1 = 0  → writes to rows 0:128 of slot 0
    # Then when computing tile for rank 0 (m_block_idx = 0, dst_rank=0):
    #   local_m_block_idx = 0 - 0*1 = 0  → ALSO writes to rows 0:128 of slot 0!
    #
    # BUG! Both tiles overwrite the same location!
    # The first tile (for rank 1) gets overwritten by the second tile (for rank 0).
    # So the partial buffer ends up only having rank 0's own contribution.
    #
    # Let me verify this hypothesis:

    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"  Checking GEMM+Epilogue correctness")
        print(f"  total_m={total_m}, n={n_dim}, k={k_dim}, num_ranks={num_ranks}")
        print(f"  tokens_per_rank={tokens_per_rank}")
        print(f"{'='*60}\n")

        # y_fused is the final output after Comm warps reduce.
        # But let's check the output directly.
        
        # If only rank 0's own contribution is used (1 copy from itself):
        my_chunk_ref = d_ref[:tokens_per_rank, :n_dim]  # rank 0's computation of row 0:128
        other_chunk_ref = d_ref[tokens_per_rank:2*tokens_per_rank, :n_dim]  # rank 0's computation of row 128:256
        
        # Reference RS for 2 ranks: y_ref = d_ref_rank0[0:128] + d_ref_rank1[0:128]
        # Since both ranks have same a,b → d_ref_rank0 = d_ref_rank1 → y_ref = 2 * d_ref[0:128]
        y_ref_expected = 2 * my_chunk_ref
        
        print(f"  d_ref[0:128] stats: mean={my_chunk_ref.float().mean():.4f}, std={my_chunk_ref.float().std():.4f}")
        print(f"  d_ref[128:256] stats: mean={other_chunk_ref.float().mean():.4f}, std={other_chunk_ref.float().std():.4f}")
        print(f"  y_fused stats: mean={y_fused.float().mean():.4f}, std={y_fused.float().std():.4f}")
        print(f"  2*d_ref[0:128] stats: mean={y_ref_expected.float().mean():.4f}, std={y_ref_expected.float().std():.4f}")
        
        # Check if y_fused ≈ 1 * d_ref[0:128] (only self contribution)
        diff_1x = (y_fused.float() - my_chunk_ref.float()).abs()
        print(f"\n  y_fused vs 1×d_ref[0:128] (only self):")
        print(f"    max_diff: {diff_1x.max():.4f}, mean_diff: {diff_1x.mean():.4f}")
        
        # Check if y_fused ≈ 2 * d_ref[0:128] (both contributions, correct RS)
        diff_2x = (y_fused.float() - y_ref_expected.float()).abs()
        print(f"\n  y_fused vs 2×d_ref[0:128] (correct RS):")
        print(f"    max_diff: {diff_2x.max():.4f}, mean_diff: {diff_2x.mean():.4f}")
        
        # Check if y_fused ≈ d_ref[128:256] (other rank's chunk — swizzle confusion)
        diff_other = (y_fused.float() - other_chunk_ref.float()).abs()
        print(f"\n  y_fused vs d_ref[128:256] (wrong chunk):")
        print(f"    max_diff: {diff_other.max():.4f}, mean_diff: {diff_other.mean():.4f}")
        
        # Check sum
        sum_check = (y_fused.float() - (my_chunk_ref.float() + other_chunk_ref.float())).abs()
        print(f"\n  y_fused vs d_ref[0:128]+d_ref[128:256] (sum of all chunks):")
        print(f"    max_diff: {sum_check.max():.4f}, mean_diff: {sum_check.mean():.4f}")
        
        print(f"\n  First 4 elements:")
        print(f"    d_ref[0,0:4]:       {d_ref[0,:4].tolist()}")
        print(f"    d_ref[128,0:4]:     {d_ref[128,:4].tolist()}")
        print(f"    y_fused[0,0:4]:     {y_fused[0,:4].tolist()}")
        print(f"    sum[0,0:4]:         {(d_ref[0,:4].float() + d_ref[128,:4].float()).tolist()}")
        
        print(f"\n{'='*60}\n")

    sym_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
