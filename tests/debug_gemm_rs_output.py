"""
Debug test: Inspect what the GEMM-RS kernel actually produces.
Check if the GEMM part is correct by examining the partial buffer.
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

    # Use a simple shape
    tokens_per_rank = 128
    n_dim = 128
    k_dim = 128
    total_m = tokens_per_rank * num_ranks

    # Deterministic data
    torch.manual_seed(42)
    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=device)
    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
    dist.broadcast(a, src=0)
    dist.broadcast(b, src=0)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Step 1: Reference GEMM (full matrix multiply)
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=device)
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    torch.cuda.synchronize(local_rank)

    # Reference reduce-scatter
    y_ref = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
    dist.reduce_scatter_tensor(y_ref, d_full, op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Step 2: Create symmetric buffer and run fused kernel
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, tokens_per_rank, n_dim, out_dtype=torch.bfloat16
    )
    dist.barrier()

    y_fused = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
    deep_gemm.bf16_gemm_rs_nt(y_fused, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Step 3: Analyze results
    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"  Shape: tokens_per_rank={tokens_per_rank}, N={n_dim}, K={k_dim}")
        print(f"  num_ranks={num_ranks}")
        print(f"{'='*60}")

        # Check GEMM reference output
        print(f"\n  d_full (full GEMM output) stats:")
        print(f"    shape: {d_full.shape}")
        print(f"    mean:  {d_full.float().mean().item():.6f}")
        print(f"    std:   {d_full.float().std().item():.6f}")
        print(f"    min:   {d_full.float().min().item():.6f}")
        print(f"    max:   {d_full.float().max().item():.6f}")

        # Check reference RS output
        print(f"\n  y_ref (reference RS) stats:")
        print(f"    shape: {y_ref.shape}")
        print(f"    mean:  {y_ref.float().mean().item():.6f}")
        print(f"    std:   {y_ref.float().std().item():.6f}")

        # Check fused output
        print(f"\n  y_fused (GEMM-RS kernel) stats:")
        print(f"    shape: {y_fused.shape}")
        print(f"    mean:  {y_fused.float().mean().item():.6f}")
        print(f"    std:   {y_fused.float().std().item():.6f}")
        print(f"    all zeros: {(y_fused == 0).all().item()}")
        print(f"    fraction zeros: {(y_fused == 0).float().mean().item():.4f}")

        # Compare
        diff = (y_fused.float() - y_ref.float()).abs()
        print(f"\n  Difference (y_fused - y_ref):")
        print(f"    max_diff:  {diff.max().item():.6f}")
        print(f"    mean_diff: {diff.mean().item():.6f}")

        # Check partial buffer contents
        print(f"\n  Symmetric buffer info:")
        print(f"    buffer shape:  {sym_buffer.buffer.shape}")
        print(f"    buffer dtype:  {sym_buffer.buffer.dtype}")
        partial = sym_buffer.buffer
        print(f"    partial mean:  {partial.float().mean().item():.6f}")
        print(f"    partial std:   {partial.float().std().item():.6f}")
        print(f"    partial zeros: {(partial == 0).float().mean().item():.4f}")

        # Check if partial matches the GEMM output for this rank's chunk
        # The partial buffer for rank 0 should contain the GEMM output for rows 0:tokens_per_rank
        partial_slice = partial[:tokens_per_rank, :n_dim] if partial.dim() == 2 else partial.view(-1)[:tokens_per_rank*n_dim].view(tokens_per_rank, n_dim)
        gemm_my_chunk = d_full[:tokens_per_rank, :n_dim]
        partial_vs_gemm = (partial_slice.float() - gemm_my_chunk.float()).abs()
        print(f"\n  partial[:tpr, :n] vs d_full[:tpr, :n]:")
        print(f"    max_diff:  {partial_vs_gemm.max().item():.6f}")
        print(f"    mean_diff: {partial_vs_gemm.mean().item():.6f}")

        # Check ready flags
        if hasattr(sym_buffer, 'ready'):
            ready = sym_buffer.ready
            print(f"\n  Ready flags:")
            print(f"    shape: {ready.shape}")
            print(f"    sum:   {ready.sum().item()}")
            print(f"    all 1: {(ready == 1).all().item()}")

        # Check first few elements
        print(f"\n  First 8 elements comparison:")
        print(f"    y_ref[0,:8]:   {y_ref[0,:8].tolist()}")
        print(f"    y_fused[0,:8]: {y_fused[0,:8].tolist()}")
        print(f"    d_full[0,:8]:  {d_full[0,:8].tolist()}")

        print(f"\n{'='*60}\n")

    sym_buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(run, args=(num_gpus,), nprocs=num_gpus, join=True)
