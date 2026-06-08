"""
Lightweight validation test for FP8 GEMM + Reduce-Scatter operator (2~8 GPUs).
Compares fp8_gemm_rs_nt kernel with fp8_gemm_nt + FP32 manual reduce-scatter.
Usage: python test_gemm_rs_fp8.py [num_gpus]  (default: 2)
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist
from deep_gemm.utils.math import per_token_cast_to_fp8


def run_test(local_rank: int, num_local_ranks: int):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    # ── Config ──
    tokens_per_rank = 256
    k_dim = 1024
    n_dim = 512
    max_tokens_per_rank = tokens_per_rank
    total_m = tokens_per_rank * num_ranks
    gran_k = 128

    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"FP8 GEMM-RS Test: {num_ranks} GPUs")
        print(f"  M_per_rank={tokens_per_rank}, K={k_dim}, N={n_dim}, gran_k={gran_k}")
        print(f"{'='*60}\n")

    # ── Create test data ──
    # A: [num_ranks * tokens_per_rank, k_dim] — same across all ranks (BF16 source)
    a_bf16 = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(a_bf16, src=0)

    # B: [n_dim, k_dim] — NT layout, each rank has different weights (BF16 source)
    b_bf16 = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')

    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # ── Quantize to FP8 (per-token for A, per-token for B in NT layout) ──
    # SM100 requires UE8M0 scale factors (packed int format)
    a_fp8, a_sf = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=gran_k)
    b_fp8, b_sf = per_token_cast_to_fp8(b_bf16, use_ue8m0=True, gran_k=gran_k)

    # ── Reference: fp8_gemm_nt (full GEMM) + FP32 manual reduce_scatter ──
    # Compute full GEMM on all ranks
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d_full, recipe=(1, 1, gran_k))
    torch.cuda.synchronize(local_rank)

    # Gather all ranks' GEMM results and reduce in FP32
    all_d_fulls = [torch.empty_like(d_full) for _ in range(num_ranks)]
    dist.all_gather(all_d_fulls, d_full)
    torch.cuda.synchronize(local_rank)

    start_row = rank_idx * tokens_per_rank
    end_row = start_row + tokens_per_rank
    ref_fp32 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    for r in range(num_ranks):
        ref_fp32 += all_d_fulls[r][start_row:end_row, :].float()
    ref = ref_fp32.bfloat16()
    del all_d_fulls, d_full, ref_fp32
    dist.barrier()

    # ── Create symmetric buffer ──
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
    )

    # ── Warm-up (JIT compilation) ──
    y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    if rank_idx == 0:
        print(">>> Phase 1: Warm-up (JIT compilation)...")
    deep_gemm.fp8_gemm_rs_nt(y, (a_fp8, a_sf), (b_fp8, b_sf), sym_buffer,
                              tokens_per_rank, recipe=(1, 1, gran_k),
                              compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # ── Consistency check ──
    if rank_idx == 0:
        print(">>> Phase 2: Second run for consistency check...")
    y2 = torch.zeros_like(y)
    deep_gemm.fp8_gemm_rs_nt(y2, (a_fp8, a_sf), (b_fp8, b_sf), sym_buffer,
                              tokens_per_rank, recipe=(1, 1, gran_k),
                              compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    warmup_diff = (y.float() - y2.float()).abs().max().item()
    if rank_idx == 0:
        print(f"  Consistency check: max_diff={warmup_diff:.6f}")
        if warmup_diff < 0.01:
            print(f"  ✅ Kernel produces consistent results across runs")
        else:
            print(f"  ❌ Inconsistent results!")

    # ── Compare with reference ──
    if rank_idx == 0:
        print(">>> Phase 3: Comparing with reference (fp8_gemm_nt + reduce_scatter)...")

    diff = (y.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if rank_idx == 0:
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Max abs diff:  {max_diff:.6f}")
        print(f"  Mean abs diff: {mean_diff:.6f}")
        if max_diff < 1.0:
            print(f"  ✅ PASS — FP8 GEMM-RS matches reference!")
        elif max_diff < 5.0:
            print(f"  ⚠️  Close but check numerical precision")
        else:
            print(f"  ❌ FAIL — Large difference")
            # Show some samples
            print(f"  y[0,0:4] = {y[0, 0:4].tolist()}")
            print(f"  ref[0,0:4] = {ref[0, 0:4].tolist()}")

    print(f"    [Rank {rank_idx}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # ── Test with FP32 communication (optional) ──
    if rank_idx == 0:
        print("\n>>> Phase 4: Testing with FP32 communication dtype...")

    # Build a proper FP32-path reference:
    # fp8_gemm_nt with FP32 output → all_gather → FP32 reduce → BF16
    # This matches the numerical path: FP32 acc → FP32 push → FP32 reduce → BF16
    d_full_fp32 = torch.zeros((total_m, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d_full_fp32, recipe=(1, 1, gran_k))
    torch.cuda.synchronize(local_rank)

    all_d_fp32 = [torch.empty_like(d_full_fp32) for _ in range(num_ranks)]
    dist.all_gather(all_d_fp32, d_full_fp32)
    torch.cuda.synchronize(local_rank)

    ref_fp32_comm = torch.zeros((tokens_per_rank, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    for r in range(num_ranks):
        ref_fp32_comm += all_d_fp32[r][start_row:end_row, :]
    ref_fp32_comm_bf16 = ref_fp32_comm.bfloat16()
    del all_d_fp32, d_full_fp32, ref_fp32_comm

    # Create a new sym_buffer with FP32 comm
    sym_buffer_fp32 = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16,
        comm_dtype=torch.float32
    )

    y_fp32_comm = torch.zeros_like(y)
    deep_gemm.fp8_gemm_rs_nt(y_fp32_comm, (a_fp8, a_sf), (b_fp8, b_sf), sym_buffer_fp32,
                              tokens_per_rank, recipe=(1, 1, gran_k),
                              compiled_dims='nk',
                              comm_dtype='fp32', reduce_in_fp32=True)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    diff_fp32 = (y_fp32_comm.float() - ref_fp32_comm_bf16.float()).abs()
    max_diff_fp32 = diff_fp32.max().item()
    mean_diff_fp32 = diff_fp32.mean().item()

    if rank_idx == 0:
        print(f"  FP32 comm: max_diff={max_diff_fp32:.6f}, mean_diff={mean_diff_fp32:.6f}")
        if max_diff_fp32 < 1.0:
            print(f"  ✅ FP32 communication matches FP32-path reference!")
        else:
            print(f"  ❌ FP32 communication has unexpected difference")

    sym_buffer.destroy()
    sym_buffer_fp32.destroy()
    dist.barrier()

    if rank_idx == 0:
        print(f"{'='*60}")
        print("Test complete.")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"Launching test with {num_gpus} GPUs...")
    mp.spawn(run_test, args=(num_gpus,), nprocs=num_gpus, join=True)
