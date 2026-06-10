"""
Comprehensive validation test for BF16 GEMM + Reduce-Scatter
(Pull-based, single-kernel fusion).

Tests:
  1. Multiple shapes covering small/medium/large hidden dimensions
  2. Correctness against reference (bf16_gemm + FP32 manual reduce-scatter)
  3. Consistency across multiple runs (determinism)
  4. Edge cases: minimum M, large K, large N (7168)

Usage:
    python tests/test_gemm_rs.py [num_gpus]        # default: 2
    python tests/test_gemm_rs.py [num_gpus] --all   # run all shapes (slow)
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.utils.dist import init_dist


# ─── Test shapes ───
# (tokens_per_rank, N, K) — designed to cover typical LLM training scenarios
SHAPES_BASIC = [
    # Small shapes (quick validation)
    (256, 512, 1024),
    (256, 1024, 2048),
    # Medium shapes (typical MoE / attention intermediate)
    (512, 2048, 4096),
    (1024, 2048, 4096),
    # Large hidden dim (7168 = DeepSeek-V3 style)
    (256, 7168, 2048),
    (512, 2048, 7168),
]

SHAPES_EXTENDED = [
    # Very large M (long context training)
    (2048, 2048, 4096),
    (4096, 4096, 4096),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
    # Stress test: large in all dimensions
    (2048, 7168, 7168),
]


def compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank):
    """
    Compute reference result using bf16_gemm_nt + FP32 manual reduce-scatter.

    This mimics what NCCL reduce_scatter does: each rank gets the sum
    of all ranks' contributions to its chunk.
    """
    total_m = tokens_per_rank * num_ranks
    n_dim = b.shape[0]

    # Full GEMM (all ranks compute the same full result)
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    torch.cuda.synchronize(local_rank)

    # All-gather full results from all ranks
    all_d_fulls = [torch.empty_like(d_full) for _ in range(num_ranks)]
    dist.all_gather(all_d_fulls, d_full)
    torch.cuda.synchronize(local_rank)

    # Reduce-scatter: each rank gets sum of all contributions to its chunk
    start_row = rank_idx * tokens_per_rank
    end_row = start_row + tokens_per_rank
    ref_fp32 = torch.zeros((tokens_per_rank, n_dim), dtype=torch.float32, device=f'cuda:{local_rank}')
    for r in range(num_ranks):
        ref_fp32 += all_d_fulls[r][start_row:end_row, :].float()
    ref = ref_fp32.bfloat16()

    del all_d_fulls, d_full, ref_fp32
    return ref


def run_single_shape_test(rank_idx, num_ranks, local_rank, group,
                          tokens_per_rank, n_dim, k_dim,
                          verbose=True):
    """Run correctness test for a single shape. Returns (passed, max_diff, mean_diff)."""
    total_m = tokens_per_rank * num_ranks
    max_tokens_per_rank = tokens_per_rank

    # Create test data (same across ranks via broadcast)
    a = torch.randn((total_m, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    dist.broadcast(a, src=0)
    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    # Note: b is NOT broadcast — each rank uses same random seed but different b
    # Actually for correctness test, b should be the same across ranks
    dist.broadcast(b, src=0)

    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Compute reference
    ref = compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank)
    dist.barrier()

    # Create symmetric buffer
    sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
        group, max_tokens_per_rank, n_dim, out_dtype=torch.bfloat16
    )

    # Run fused kernel
    y = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_rs_nt(y, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Run again for consistency check
    y2 = torch.zeros_like(y)
    deep_gemm.bf16_gemm_rs_nt(y2, a, b, sym_buffer, tokens_per_rank, compiled_dims='nk')
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    # Compute differences
    consistency_diff = (y.float() - y2.float()).abs().max().item()
    max_diff = (y.float() - ref.float()).abs().max().item()
    mean_diff = (y.float() - ref.float()).abs().mean().item()

    # Relative error (more meaningful for varying magnitudes)
    ref_abs_mean = ref.float().abs().mean().item()
    rel_error = mean_diff / max(ref_abs_mean, 1e-8)

    # Determine pass/fail
    # BF16 GEMM has inherent precision limits.
    # For multi-GPU reduce-scatter, each intermediate BF16 → FP32 → BF16 round-trip
    # introduces ~0.4% relative error. With N ranks, we accumulate N-1 such truncations.
    # Use relative error as the primary metric, with a per-rank scaling factor.
    max_rel_error_threshold = 0.01 * num_ranks  # ~1% per rank is acceptable for BF16
    passed = rel_error < max_rel_error_threshold and consistency_diff < 0.01

    sym_buffer.destroy()
    dist.barrier()

    return passed, max_diff, mean_diff, rel_error, consistency_diff


def run_test(local_rank: int, num_local_ranks: int, run_all: bool = False):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(42 + rank_idx)
    torch.cuda.manual_seed(42 + rank_idx)

    shapes = SHAPES_BASIC + (SHAPES_EXTENDED if run_all else [])

    if rank_idx == 0:
        print(f"\n{'='*80}")
        print(f"  BF16 GEMM-RS Correctness Test (Pull-based): {num_ranks} GPUs")
        print(f"  Testing {len(shapes)} shapes {'(full suite)' if run_all else '(basic suite)'}")
        print(f"{'='*80}\n")
        print(f"{'Shape (M/rank×N×K)':<22} | {'Max Diff':>9} {'Mean Diff':>10} {'Rel Err':>9} {'Consist':>9} | {'Status'}")
        print(f"{'-'*22} | {'-'*9} {'-'*10} {'-'*9} {'-'*9} | {'-'*8}")

    all_passed = True
    results = []

    for tokens_per_rank, n_dim, k_dim in shapes:
        total_m = tokens_per_rank * num_ranks

        try:
            passed, max_diff, mean_diff, rel_error, consistency_diff = run_single_shape_test(
                rank_idx, num_ranks, local_rank, group,
                tokens_per_rank, n_dim, k_dim
            )
        except Exception as e:
            passed = False
            max_diff = mean_diff = rel_error = consistency_diff = float('nan')
            if rank_idx == 0:
                print(f"  ❌ ERROR: {e}")

        results.append((tokens_per_rank, n_dim, k_dim, passed, max_diff, mean_diff, rel_error))

        if rank_idx == 0:
            status = "✅ PASS" if passed else "❌ FAIL"
            shape_str = f"{tokens_per_rank}×{n_dim}×{k_dim}"
            print(f"{shape_str:<22} | {max_diff:>9.6f} {mean_diff:>10.7f} {rel_error:>9.6f} {consistency_diff:>9.6f} | {status}")

        if not passed:
            all_passed = False

        dist.barrier()

    # Summary
    if rank_idx == 0:
        num_passed = sum(1 for r in results if r[3])
        num_total = len(results)
        print(f"\n{'='*80}")
        print(f"  Summary: {num_passed}/{num_total} shapes passed")
        if all_passed:
            print(f"  ✅ ALL TESTS PASSED!")
        else:
            print(f"  ❌ SOME TESTS FAILED!")
            print(f"  Failed shapes:")
            for tokens_per_rank, n_dim, k_dim, passed, max_diff, _, _, in results:
                if not passed:
                    print(f"    - {tokens_per_rank}×{n_dim}×{k_dim} (max_diff={max_diff:.6f})")
        print(f"{'='*80}\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_all = '--all' in sys.argv

    print(f"Launching GEMM-RS correctness test with {num_gpus} GPUs...")
    if run_all:
        print("  Running full test suite (including extended shapes)")

    mp.spawn(run_test, args=(num_gpus, run_all), nprocs=num_gpus, join=True)
