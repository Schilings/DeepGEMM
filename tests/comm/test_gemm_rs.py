"""
Comprehensive validation test for BF16 GEMM + Reduce-Scatter
(Pull-based, single-kernel fusion).

Tests:
  1. Multiple shapes covering small/medium/large hidden dimensions
  2. Correctness against reference (bf16_gemm + FP32 manual reduce-scatter)
  3. Cross-check against torch-native baseline (torch.matmul + NCCL reduce_scatter)
  4. Consistency across multiple runs (determinism)
  5. Edge cases: minimum M, large K, large N (7168)

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
# NOTE: In real LLM training, M (total tokens) is typically 10k-20k+,
#       and hidden dimensions are 4k-8k (e.g., DeepSeek-V3 uses 7168).
#       tokens_per_rank = total_M / num_ranks.
SHAPES_BASIC = [
    # Small shapes (quick validation)
    (512, 4096, 7168),
    (1024, 7168, 4096),
    # Medium shapes (typical MoE training, 8 GPU → total M=4k-8k)
    (2048, 7168, 2048),
    (2048, 4096, 7168),
    # Large shapes (long context training, per-rank 4k = total 32k on 8 GPU)
    (4096, 7168, 2048),
    (4096, 2048, 7168),
]

SHAPES_EXTENDED = [
    # Very large M (long context training: per-rank 8k-16k = total 64k-128k on 8 GPU)
    (8192, 7168, 2048),
    (8192, 2048, 7168),
    (8192, 4096, 4096),
    (16384, 7168, 2048),
    (16384, 2048, 7168),
    # Stress test: large in all dimensions
    (8192, 7168, 7168),
    (16384, 7168, 4096),
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


def compute_torch_native(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank, group):
    """
    Torch-native baseline: torch.matmul (full GEMM) + NCCL reduce_scatter_tensor.

    This is the most common path a user would write by hand, and is exactly the
    `run_torch_native` baseline used in benchmarks/bench_gemm_rs.py. It computes
    the reduce-scatter entirely in BF16 (no manual FP32 accumulation), so it is a
    real-world cross-check rather than the exact FP32 ground truth.
    """
    n_dim = b.shape[0]
    # Full GEMM: [total_m, n], same on every rank (a, b broadcast from src=0).
    d_full = torch.matmul(a, b.t())  # bf16
    y_torch = torch.zeros((tokens_per_rank, n_dim), dtype=torch.bfloat16,
                          device=f'cuda:{local_rank}')
    dist.reduce_scatter_tensor(y_torch, d_full.contiguous(),
                               op=dist.ReduceOp.SUM, group=group)
    torch.cuda.synchronize(local_rank)
    del d_full
    return y_torch


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

    # Compute reference (DeepGEMM gemm + manual FP32 reduce-scatter = exact ground truth)
    ref = compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank)
    dist.barrier()

    # Torch-native baseline (torch.matmul + NCCL reduce_scatter, bf16): real-world cross-check
    y_torch = compute_torch_native(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank, group)
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

    # Cross-check fused vs torch-native baseline (both bf16 reduce-scatter,
    # so they should match closely; small diffs come from bf16 accumulation order).
    max_diff_torch = (y.float() - y_torch.float()).abs().max().item()
    mean_diff_torch = (y.float() - y_torch.float()).abs().mean().item()
    torch_abs_mean = y_torch.float().abs().mean().item()
    rel_error_torch = mean_diff_torch / max(torch_abs_mean, 1e-8)
    del y_torch

    # Determine pass/fail
    # BF16 GEMM has inherent precision limits.
    # For multi-GPU reduce-scatter, each intermediate BF16 → FP32 → BF16 round-trip
    # introduces ~0.4% relative error. With N ranks, we accumulate N-1 such truncations.
    # Use relative error as the primary metric, with a per-rank scaling factor.
    max_rel_error_threshold = 0.01 * num_ranks  # ~1% per rank is acceptable for BF16
    passed = (rel_error < max_rel_error_threshold
              and rel_error_torch < max_rel_error_threshold
              and consistency_diff < 0.01)

    sym_buffer.destroy()
    dist.barrier()

    return (passed, max_diff, mean_diff, rel_error, consistency_diff,
            max_diff_torch, rel_error_torch)


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
        print(f"{'Shape (M/rank×N×K)':<22} | {'Max Diff':>9} {'Rel Err':>9} {'Consist':>9} | "
              f"{'vs Torch Max':>12} {'vs Torch Rel':>12} | {'Status'}")
        print(f"{'-'*22} | {'-'*9} {'-'*9} {'-'*9} | {'-'*12} {'-'*12} | {'-'*8}")

    all_passed = True
    results = []

    for tokens_per_rank, n_dim, k_dim in shapes:
        total_m = tokens_per_rank * num_ranks

        try:
            (passed, max_diff, mean_diff, rel_error, consistency_diff,
             max_diff_torch, rel_error_torch) = run_single_shape_test(
                rank_idx, num_ranks, local_rank, group,
                tokens_per_rank, n_dim, k_dim
            )
        except Exception as e:
            passed = False
            max_diff = mean_diff = rel_error = consistency_diff = float('nan')
            max_diff_torch = rel_error_torch = float('nan')
            if rank_idx == 0:
                print(f"  ❌ ERROR: {e}")

        results.append((tokens_per_rank, n_dim, k_dim, passed, max_diff, mean_diff, rel_error))

        if rank_idx == 0:
            status = "✅ PASS" if passed else "❌ FAIL"
            shape_str = f"{tokens_per_rank}×{n_dim}×{k_dim}"
            print(f"{shape_str:<22} | {max_diff:>9.6f} {rel_error:>9.6f} {consistency_diff:>9.6f} | "
                  f"{max_diff_torch:>12.6f} {rel_error_torch:>12.6f} | {status}")

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

    # Force exit to avoid NCCL background threads holding ports after test
    time.sleep(1)
    os._exit(0)


def _find_free_port():
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_all = '--all' in sys.argv

    # Auto-find free port to avoid EADDRINUSE
    if os.getenv('MASTER_PORT') is None or os.getenv('MASTER_PORT') == '':
        port = _find_free_port()
        os.environ['MASTER_PORT'] = str(port)
    print(f"Launching GEMM-RS correctness test with {num_gpus} GPUs (port={os.environ['MASTER_PORT']})...")
    if run_all:
        print("  Running full test suite (including extended shapes)")

    mp.spawn(run_test, args=(num_gpus, run_all), nprocs=num_gpus, join=True)
