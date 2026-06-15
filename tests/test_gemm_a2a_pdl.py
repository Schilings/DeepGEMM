"""
Test for GEMM + A2A + PDL Local Reduce approach.

The key insight: in ReduceScatter for MoE, all ranks have identical A (after AllGather)
and identical B (expert weight). So:
  ReduceScatter(C) where C = A × B^T, all ranks have same C
  = sum across ranks (all identical) → num_ranks * C, then scatter
  = num_ranks * C[my_rows, :]

This means we can just compute GEMM for our rows only and multiply by num_ranks.
No communication needed at all!

Usage:
    python tests/test_gemm_a2a_pdl.py [num_gpus]
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


SHAPES = [
    (512, 4096, 7168),
    (1024, 7168, 4096),
    (2048, 7168, 2048),
    (2048, 4096, 7168),
    (4096, 7168, 2048),
    (4096, 2048, 7168),
]


def test_worker(local_rank, num_local_ranks):
    from deep_gemm.utils.dist import init_dist
    import deep_gemm

    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    device = f'cuda:{local_rank}'
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    if rank_idx == 0:
        print(f"\n  GEMM + A2A + PDL Reduce Test: {num_ranks} GPUs")
        print(f"  Key insight: all ranks have same A,B → ReduceScatter = num_ranks * C[my_rows]")
        print()

    all_pass = True
    for tokens_per_rank, n, k in SHAPES:
        total_m = tokens_per_rank * num_ranks

        # Create data (same on all ranks, as in MoE after AllGather)
        a = torch.randn((total_m, k), dtype=torch.bfloat16, device=device)
        b = torch.randn((n, k), dtype=torch.bfloat16, device=device)

        # ── Reference: manual ReduceScatter ──
        # GEMM: full C = A × B^T
        c_full = torch.empty((total_m, n), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_nt(a, b, c_full)

        # ReduceScatter: sum across ranks (all identical) + scatter
        # = num_ranks * C[my_rows]
        start_row = rank_idx * tokens_per_rank
        end_row = start_row + tokens_per_rank
        ref = c_full[start_row:end_row, :] * num_ranks

        # ── Test: A2A PDL approach (compute only our rows) ──
        my_a = a[start_row:end_row, :]
        y_test = torch.empty((tokens_per_rank, n), dtype=torch.bfloat16, device=device)
        deep_gemm.bf16_gemm_nt(my_a, b, y_test)
        y_test.mul_(num_ranks)

        # ── Compare ──
        max_diff = (ref - y_test).abs().max().item()
        ref_max = ref.abs().max().item()
        rel_err = max_diff / ref_max if ref_max > 0 else 0

        passed = max_diff <= 1  # BF16 tolerance
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed

        if rank_idx == 0:
            print(f"  {tokens_per_rank:>5}x{n:>5}x{k:<5} | max_diff={max_diff:>8.4f}  "
                  f"rel_err={rel_err:.6f}  ref_max={ref_max:.4f} | {status}")

    if rank_idx == 0:
        print(f"\n  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mp.spawn(test_worker, args=(num_gpus,), nprocs=num_gpus, join=True)
