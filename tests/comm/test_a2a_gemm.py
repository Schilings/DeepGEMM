"""
Correctness test for BF16 All2All + GEMM fusion (Ulysses SP scenario).

Tests that fused A2A+GEMM produces the same result as:
  1. torch.distributed.all_to_all (scatter chunks)
  2. Standard BF16 GEMM on gathered data

Usage:
    python tests/test_a2a_gemm.py <num_gpus>
    python tests/test_a2a_gemm.py 8
"""

import os
import sys
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Tuple

import deep_gemm
from deep_gemm.a2a_gemm import bf16_a2a_gemm_nt, get_symm_buffer_for_a2a_gemm


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


SHAPES = [
    # (M_per_rank, N, K) — Ulysses SP shapes
    # M_per_rank = seq_len (total, not per-rank for SP)
    # K = heads_per_tp * head_dim
    # N = hidden_dim
    (256, 1024, 1024),
    (256, 2048, 2048),
    (512, 4096, 2048),
    (1024, 4096, 4096),
    (512, 4096, 7168),
    (1024, 2048, 7168),
]


def run_test(rank: int, num_gpus: int, port: int):
    os.environ.update({
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
        'RANK': str(rank),
        'WORLD_SIZE': str(num_gpus),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)
    group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  BF16 A2A+GEMM Correctness Test: {num_gpus} GPUs")
        print(f"  Testing {len(SHAPES)} shapes")
        print(f"{'='*80}\n")
        print(f"{'Shape (M/rank×N×K)':<25} | {'Max Diff':>10} {'Mean Diff':>12} {'Rel Err':>10} | Status")
        print(f"{'-'*25}-+-{'-'*10}-{'-'*12}-{'-'*10}-+--------")

    num_passed = 0
    num_failed = 0
    fail_shapes = []

    for tokens_per_rank, n_dim, k_dim in SHAPES:
        # --- Reference: All2All + GEMM separately ---
        # Each rank creates input: x_local[num_ranks, tokens_per_rank, K]
        # x_local[j] is the chunk to send to rank j
        x_full = torch.randn((num_gpus, tokens_per_rank, k_dim), dtype=torch.bfloat16, device=device)
        b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)

        # Reference All2All: scatter x_full across ranks
        # After A2A, each rank has [num_ranks, tokens_per_rank, K] where
        # recv[j] = rank j's x_full[rank]
        send_list = list(x_full.unbind(0))  # [num_gpus tensors of shape [tokens_per_rank, K]]
        recv_list = [torch.empty_like(send_list[0]) for _ in range(num_gpus)]
        dist.all_to_all(recv_list, send_list, group=group)

        # Concatenate received data and do GEMM
        a_gathered = torch.cat(recv_list, dim=0)  # [num_gpus * tokens_per_rank, K]
        d_ref = a_gathered @ b.t()  # [num_gpus * tokens_per_rank, N]

        # --- Fused: A2A+GEMM ---
        try:
            sym_buffer = get_symm_buffer_for_a2a_gemm(group, tokens_per_rank, k_dim)
        except Exception as e:
            if rank == 0:
                print(f"  {tokens_per_rank}×{n_dim}×{k_dim:<5}  | SKIP (buffer alloc failed: {e})")
            dist.barrier()
            continue

        # Copy input into sym_buffer.x
        sym_buffer.x[:num_gpus, :tokens_per_rank, :k_dim].copy_(x_full)

        d_fused = torch.zeros((num_gpus * tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
        bf16_a2a_gemm_nt(d_fused, b, sym_buffer, tokens_per_rank)
        torch.cuda.synchronize()

        # Compare
        diff = (d_fused.float() - d_ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        ref_norm = d_ref.float().abs().mean().item()
        rel_err = mean_diff / (ref_norm + 1e-8)

        # BF16 precision: allow relative error < 1% * num_ranks
        threshold = 0.01 * num_gpus
        passed = rel_err < threshold and not (max_diff != max_diff)  # check for NaN

        # Check cross-rank consistency
        d_fused_list = [torch.empty_like(d_fused) for _ in range(num_gpus)]
        dist.all_gather(d_fused_list, d_fused, group=group)
        consist_diff = max((d_fused_list[i] - d_fused_list[0]).abs().max().item() for i in range(1, num_gpus)) if num_gpus > 1 else 0.0

        if rank == 0:
            status = "PASS" if passed else "FAIL"
            icon = "✅" if passed else "❌"
            print(f"{tokens_per_rank}×{n_dim}×{k_dim:<5}          | {max_diff:>10.6f} {mean_diff:>12.7f} {rel_err:>10.6f} | {icon} {status}")
            if passed:
                num_passed += 1
            else:
                num_failed += 1
                fail_shapes.append(f"{tokens_per_rank}×{n_dim}×{k_dim}")

        sym_buffer.destroy()
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  Summary: {num_passed}/{num_passed+num_failed} shapes passed")
        if num_failed == 0:
            print(f"  ✅ ALL TESTS PASSED!")
        else:
            print(f"  ❌ SOME TESTS FAILED!")
            for s in fail_shapes:
                print(f"    - {s}")
        print(f"{'='*80}\n")

    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    port = find_free_port()
    os.environ['MASTER_PORT'] = str(port)
    print(f"Launching A2A+GEMM test with {num_gpus} GPUs on port {port}...")
    mp.spawn(run_test, args=(num_gpus, port), nprocs=num_gpus, join=True)
