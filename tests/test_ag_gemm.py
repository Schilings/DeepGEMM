"""
Correctness test for BF16 All-Gather + GEMM fusion.

Tests that fused AG+GEMM produces the same result as:
  1. torch.distributed.all_gather
  2. Standard BF16 GEMM on the gathered activation matrix

Usage:
    python tests/test_ag_gemm.py [num_gpus]        # default: 2
    python tests/test_ag_gemm.py [num_gpus] --all  # run extended shapes
"""

import os
import sys
import time
import signal
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm
from deep_gemm.ag_gemm import bf16_ag_gemm_nt, get_symm_buffer_for_bf16_ag_gemm


TIMEOUT_SECONDS = int(os.getenv('TEST_TIMEOUT', '600'))
PROBE_ONLY = os.getenv('AG_GEMM_PROBE_ONLY', '0') == '1'
SKIP_DESTROY = os.getenv('AG_GEMM_SKIP_DESTROY', '0') == '1'
SHAPE_LIMIT = int(os.getenv('AG_GEMM_SHAPE_LIMIT', '0'))
VERBOSE_STEPS = os.getenv('AG_GEMM_VERBOSE_STEPS', '0') == '1'


SHAPES_BASIC = [
    (256, 1024, 1024),
    (512, 2048, 2048),
    (512, 4096, 2048),
    (1024, 4096, 4096),
    (1024, 7168, 4096),
]

SHAPES_EXTENDED = [
    (2048, 4096, 4096),
    (2048, 4096, 7168),
    (2048, 7168, 4096),
    (4096, 4096, 4096),
    (4096, 7168, 4096),
]


def _timeout_handler(signum, frame):
    print(f"[TIMEOUT] AG GEMM test timed out after {TIMEOUT_SECONDS} seconds", flush=True)
    os._exit(1)


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        return s.getsockname()[1]


def _run_single_shape(rank: int, num_gpus: int, group, tokens_per_rank: int, n_dim: int, k_dim: int):
    device = torch.device(f'cuda:{rank}')

    def _log(stage: str):
        if VERBOSE_STEPS:
            print(f"[rank {rank}] {tokens_per_rank}×{n_dim}×{k_dim} :: {stage}", flush=True)

    x_local = torch.randn((tokens_per_rank, k_dim), dtype=torch.bfloat16, device=device)

    if PROBE_ONLY:
        sym_buffer = get_symm_buffer_for_bf16_ag_gemm(group, tokens_per_rank, k_dim)
        torch.cuda.synchronize(device)
        dist.barrier(group)
        sym_buffer.destroy()
        dist.barrier(group)
        return True, 0.0, 0.0, 0.0, 0.0

    b = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device=device)
    dist.broadcast(b, src=0, group=group)
    torch.cuda.synchronize(device)
    dist.barrier(group)

    gathered_x = [torch.empty_like(x_local) for _ in range(num_gpus)]
    dist.all_gather(gathered_x, x_local, group=group)
    x_full = torch.cat(gathered_x, dim=0).contiguous()
    d_ref = torch.matmul(x_full, b.t())
    torch.cuda.synchronize(device)
    dist.barrier(group)

    _log('before buffer')
    sym_buffer = get_symm_buffer_for_bf16_ag_gemm(group, tokens_per_rank, k_dim)
    _log('after buffer')
    sym_buffer.x[:tokens_per_rank, :k_dim].copy_(x_local)
    _log('after x copy')

    d_fused = torch.zeros((num_gpus * tokens_per_rank, n_dim), dtype=torch.bfloat16, device=device)
    _log('before kernel')
    bf16_ag_gemm_nt(d_fused, b, sym_buffer, tokens_per_rank)
    _log('after kernel launch')
    torch.cuda.synchronize(device)
    _log('after cuda sync')
    dist.barrier(group)
    _log('after barrier')

    d_fused_all = [torch.empty_like(d_fused) for _ in range(num_gpus)]
    dist.all_gather(d_fused_all, d_fused, group=group)
    consistency_diff = 0.0
    if num_gpus > 1:
        consistency_diff = max((d_fused_all[i].float() - d_fused_all[0].float()).abs().max().item() for i in range(1, num_gpus))

    diff = (d_fused.float() - d_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_mean = d_ref.float().abs().mean().item()
    rel_error = mean_diff / max(ref_mean, 1e-8)
    passed = rel_error < 0.01 * num_gpus and consistency_diff < 0.01 and not torch.isnan(diff).any().item()

    sym_buffer.destroy()
    dist.barrier(group)
    return passed, max_diff, mean_diff, rel_error, consistency_diff


def _worker(rank: int, num_gpus: int, port: int, run_all: bool):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    os.environ.update({
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
        'RANK': str(rank),
        'WORLD_SIZE': str(num_gpus),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=num_gpus)
    group = dist.group.WORLD

    torch.manual_seed(20260612 + rank)
    torch.cuda.manual_seed(20260612 + rank)

    shapes = SHAPES_BASIC + (SHAPES_EXTENDED if run_all else [])
    if SHAPE_LIMIT > 0:
        shapes = shapes[:SHAPE_LIMIT]

    if rank == 0:
        print(f"\n{'=' * 88}")
        print(f"  BF16 AG+GEMM Correctness Test: {num_gpus} GPUs")
        print(f"  Testing {len(shapes)} shapes {'(full suite)' if run_all else '(basic suite)'}")
        print(f"{'=' * 88}\n")
        print(f"{'Shape (M/rank×N×K)':<22} | {'Max Diff':>9} {'Mean Diff':>10} {'Rel Err':>9} {'Consist':>9} | Status")
        print(f"{'-' * 22} | {'-' * 9} {'-' * 10} {'-' * 9} {'-' * 9} | {'-' * 8}")

    all_results = []
    all_passed = True
    for tokens_per_rank, n_dim, k_dim in shapes:
        try:
            passed, max_diff, mean_diff, rel_error, consistency_diff = _run_single_shape(
                rank, num_gpus, group, tokens_per_rank, n_dim, k_dim
            )
        except Exception as exc:
            passed = False
            max_diff = mean_diff = rel_error = consistency_diff = float('nan')
            if rank == 0:
                print(f"  ❌ ERROR on {tokens_per_rank}×{n_dim}×{k_dim}: {type(exc).__name__}: {exc}", flush=True)

        all_results.append((tokens_per_rank, n_dim, k_dim, passed, max_diff, mean_diff, rel_error, consistency_diff))
        all_passed = all_passed and passed

        if rank == 0:
            shape_str = f"{tokens_per_rank}×{n_dim}×{k_dim}"
            status = '✅ PASS' if passed else '❌ FAIL'
            print(f"{shape_str:<22} | {max_diff:>9.6f} {mean_diff:>10.7f} {rel_error:>9.6f} {consistency_diff:>9.6f} | {status}")
        dist.barrier(group)

    if rank == 0:
        num_passed = sum(1 for result in all_results if result[3])
        print(f"\n{'=' * 88}")
        print(f"  Summary: {num_passed}/{len(all_results)} shapes passed")
        if all_passed:
            print('  ✅ ALL TESTS PASSED!')
        else:
            print('  ❌ SOME TESTS FAILED!')
            for tokens_per_rank, n_dim, k_dim, passed, max_diff, _, _, _ in all_results:
                if not passed:
                    print(f"    - {tokens_per_rank}×{n_dim}×{k_dim} (max_diff={max_diff})")
        print(f"{'=' * 88}\n")

    dist.barrier(group)
    dist.destroy_process_group()
    time.sleep(1)
    os._exit(0)


if __name__ == '__main__':
    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    run_all = '--all' in sys.argv
    port = int(os.getenv('MASTER_PORT', '0')) or _find_free_port()
    os.environ['MASTER_PORT'] = str(port)
    print(f"Launching AG+GEMM correctness test with {num_gpus} GPUs (port={port})...")
    if run_all:
        print('  Running full test suite')
    mp.spawn(_worker, args=(num_gpus, port, run_all), nprocs=num_gpus, join=True)
