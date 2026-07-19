"""Component benchmark for Wan2.1 14B Ulysses POST backward."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.bench_utils import find_free_port


def _time(fn, group, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    total = 0.0
    for _ in range(iters):
        dist.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        total += elapsed.item()
    return total / iters


def run(rank, world_size, port, args):
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "DG_AG_PUBLISH_SYNC": args.publish_sync,
    })
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    import deep_gemm
    from deep_gemm import (
        bf16_ag_gemm_nt_with_input,
        bf16_gemm_rs_nt,
        get_unified_symm_buffer,
    )

    hidden = 5120
    if world_size <= 1 or hidden % world_size or args.seq % world_size:
        raise ValueError("SP must divide both hidden=5120 and sequence length")
    local_hidden = hidden // world_size
    local_m = args.seq // world_size
    if local_m % 128:
        raise ValueError("sequence/SP must be a multiple of 128")
    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(100 + rank)

    grad_y = torch.randn(local_m, hidden, dtype=dtype, device="cuda", generator=generator)
    full_weight = torch.randn(hidden, hidden, dtype=dtype, device="cuda", generator=generator)
    local_weight = full_weight[:, rank * local_hidden:(rank + 1) * local_hidden].contiguous()
    local_weight_t = local_weight.t().contiguous()
    gathered_x = torch.randn(local_m, hidden, dtype=dtype, device="cuda", generator=generator)
    head_x = torch.randn(args.seq, local_hidden, dtype=dtype, device="cuda", generator=generator)

    baseline_y = torch.empty_like(gathered_x)
    baseline_dx = torch.empty_like(gathered_x)
    baseline_dw = torch.empty_like(full_weight)
    a2a_send = torch.empty(world_size, local_m, local_hidden, dtype=dtype, device="cuda")
    a2a_recv = torch.empty_like(a2a_send)
    variant_y = torch.empty(local_m, hidden, dtype=dtype, device="cuda")
    variant_dx = torch.empty(args.seq, local_hidden, dtype=dtype, device="cuda")
    variant_dw_t = torch.empty(local_hidden, hidden, dtype=dtype, device="cuda")
    variant_dw = torch.empty(hidden, local_hidden, dtype=dtype, device="cuda")
    transpose_out = torch.empty_like(local_weight_t)
    workspace = get_unified_symm_buffer(group, 1, args.seq, hidden, out_dtype=dtype)
    gathered_grad = None

    def baseline_forward_gemm():
        torch.mm(gathered_x, full_weight.t(), out=baseline_y)

    def variant_forward_gemm_rs():
        bf16_gemm_rs_nt(variant_y, head_x, local_weight, workspace, local_m)

    def baseline_dx_gemm():
        torch.mm(grad_y, full_weight, out=baseline_dx)

    def baseline_dw_gemm():
        torch.mm(grad_y.t(), gathered_x, out=baseline_dw)

    def baseline_a2a():
        dist.all_to_all_single(a2a_recv, a2a_send, group=group)

    def weight_transpose():
        transpose_out.copy_(local_weight.t())

    def variant_ag_gemm():
        nonlocal gathered_grad
        gathered_grad = bf16_ag_gemm_nt_with_input(
            variant_dx, grad_y, local_weight_t, workspace, local_m
        )

    variant_ag_gemm()
    torch.cuda.synchronize()

    def variant_dx_gemm_only():
        torch.mm(gathered_grad, local_weight, out=variant_dx)

    def variant_dw_gemm():
        torch.mm(head_x.t(), gathered_grad, out=variant_dw_t)

    def variant_dw_transpose():
        variant_dw.copy_(variant_dw_t.t())

    components = [
        ("baseline FWD A2A", baseline_a2a),
        ("baseline FWD GEMM", baseline_forward_gemm),
        ("variant FWD GEMM-RS", variant_forward_gemm_rs),
        ("baseline dX GEMM", baseline_dx_gemm),
        ("baseline dW GEMM", baseline_dw_gemm),
        ("baseline dX A2A", baseline_a2a),
        ("variant W transpose", weight_transpose),
        (f"variant AG+GEMM ({args.publish_sync})", variant_ag_gemm),
        ("diagnostic variant dX GEMM only", variant_dx_gemm_only),
        ("variant dW GEMM", variant_dw_gemm),
        ("variant dW transpose", variant_dw_transpose),
    ]
    results = [(name, _time(fn, group, args.iters, args.warmup)) for name, fn in components]

    if rank == 0:
        baseline_fwd = sum(value for name, value in results if name.startswith("baseline FWD"))
        variant_fwd = sum(value for name, value in results if name.startswith("variant FWD"))
        baseline_sum = sum(
            value for name, value in results
            if name.startswith("baseline") and "FWD" not in name
        )
        variant_sum = sum(
            value for name, value in results
            if name.startswith("variant") and "FWD" not in name
        )
        bytes_local = local_m * hidden * torch.tensor([], dtype=dtype).element_size()
        baseline_remote = bytes_local * (world_size - 1) / world_size
        variant_remote = bytes_local * (world_size - 1)
        print(f"POST BWD components: SP={world_size}, seq={args.seq}, local_m={local_m}")
        print(f"Baseline A2A remote payload/rank: {baseline_remote / 1024**2:.2f} MiB")
        print(f"Variant AG remote payload/rank: {variant_remote / 1024**2:.2f} MiB")
        print(f"Variant/baseline remote payload: {variant_remote / baseline_remote:.1f}x")
        for name, elapsed in results:
            print(f"{name:<32} {elapsed:>8.3f} ms")
        print(f"{'FWD component sum baseline':<32} {baseline_fwd:>8.3f} ms")
        print(f"{'FWD component sum variant':<32} {variant_fwd:>8.3f} ms")
        print(f"{'BWD local-op sum baseline':<32} {baseline_sum:>8.3f} ms")
        print(f"{'BWD local-op sum variant':<32} {variant_sum:>8.3f} ms")
        print("BWD sums exclude replicated-parameter all-reduce; full training measures DDP overlap separately.")

    workspace.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int, nargs="?", default=8)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--publish-sync", choices=("symm", "host", "none"), default="symm",
        help="'none' is an unsafe timing-only diagnostic that skips publication/consumption ordering",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(
        run,
        args=(args.num_gpus, find_free_port(), args),
        nprocs=args.num_gpus,
        join=True,
    )
