"""Component benchmark for Wan2.1 14B Ulysses POST backward — v1 vs v2.

Compares POST backward components and actual autograd backward time for:
  - baseline (serial): NCCL A2A + torch GEMM
  - v1 (fused_var): DeepGEMM fused AG+GEMM
  - v2 (fused_var_v2): native NCCL all_gather + torch GEMM with deferred
    QKV weight-grad overlap

Usage:
    DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \\
    python3 examples/ulysses_variant_v2/bench_wan21_post_bwd.py 8 --seq 8192
"""
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

from wan21.autograd_ops_v2 import (
    DeferredGradManager,
    finalize_deferred_grads,
    post_linear_v2,
)
from wan21.bench_utils import find_free_port


def _time(name, fn, group, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    total = 0.0
    nvtx_name = name.replace(" ", "_")
    for _ in range(iters):
        dist.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.nvtx.range_push(nvtx_name)
        fn()
        torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        total += elapsed.item()
    return total / iters


def _time_backward(name, build_graph, grad_output, group, iters, warmup, finalize_fn=None):
    def run_once(measure):
        output, inputs = build_graph()
        torch.cuda.synchronize()
        dist.barrier(group)
        if measure:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.cuda.nvtx.range_push(name.replace(" ", "_"))
        torch.autograd.grad(output, inputs, grad_output)
        if finalize_fn:
            finalize_fn()
        if not measure:
            return 0.0
        torch.cuda.nvtx.range_pop()
        end.record()
        torch.cuda.synchronize()
        elapsed = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        return elapsed.item()

    for _ in range(warmup):
        run_once(False)
    return sum(run_once(True) for _ in range(iters)) / iters


def run(rank, world_size, port, args):
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
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
    from wan21.autograd_ops import fused_post_linear
    from wan21.sp.base import NCCLAllToAll

    hidden = 5120
    if world_size <= 1 or hidden % world_size or args.seq % world_size:
        raise ValueError("SP must divide both hidden=5120 and sequence length")
    local_hidden = hidden // world_size
    local_m = args.seq // world_size
    num_heads = 40
    head_dim = 128
    if num_heads % world_size:
        raise ValueError("SP must divide 40 attention heads")
    local_heads = num_heads // world_size
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
    workspace = get_unified_symm_buffer(group, 1, args.seq, hidden, out_dtype=dtype)
    gathered_grad = None
    attn_seed = torch.randn(
        1, args.seq, local_heads, head_dim,
        dtype=dtype, device="cuda", generator=generator,
    )
    full_weight_grad = full_weight.detach().requires_grad_(True)
    local_weight_grad = local_weight.detach().requires_grad_(True)
    local_weight_v2 = local_weight.detach().clone().requires_grad_(True)

    # v2 grad manager
    grad_manager = DeferredGradManager(group, world_size)

    def build_baseline_graph():
        attn = attn_seed.detach().requires_grad_(True)
        send = (
            attn.transpose(1, 2)
            .reshape(1, local_heads, world_size, local_m, head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        recv = NCCLAllToAll.apply(send, group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(local_m, hidden)
        output = torch.nn.functional.linear(gathered, full_weight_grad)
        return output, (attn, full_weight_grad)

    def build_v1_graph():
        attn = attn_seed.detach().requires_grad_(True)
        output = fused_post_linear(
            attn.reshape(args.seq, local_hidden).contiguous(),
            local_weight_grad,
            workspace,
            local_m,
        )
        return output, (attn, local_weight_grad)

    def build_v2_graph():
        attn = attn_seed.detach().requires_grad_(True)
        output = post_linear_v2(
            attn.reshape(args.seq, local_hidden).contiguous(),
            local_weight_v2,
            workspace,
            local_m,
            grad_manager,
        )
        return output, (attn, local_weight_v2)

    # --- Component timing ---
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

    def v2_native_ag():
        ag_buf = grad_manager.get_ag_buffer(args.seq, hidden, dtype, grad_y.device)
        dist.all_gather_into_tensor(ag_buf, grad_y.contiguous(), group=group)

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

    # Pre-allocate for v2 native AG+GEMM
    v2_ag_buf = torch.empty(args.seq, hidden, dtype=dtype, device="cuda")
    v2_dx = torch.empty(args.seq, local_hidden, dtype=dtype, device="cuda")
    v2_dw = torch.empty(hidden, local_hidden, dtype=dtype, device="cuda")

    def v2_ag():
        dist.all_gather_into_tensor(v2_ag_buf, grad_y.contiguous(), group=group)

    def v2_dx_gemm():
        torch.mm(v2_ag_buf, local_weight, out=v2_dx)

    def v2_dw_gemm():
        torch.mm(v2_ag_buf.t(), head_x, out=v2_dw)

    components = [
        ("baseline FWD A2A", baseline_a2a),
        ("baseline FWD GEMM", baseline_forward_gemm),
        ("v1/v2 FWD GEMM-RS", variant_forward_gemm_rs),
        ("baseline dX GEMM", baseline_dx_gemm),
        ("baseline dW GEMM", baseline_dw_gemm),
        ("baseline dX A2A", baseline_a2a),
        ("v1 AG+GEMM (fused)", variant_ag_gemm),
        ("diagnostic v1 dX GEMM only", variant_dx_gemm_only),
        ("v1 dW GEMM", variant_dw_gemm),
        ("v2 native AG (all_gather)", v2_ag),
        ("v2 dX GEMM (native)", v2_dx_gemm),
        ("v2 dW GEMM (native)", v2_dw_gemm),
    ]
    results = [
        (name, _time(name, fn, group, args.iters, args.warmup))
        for name, fn in components
    ]

    actual_baseline_bwd = _time_backward(
        "actual baseline POST BWD", build_baseline_graph, grad_y,
        group, args.iters, args.warmup,
    )
    actual_v1_bwd = _time_backward(
        "actual v1 POST BWD", build_v1_graph, grad_y,
        group, args.iters, args.warmup,
    )
    actual_v2_bwd = _time_backward(
        "actual v2 POST BWD", build_v2_graph, grad_y,
        group, args.iters, args.warmup,
        finalize_fn=lambda: grad_manager.finalize(),
    )

    if rank == 0:
        print(f"POST BWD components: SP={world_size}, seq={args.seq}, local_m={local_m}")
        bytes_local = local_m * hidden * torch.tensor([], dtype=dtype).element_size()
        baseline_remote = bytes_local * (world_size - 1) / world_size
        variant_remote = bytes_local * (world_size - 1)
        print(f"Baseline A2A remote payload/rank: {baseline_remote / 1024**2:.2f} MiB")
        print(f"Variant AG remote payload/rank: {variant_remote / 1024**2:.2f} MiB")
        print(f"Variant/baseline remote payload: {variant_remote / baseline_remote:.1f}x")
        print()
        for name, elapsed in results:
            print(f"  {name:<32} {elapsed:>8.3f} ms")

        v1_sum = sum(v for n, v in results if n.startswith("v1") and "FWD" not in n)
        v2_ag_t = next(v for n, v in results if "v2 native AG" in n)
        v2_dx_t = next(v for n, v in results if "v2 dX GEMM" in n)
        v2_dw_t = next(v for n, v in results if "v2 dW GEMM" in n)
        v2_sum_serial = v2_ag_t + v2_dx_t + v2_dw_t
        v2_overlap = max(v2_ag_t, v2_dx_t + v2_dw_t)  # AG overlaps with deferred grads

        print(f"\n  {'v1 BWD local-op sum':<32} {v1_sum:>8.3f} ms")
        print(f"  {'v2 BWD serial sum (AG+dX+dW)':<32} {v2_sum_serial:>8.3f} ms")
        print(f"  {'v2 BWD overlap (max(AG, dX+dW))':<32} {v2_overlap:>8.3f} ms")
        print(f"  {'actual autograd POST BWD baseline':<32} {actual_baseline_bwd:>8.3f} ms")
        print(f"  {'actual autograd POST BWD v1':<32} {actual_v1_bwd:>8.3f} ms")
        print(f"  {'actual autograd POST BWD v2':<32} {actual_v2_bwd:>8.3f} ms")
        print(f"  {'actual v1 / baseline':<32} {actual_v1_bwd / actual_baseline_bwd:>8.3f}x")
        print(f"  {'actual v2 / baseline':<32} {actual_v2_bwd / actual_baseline_bwd:>8.3f}x")
        print(f"  {'actual v2 / v1':<32} {actual_v2_bwd / actual_v1_bwd:>8.3f}x")
        print("\nBWD sums exclude replicated-parameter all-reduce; full training measures DDP overlap separately.")

    workspace.destroy()
    dist.barrier(group)
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int, nargs="?", default=8)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mp.spawn(
        run,
        args=(args.num_gpus, find_free_port(), args),
        nprocs=args.num_gpus,
        join=True,
    )
