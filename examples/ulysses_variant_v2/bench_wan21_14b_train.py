"""Real-weight Wan2.1 T2V-14B transformer training benchmark for variant v2.

Compares serial baseline, v1 (fused_var), and v2 (fused_var_v2).

  serial:       torch PRE + sync NCCL A2A + torch Wo (replicated)
  fused_var:    torch PRE + DeepGEMM GEMM+RS / AG+GEMM POST (Wo sharded)
  fused_var_v2: torch PRE + DeepGEMM GEMM+RS fwd, native AG+GEMM bwd
                with deferred QKV weight-grad overlap (Wo sharded)

Usage:
    DG_JIT_USE_NVRTC=1 \\
    PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \\
    python3 examples/ulysses_variant_v2/bench_wan21_14b_train.py \\
      8 --layers 40 --seq 8192 --warmup 3 --iters 10 \\
      --strategies serial,fused_var,fused_var_v2 --sync-mode ddp
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.autograd_ops_v2 import finalize_deferred_grads, sync_deferred_grads
from wan21.bench_utils import find_free_port
from wan21.checkpoint import (
    OFFICIAL_REPO_ID,
    load_and_broadcast_official_parameters,
    resolve_official_checkpoint,
)
from wan21.config import SPConfig, Wan21Config
from wan21.grad_sync import sync_replicated_grads
from wan21.sp_training import SPWanTransformer


def _grid_for_sequence(sequence_length: int) -> torch.Tensor:
    spatial = 16 * 128
    if sequence_length % spatial:
        raise ValueError(f"sequence length must be divisible by {spatial}")
    temporal = sequence_length // spatial
    if temporal > 1024:
        raise ValueError("temporal RoPE extent exceeds the official 1024-entry table")
    return torch.tensor([[temporal, 16, 128]], dtype=torch.long)


def _max_across_ranks(values, group, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.cpu().tolist()


def _parameter_sync_sizes(module):
    replicated = 0
    sharded = 0
    for parameter in module.parameters():
        if getattr(parameter, "_sp_sharded", False):
            sharded += parameter.numel()
        else:
            replicated += parameter.numel()
    return replicated, sharded


def _wrap_ddp(module, group, rank, bucket_cap_mb):
    ignored = [
        name for name, parameter in module.named_parameters()
        if getattr(parameter, "_sp_sharded", False)
        or getattr(parameter, "_deferred_grad", False)
    ]
    if ignored:
        DDP._set_params_and_buffers_to_ignore_for_model(module, ignored)
    return DDP(
        module,
        device_ids=[rank],
        output_device=rank,
        process_group=group,
        broadcast_buffers=False,
        bucket_cap_mb=bucket_cap_mb,
        gradient_as_bucket_view=True,
        static_graph=True,
    )


def _run_iteration(train_model, raw_model, x_seed, grad_output, e, grid, context,
                   sync_mode, group, bucket_cap_mb, measure):
    raw_model.zero_grad(set_to_none=True)
    x_local = x_seed.detach().requires_grad_(True)

    if measure:
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        total_start = torch.cuda.Event(enable_timing=True)
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        sync_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        fwd_start.record()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = train_model(x_local, e, grid, context)

    if measure:
        fwd_end.record()
    output.backward(grad_output)

    # v2: finalize deferred QKV weight grads before grad sync
    finalize_deferred_grads(raw_model)

    if sync_mode == "ddp":
        sp = dist.get_world_size(group)
        for parameter in raw_model.parameters():
            if getattr(parameter, "_sp_sharded", False) and parameter.grad is not None:
                parameter.grad.div_(sp)
        # v2: manually all-reduce deferred-grad params excluded from DDP
        sync_deferred_grads(raw_model, group)
    if measure:
        bwd_end.record()

    if sync_mode == "manual":
        sync_replicated_grads(raw_model, group, bucket_cap_mb=bucket_cap_mb)
    if measure:
        sync_end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return (
            fwd_start.elapsed_time(fwd_end),
            fwd_end.elapsed_time(bwd_end),
            bwd_end.elapsed_time(sync_end),
            total_start.elapsed_time(sync_end),
            wall_ms,
        )
    torch.cuda.synchronize()
    return None


def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(hours=2),
    )
    group = dist.group.WORLD
    device = torch.device(f"cuda:{rank}")

    import deep_gemm  # noqa

    config = Wan21Config()
    if config.num_heads % world_size:
        raise ValueError(f"40 attention heads are not divisible by SP={world_size}")
    if args.seq % world_size or (args.seq // world_size) % 128:
        raise ValueError("seq/SP must be an integer multiple of 128")

    grid = _grid_for_sequence(args.seq)
    local_seq = args.seq // world_size
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    x_seed = torch.randn(local_seq, config.dim, device=device,
                         dtype=torch.bfloat16, generator=generator)
    grad_output = torch.randn(local_seq, config.dim, device=device,
                              dtype=torch.float32, generator=generator)
    condition_generator = torch.Generator(device=device).manual_seed(5678)
    e = torch.randn(1, 6, config.dim, device=device,
                    dtype=torch.float32, generator=condition_generator) * 0.01
    context = torch.randn(1, 512, config.dim, device=device,
                          dtype=torch.bfloat16, generator=condition_generator) * 0.02

    results = {}
    for strategy in args.strategies.split(","):
        strategy = strategy.strip()
        torch.manual_seed(42)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                raw_model = SPWanTransformer(
                    config, SPConfig(sp_size=world_size, group=group, layout="THD"),
                    args.layers, strategy,
                )
        finally:
            torch.set_default_dtype(old_dtype)
        raw_model.to(device=device)

        if checkpoint_dir is not None:
            loaded, elements = load_and_broadcast_official_parameters(
                raw_model, checkpoint_dir, group, key_map=raw_model.official_key
            )
            if rank == 0:
                print(
                    f"{strategy}: strictly loaded {loaded} tensors / "
                    f"{elements / 1e9:.3f}B parameters from official checkpoint",
                    flush=True,
                )
        else:
            for parameter in raw_model.parameters():
                dist.broadcast(parameter.data, src=0, group=group)

        raw_model.setup_shape(1, args.seq, config.num_heads, config.head_dim)
        raw_model.train()
        replicated, sharded = _parameter_sync_sizes(raw_model)
        train_model = raw_model
        if args.sync_mode == "ddp":
            train_model = _wrap_ddp(raw_model, group, rank, args.bucket_cap_mb)

        dist.barrier(group)
        for _ in range(args.warmup):
            _run_iteration(
                train_model, raw_model, x_seed, grad_output, e, grid, context,
                args.sync_mode, group, args.bucket_cap_mb, measure=False,
            )
            dist.barrier(group)

        torch.cuda.reset_peak_memory_stats(device)
        accumulated = [0.0, 0.0, 0.0, 0.0, 0.0]
        for _ in range(args.iters):
            dist.barrier(group)
            local_times = _run_iteration(
                train_model, raw_model, x_seed, grad_output, e, grid, context,
                args.sync_mode, group, args.bucket_cap_mb, measure=True,
            )
            rank_max = _max_across_ranks(local_times, group, device)
            accumulated = [a + b for a, b in zip(accumulated, rank_max)]

        times = [value / args.iters for value in accumulated]
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_mb += raw_model.sym_buffer_bytes() / 1024**2
        peak_mb = _max_across_ranks([peak_mb], group, device)[0]
        throughput = args.seq / (times[4] / 1000.0)
        results[strategy] = (times, throughput, peak_mb, replicated, sharded)

        if rank == 0:
            fwd, bwd, sync, total, wall = times
            sync_label = "overlapped in BWD" if args.sync_mode == "ddp" else f"{sync:.2f} ms"
            print(
                f"{strategy:<14} fwd={fwd:>9.2f} ms  bwd={bwd:>9.2f} ms  "
                f"sync={sync_label:>18}  cuda={total:>9.2f} ms  wall={wall:>9.2f} ms  "
                f"tokens/s={throughput:>10.1f}  peak={peak_mb:>10.1f} MiB",
                flush=True,
            )
            print(
                f"{'':<14} SP-synced={replicated / 1e9:.3f}B params, "
                f"SP-local Wo={sharded / 1e9:.3f}B params",
                flush=True,
            )

        del train_model
        raw_model.destroy_buffers()
        del raw_model
        torch.cuda.empty_cache()
        dist.barrier(group)

    if rank == 0:
        if "serial" in results and "fused_var_v2" in results:
            serial_tps = results["serial"][1]
            v2_tps = results["fused_var_v2"][1]
            delta = (v2_tps / serial_tps - 1.0) * 100.0
            print(
                f"\nThroughput verdict: fused_var_v2 / serial = "
                f"{v2_tps / serial_tps:.4f}x ({delta:+.2f}%)",
                flush=True,
            )
        if "fused_var" in results and "fused_var_v2" in results:
            v1_tps = results["fused_var"][1]
            v2_tps = results["fused_var_v2"][1]
            delta = (v2_tps / v1_tps - 1.0) * 100.0
            print(
                f"Throughput verdict: fused_var_v2 / fused_var = "
                f"{v2_tps / v1_tps:.4f}x ({delta:+.2f}%)",
                flush=True,
            )

    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int, nargs="?", default=8)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--strategies", default="serial,fused_var,fused_var_v2")
    parser.add_argument("--sync-mode", choices=("manual", "ddp"), default="ddp")
    parser.add_argument("--bucket-cap-mb", type=float, default=64.0)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    local_checkpoint = None
    if not cli_args.synthetic:
        local_checkpoint = resolve_official_checkpoint(
            cli_args.checkpoint_dir, cli_args.repo_id, cli_args.revision
        )
        print(f"Official checkpoint: {local_checkpoint}")
    else:
        print("WARNING: using synthetic weights by explicit request")
    launch_port = find_free_port()
    mp.spawn(
        run,
        args=(cli_args.num_gpus, launch_port, cli_args, local_checkpoint),
        nprocs=cli_args.num_gpus,
        join=True,
    )
