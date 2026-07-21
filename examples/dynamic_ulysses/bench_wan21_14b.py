"""Real Wan2.1 T2V-14B Dynamic SP vs Static SP benchmark.

This benchmark measures real training throughput (tokens/s) on the complete
Wan2.1 14B transformer (40 blocks, official weights) with:

  - Static SP=8:  every microbatch uses SP=8 (8 GPUs, no DP)
  - Dynamic SP:   BalancedDataLoader assigns SP size per sequence;
                  microbatches with the same SP run DP copies in parallel

Control variables (identical for both arms):
  - Same model: SPWanTransformer with SerialUlysses self-attention
  - Same weights: official Wan2.1-T2V-14B checkpoint (or broadcast random)
  - Same input data
  - Same gradient sync (manual all-reduce across all ranks)

The ONLY independent variable is the SP scheduling strategy.

Usage:
  python examples/dynamic_ulysses/bench_wan21_14b.py [num_gpus] [--seq N] \\
      [--layers N] [--checkpoint-dir PATH] [--synthetic]

Examples:
  # Full 14B, official weights, 8 GPUs
  python examples/dynamic_ulysses/bench_wan21_14b.py 8

  # Fewer layers for quick test
  python examples/dynamic_ulysses/bench_wan21_14b.py 8 --layers 4 --synthetic

  # Custom checkpoint dir
  python examples/dynamic_ulysses/bench_wan21_14b.py 8 --checkpoint-dir /path/to/wan
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.bench_utils import find_free_port
from wan21.checkpoint import (
    OFFICIAL_REPO_ID,
    load_and_broadcast_official_parameters,
    resolve_official_checkpoint,
)
from wan21.config import SPConfig, Wan21Config
from wan21.grad_sync import sync_replicated_grads
from wan21.sp_training import SPWanTransformer

# Dynamic SP imports
DYN_DIR = os.path.join(EXAMPLES_DIR, 'dynamic_ulysses')
if DYN_DIR not in sys.path:
    sys.path.insert(0, DYN_DIR)

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader, Microbatch


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _grid_for_sequence(sequence_length: int) -> torch.Tensor:
    """Build a Wan2.1 3D grid (T, H=16, W=128) for the given sequence length."""
    spatial = 16 * 128
    if sequence_length % spatial:
        # Try 8x128
        spatial = 8 * 128
        if sequence_length % spatial:
            raise ValueError(f"seq {sequence_length} not divisible by {16*128} or {8*128}")
    temporal = sequence_length // spatial
    if temporal > 1024:
        raise ValueError("temporal exceeds official 1024-entry RoPE table")
    return torch.tensor([[temporal, 16 if spatial == 16 * 128 else 8, 128]],
                        dtype=torch.long)


def _max_across_ranks(values, group, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.cpu().tolist()


def _parameter_sync_sizes(module):
    replicated = 0
    sharded = 0
    for parameter in module.parameters():
        if getattr(parameter, '_sp_sharded', False):
            sharded += parameter.numel()
        else:
            replicated += parameter.numel()
    return replicated, sharded


# ----------------------------------------------------------------------------
# Reconfigure SPWanTransformer for a different SP size at runtime
# ----------------------------------------------------------------------------
def reconfigure_sp(model: SPWanTransformer, sp_size: int, sp_group, seq_len: int):
    """Switch every self-attention layer to a new SP size + group.

    SerialUlysses has no pre-allocated buffers, so we only need to update
    the scalar attributes and re-run setup_shape.
    """
    config_dim = model.blocks[0].self_attn.cfg.dim
    num_heads = model.blocks[0].self_attn.cfg.num_heads
    head_dim = model.blocks[0].self_attn.head_dim
    bs = 1

    for block in model.blocks:
        attn = block.self_attn
        # Update SP config
        attn.sp_size = sp_size
        attn.group = sp_group
        attn.sp = replace(attn.sp, sp_size=sp_size, group=sp_group)
        # Re-run setup_shape with new SP (recomputes local_nh, local_seq, etc.)
        attn.setup_shape(bs, seq_len, num_heads, head_dim)


# ----------------------------------------------------------------------------
# Static SP run (baseline)
# ----------------------------------------------------------------------------
def run_static(model, sp_size, sp_group, seq_len, dim, device, e_base, context_base,
               num_iters, warmup, world_group):
    """Run forward+backward+grad_sync with a fixed SP size."""
    # Configure model for this SP size
    reconfigure_sp(model, sp_size, sp_group, seq_len)

    local_seq = seq_len // sp_size
    grid = _grid_for_sequence(seq_len).to(device)

    # Fixed input (same for all iters)
    g = torch.Generator(device=device).manual_seed(1234)
    x_seed = torch.randn(local_seq, dim, device=device, dtype=torch.bfloat16, generator=g)
    grad_output = torch.randn(local_seq, dim, device=device, dtype=torch.float32, generator=g)
    e = e_base.to(device)
    context = context_base.to(device)

    times = []
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        x_local = x_seed.detach().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x_local, e, grid, context)
        output.backward(grad_output)
        sync_replicated_grads(model, world_group, average=True)
        dist.barrier(world_group)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        model.zero_grad(set_to_none=True)
        x_local = x_seed.detach().requires_grad_(True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(x_local, e, grid, context)
        output.backward(grad_output)
        sync_replicated_grads(model, world_group, average=True)

        e_evt.record()
        torch.cuda.synchronize()
        wall_ms = s_evt.elapsed_time(e_evt)
        # Take max across ranks (wall-clock = slowest rank)
        wall_max = _max_across_ranks([wall_ms], world_group, device)[0]
        times.append(wall_max)
        dist.barrier(world_group)

    avg_ms = sum(times) / len(times)
    return avg_ms


# ----------------------------------------------------------------------------
# Dynamic SP run
# ----------------------------------------------------------------------------
def run_dynamic(model, gm: DynamicSPGroupManager, loader: BalancedDataLoader,
                seq_lengths, dim, device, e_base, context_base,
                num_iters, warmup, world_group):
    """Run forward+backward+grad_sync with dynamic SP scheduling."""
    # Pre-compute grids and inputs for each possible sequence length
    seq_set = set(seq_lengths)
    grids = {s: _grid_for_sequence(s).to(device) for s in seq_set}
    inputs = {}
    for s in seq_set:
        local_seqs = {}
        for sp in gm.get_valid_sp_sizes():
            if s % sp == 0 and (s // sp) % 128 == 0:
                ls = s // sp
                g = torch.Generator(device=device).manual_seed(1234 + s)
                local_seqs[sp] = torch.randn(ls, dim, device=device,
                                             dtype=torch.bfloat16, generator=g)
        inputs[s] = local_seqs

    e = e_base.to(device)
    context = context_base.to(device)

    # Schedule microbatches (computed once, reused across iters)
    mbs = loader.schedule(seq_lengths)

    # Group by SP size for parallel DP execution
    by_sp = {}
    for mb in mbs:
        by_sp.setdefault(mb.sp_size, []).append(mb)

    grad_output_cache = {}
    for s in seq_set:
        for sp, ls in inputs[s].items():
            g = torch.Generator(device=device).manual_seed(5678 + s)
            grad_output_cache[(s, sp)] = torch.randn(ls, dim, device=device,
                                                      dtype=torch.float32, generator=g)

    times = []
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _run_dynamic_iter(model, gm, by_sp, inputs, grad_output_cache,
                          grids, e, context, dim, device, world_group)
        sync_replicated_grads(model, world_group, average=True)
        dist.barrier(world_group)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()

        _run_dynamic_iter(model, gm, by_sp, inputs, grad_output_cache,
                          grids, e, context, dim, device, world_group)
        sync_replicated_grads(model, world_group, average=True)

        e_evt.record()
        torch.cuda.synchronize()
        wall_ms = s_evt.elapsed_time(e_evt)
        wall_max = _max_across_ranks([wall_ms], world_group, device)[0]
        times.append(wall_max)
        dist.barrier(world_group)

    avg_ms = sum(times) / len(times)
    return avg_ms, mbs


def _run_dynamic_iter(model, gm, by_sp, inputs, grad_output_cache,
                      grids, e, context, dim, device, world_group):
    """Execute one dynamic SP training iteration."""
    rank = dist.get_rank(world_group)

    # Process SP groups sequentially (largest SP first)
    for sp_size in sorted(by_sp.keys(), reverse=True):
        info = gm.get_groups(sp_size)
        dp_size = gm.world_size // sp_size
        group_mbs = by_sp[sp_size]

        # Configure model for this SP size
        # Use the max seq_len in this group for setup_shape
        max_seq = max(mb.seq_len for mb in group_mbs)
        reconfigure_sp(model, sp_size, info.sp_group, max_seq)

        # Process DP copies in parallel rounds
        for round_idx in range(0, len(group_mbs), dp_size):
            dp_idx = rank % dp_size
            mb_idx = round_idx + dp_idx
            if mb_idx < len(group_mbs):
                mb = group_mbs[mb_idx]
                x_local = inputs[mb.seq_len][sp_size].detach().requires_grad_(True)
                grid = grids[mb.seq_len]
                grad_out = grad_output_cache[(mb.seq_len, sp_size)]

                # Reconfigure for this specific seq_len if different from max_seq
                if mb.seq_len != max_seq:
                    reconfigure_sp(model, sp_size, info.sp_group, mb.seq_len)

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(x_local, e, grid, context)
                output.backward(grad_out)
            dist.barrier(world_group)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(port),
        'RANK': str(rank),
        'WORLD_SIZE': str(world_size),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=2))
    world_group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    config = Wan21Config()
    if config.num_heads % world_size:
        raise ValueError(f"{config.num_heads} heads not divisible by SP={world_size}")

    # Dynamic SP group manager
    gm = DynamicSPGroupManager(world_size, group=world_group)
    loader = BalancedDataLoader(world_size)

    # Build the real Wan2.1 14B transformer (SerialUlysses strategy)
    strategy = 'serial'  # Pure PyTorch A2A — no DeepGEMM buffers, easy to reconfigure
    torch.manual_seed(42)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = SPWanTransformer(
                config,
                SPConfig(sp_size=world_size, group=world_group, layout='THD'),
                args.layers, strategy,
            )
    finally:
        torch.set_default_dtype(old_dtype)
    model.to(device=device)

    # Load weights
    if checkpoint_dir is not None:
        loaded, elements = load_and_broadcast_official_parameters(
            model, checkpoint_dir, world_group, key_map=model.official_key
        )
        if rank == 0:
            print(f'Loaded {loaded} tensors / {elements / 1e9:.3f}B params '
                  f'from official Wan2.1-T2V-14B checkpoint', flush=True)
    else:
        for param in model.parameters():
            dist.broadcast(param.data, src=0, group=world_group)
        if rank == 0:
            print('Using synthetic (random) weights', flush=True)

    # Initial setup (will be reconfigured per-SP during benchmark)
    model.setup_shape(1, args.seq, config.num_heads, config.head_dim)
    model.train()

    replicated, sharded = _parameter_sync_sizes(model)
    if rank == 0:
        print(f'Model: {args.layers} layers, dim={config.dim}, '
              f'SP-synced={replicated / 1e9:.3f}B, SP-local={sharded / 1e9:.3f}B', flush=True)

    # Shared conditioning inputs (e and context)
    cond_gen = torch.Generator(device=device).manual_seed(9999)
    e_base = torch.randn(1, 6, config.dim, device=device,
                         dtype=torch.float32, generator=cond_gen) * 0.01
    context_base = torch.randn(1, 512, config.dim, device=device,
                               dtype=torch.bfloat16, generator=cond_gen) * 0.02

    # ---- Benchmark scenarios ----
    scenarios = {
        'uniform_8K':    [args.seq] * world_size,
        'mixed':         [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048][:world_size],
        'all_short_2K':  [2048] * world_size,
        'bimodal':       [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048][:world_size],
        'one_long_tail': [32768] + [2048] * (world_size - 1),
    }

    if rank == 0:
        print(f'\n{"=" * 110}')
        print(f'Wan2.1 T2V-14B Real Training Benchmark')
        print(f'  B300 x{world_size}, {args.layers} layers, dim={config.dim}, '
              f'heads={config.num_heads}, head_dim={config.head_dim}')
        print(f'  Strategy: {strategy} (same code path for all arms)')
        print(f'  Control: identical model, weights, data, grad sync — '
              f'ONLY SP scheduling differs')
        print(f'{"=" * 110}')
        print(f'{"Scenario":<18} {"Static SP=8":>14} {"Dynamic SP":>14} '
              f'{"Speedup":>9} {"Dyn Schedule":>30}')
        print(f'{"":<18} {"(ms / tok/s)":>14} {"(ms / tok/s)":>14} {"":>9} {"":>30}')
        print('-' * 110)

    speedups = []
    for name, seqs in scenarios.items():
        # Filter sequences that are too long for RoPE
        max_seq = max(seqs)
        if max_seq > 32768:
            if rank == 0:
                print(f'{name}: skipped (seq {max_seq} > 32768)')
            continue

        # Check divisibility
        valid = True
        for s in seqs:
            if s % (8 * 128) != 0:
                valid = False
                break
        if not valid:
            if rank == 0:
                print(f'{name}: skipped (seq not divisible by 1024)')
            continue

        total_tokens = sum(seqs)

        # --- Static SP=world_size ---
        sp_info = gm.get_groups(world_size)
        t_static = run_static(model, world_size, sp_info.sp_group,
                              max_seq, config.dim, device,
                              e_base, context_base,
                              args.iters, args.warmup, world_group)
        tps_static = total_tokens / (t_static / 1000.0)

        # --- Dynamic SP ---
        t_dynamic, mbs = run_dynamic(model, gm, loader, seqs,
                                     config.dim, device,
                                     e_base, context_base,
                                     args.iters, args.warmup, world_group)
        tps_dynamic = total_tokens / (t_dynamic / 1000.0)

        speedup = t_static / t_dynamic if t_dynamic > 0 else 0
        speedups.append(speedup)

        # SP distribution
        sp_dist = {}
        for mb in mbs:
            sp_dist[mb.sp_size] = sp_dist.get(mb.sp_size, 0) + 1

        if rank == 0:
            print(f'{name:<18} {t_static:>7.1f}ms {tps_static:>6.0f}  '
                  f'{t_dynamic:>7.1f}ms {tps_dynamic:>6.0f}  '
                  f'{speedup:>7.3f}x  {str(sp_dist):>30}')

    if rank == 0 and speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f'{"=" * 110}')
        print(f'Geometric mean speedup (Dynamic vs Static SP={world_size}): {geo:.3f}x')
        print(f'{"=" * 110}\n')

    model.destroy_buffers()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Wan2.1 T2V-14B Dynamic SP vs Static SP benchmark')
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('--seq', type=int, default=8192,
                        help='Sequence length for uniform scenario (default 8192)')
    parser.add_argument('--layers', type=int, default=40,
                        help='Number of transformer layers (14B=40)')
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--bucket-cap-mb', type=float, default=64.0)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--repo-id', default=OFFICIAL_REPO_ID)
    parser.add_argument('--revision')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use random weights instead of official checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    local_checkpoint = None
    if not cli_args.synthetic:
        local_checkpoint = resolve_official_checkpoint(
            cli_args.checkpoint_dir, cli_args.repo_id, cli_args.revision
        )
        if cli_args.num_gpus <= 1:
            print(f'Official checkpoint: {local_checkpoint}')
    else:
        print('WARNING: using synthetic weights by explicit request')
    launch_port = find_free_port()
    mp.spawn(
        run,
        args=(cli_args.num_gpus, launch_port, cli_args, local_checkpoint),
        nprocs=cli_args.num_gpus,
        join=True,
    )
