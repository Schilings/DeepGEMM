"""Real Wan2.1 T2V-14B Dynamic SP×DP vs Static SP benchmark.

This benchmark measures real training throughput (tokens/s) on the complete
Wan2.1 14B transformer (40 blocks, official weights).

Two arms compared:

  - Static SP=8:  ALL sequences use SP=8 (dp=1), processed SEQUENTIALLY.
                  No DP parallelism. This is the standard Ulysses baseline.

  - Dynamic SP×DP: Each sequence assigned optimal SP size by BalancedDataLoader.
                  Sequences with the same (sp_size, seq_len) run as PARALLEL
                  DP copies. Different (sp_size, seq_len) groups run sequentially.
                  dp_size = world_size / sp_size copies run in parallel.

The key insight: for weight gradients, SP all-reduce and DP all-reduce are
the SAME operation (cross-rank gradient aggregation). So SP size can be
dynamically adjusted across the SP×DP process grid. Short sequences use
small SP (e.g. SP=2, dp=4) → 4 copies run in parallel. Long sequences use
large SP (e.g. SP=8, dp=1) → all 8 GPUs collaborate.

Control variables (identical for both arms):
  - Model: SPWanTransformer with SerialUlysses (same code path)
  - Weights: official Wan2.1-T2V-14B checkpoint
  - Input data: same per sequence length
  - Gradient sync: manual all-reduce across all ranks (after all groups)
  - Total tokens: same

The ONLY independent variable is the SP×DP scheduling strategy.

Usage:
  python examples/dynamic_ulysses/bench_wan21_14b.py [num_gpus] [--seq N] \\
      [--layers N] [--checkpoint-dir PATH] [--synthetic]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

DYN_DIR = os.path.join(EXAMPLES_DIR, 'dynamic_ulysses')
if DYN_DIR not in sys.path:
    sys.path.insert(0, DYN_DIR)

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader, Microbatch


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _grid_for_sequence(sequence_length: int) -> torch.Tensor:
    for spatial in [16 * 128, 8 * 128]:
        if sequence_length % spatial == 0:
            temporal = sequence_length // spatial
            if temporal <= 1024:
                h = 16 if spatial == 16 * 128 else 8
                return torch.tensor([[temporal, h, 128]], dtype=torch.long)
    raise ValueError(f"seq {sequence_length} not compatible with RoPE grid")


def _max_across_ranks(values, group, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.cpu().tolist()


def _parameter_sync_sizes(module):
    replicated = sharded = 0
    for p in module.parameters():
        if getattr(p, '_sp_sharded', False):
            sharded += p.numel()
        else:
            replicated += p.numel()
    return replicated, sharded


def reconfigure_sp(model: SPWanTransformer, sp_size: int, sp_group, seq_len: int):
    """Switch every self-attention layer to a new SP size + group."""
    num_heads = model.blocks[0].self_attn.cfg.num_heads
    head_dim = model.blocks[0].self_attn.head_dim
    for block in model.blocks:
        attn = block.self_attn
        attn.sp_size = sp_size
        attn.group = sp_group
        attn.sp = replace(attn.sp, sp_size=sp_size, group=sp_group)
        attn.setup_shape(1, seq_len, num_heads, head_dim)


def _precompute_inputs(seq_lengths, sp_sizes, dim, device):
    """Pre-generate inputs and grad outputs for each (seq_len, sp_size)."""
    seq_set = set(seq_lengths)
    grids = {s: _grid_for_sequence(s).to(device) for s in seq_set}
    inputs = {}
    grad_outputs = {}
    for s in seq_set:
        inputs[s] = {}
        grad_outputs[s] = {}
        for sp in sp_sizes:
            if s % sp == 0 and (s // sp) % 128 == 0:
                ls = s // sp
                g = torch.Generator(device=device).manual_seed(1234 + s)
                inputs[s][sp] = torch.randn(ls, dim, device=device,
                                             dtype=torch.bfloat16, generator=g)
                g2 = torch.Generator(device=device).manual_seed(5678 + s)
                grad_outputs[s][sp] = torch.randn(ls, dim, device=device,
                                                   dtype=torch.float32, generator=g2)
    return grids, inputs, grad_outputs


# ----------------------------------------------------------------------------
# Static SP run (baseline): all sequences SP=world_size, sequential
# ----------------------------------------------------------------------------
def run_static(model, gm, seq_lengths, dim, device, e, context,
               num_iters, warmup, world_group, world_size):
    """Static SP=world_size: process all sequences sequentially, no DP."""
    sp_size = world_size
    sp_group = gm.get_groups(sp_size).sp_group
    grids, inputs, grad_outputs = _precompute_inputs(
        seq_lengths, [sp_size], dim, device)

    times = []
    for it in range(warmup + num_iters):
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()

        for seq_len in seq_lengths:
            reconfigure_sp(model, sp_size, sp_group, seq_len)
            x = inputs[seq_len][sp_size].detach().requires_grad_(True)
            grid = grids[seq_len]
            grad_out = grad_outputs[seq_len][sp_size]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(x, e, grid, context)
            output.backward(grad_out)

        sync_replicated_grads(model, world_group, average=True)

        e_evt.record()
        torch.cuda.synchronize()
        wall_ms = s_evt.elapsed_time(e_evt)
        wall_max = _max_across_ranks([wall_ms], world_group, device)[0]
        dist.barrier(world_group)
        if it >= warmup:
            times.append(wall_max)

    return sum(times) / len(times)


# ----------------------------------------------------------------------------
# Dynamic SP×DP run: parallel DP copies within each (sp_size, seq_len) group
# ----------------------------------------------------------------------------
def run_dynamic(model, gm, seq_lengths, dim, device, e, context,
                num_iters, warmup, world_group, world_size, loader):
    """Dynamic SP×DP: sequences grouped by (sp_size, seq_len), DP copies parallel.

    For each (sp_size, seq_len) group:
      - dp_size = world_size / sp_size DP copies can run in parallel
      - All ranks in the SP group must call A2A together (same seq_len)
      - If fewer sequences than dp_size, extra slots run dummy forward+backward
        (same data) to keep A2A and grad sync synchronized
    """
    rank = dist.get_rank(world_group)

    # Assign SP size to each sequence
    seq_sp = []
    for s in seq_lengths:
        sp = loader.assign_sp_size(s)
        if s % sp != 0 or (s // sp) % 128 != 0:
            for try_sp in sorted(gm.get_valid_sp_sizes(), reverse=True):
                if s % try_sp == 0 and (s // try_sp) % 128 == 0:
                    sp = try_sp
                    break
        seq_sp.append((s, sp))

    # Group by (sp_size, seq_len)
    groups = {}
    for s, sp in seq_sp:
        key = (sp, s)
        groups.setdefault(key, 0)
        groups[key] += 1

    sp_sizes_used = set(sp for _, sp in seq_sp)
    grids, inputs, grad_outputs = _precompute_inputs(
        seq_lengths, sp_sizes_used, dim, device)

    # Sort groups: largest SP first (longest jobs first), then longest seq
    sorted_keys = sorted(groups.keys(), key=lambda k: (-k[0], -k[1]))

    mbs = [Microbatch(sp_size=sp, seq_len=s, local_seq=s // sp,
                       dp_copy=0, tokens=s) for s, sp in seq_sp]

    times = []
    for it in range(warmup + num_iters):
        model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()

        # Process each (sp_size, seq_len) group sequentially.
        # Within a group, dp_size DP copies run in parallel.
        for (sp_size, seq_len) in sorted_keys:
            count = groups[(sp_size, seq_len)]
            info = gm.get_groups(sp_size)
            sp_group = info.sp_group
            dp_size = world_size // sp_size

            reconfigure_sp(model, sp_size, sp_group, seq_len)

            # Number of parallel rounds needed
            num_rounds = max(1, math.ceil(count / dp_size))

            for round_idx in range(num_rounds):
                # Which DP copy this rank belongs to
                dp_idx = rank // sp_size
                mb_idx = round_idx * dp_size + dp_idx
                has_real_seq = mb_idx < count

                # ALL ranks do forward+backward (even dummy) to keep
                # A2A and grad sync synchronized. Dummy uses same data;
                # its gradient will be averaged out by all-reduce.
                x = inputs[seq_len][sp_size].detach().requires_grad_(True)
                grid = grids[seq_len]
                grad_out = grad_outputs[seq_len][sp_size]

                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(x, e, grid, context)
                output.backward(grad_out)

                dist.barrier(world_group)

        sync_replicated_grads(model, world_group, average=True)

        e_evt.record()
        torch.cuda.synchronize()
        wall_ms = s_evt.elapsed_time(e_evt)
        wall_max = _max_across_ranks([wall_ms], world_group, device)[0]
        dist.barrier(world_group)
        if it >= warmup:
            times.append(wall_max)

    return sum(times) / len(times), mbs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({
        'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
        'RANK': str(rank), 'WORLD_SIZE': str(world_size),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=2))
    world_group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    config = Wan21Config()
    if config.num_heads % world_size:
        raise ValueError(f"{config.num_heads} heads not divisible by SP={world_size}")

    gm = DynamicSPGroupManager(world_size, group=world_group)
    loader = BalancedDataLoader(world_size)

    strategy = 'serial'
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

    if checkpoint_dir is not None:
        loaded, elements = load_and_broadcast_official_parameters(
            model, checkpoint_dir, world_group, key_map=model.official_key)
        if rank == 0:
            print(f'Loaded {loaded} tensors / {elements / 1e9:.3f}B params '
                  f'from official Wan2.1-T2V-14B checkpoint', flush=True)
    else:
        for p in model.parameters():
            dist.broadcast(p.data, src=0, group=world_group)
        if rank == 0:
            print('Using synthetic (random) weights', flush=True)

    model.setup_shape(1, args.seq, config.num_heads, config.head_dim)
    model.train()

    replicated, sharded = _parameter_sync_sizes(model)
    if rank == 0:
        print(f'Model: {args.layers} layers, dim={config.dim}, '
              f'params={replicated / 1e9 + sharded / 1e9:.3f}B '
              f'(SP-synced={replicated / 1e9:.3f}B, SP-local={sharded / 1e9:.3f}B)',
              flush=True)

    cond_gen = torch.Generator(device=device).manual_seed(9999)
    e = torch.randn(1, 6, config.dim, device=device,
                     dtype=torch.float32, generator=cond_gen) * 0.01
    context = torch.randn(1, 512, config.dim, device=device,
                           dtype=torch.bfloat16, generator=cond_gen) * 0.02

    # ---- Scenarios ----
    scenarios = {
        'uniform_8K':    [args.seq] * world_size,
        'uniform_32K':   [32768] * (world_size // 4),
        'mixed':         [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048][:world_size],
        'all_short_2K':  [2048] * world_size,
        'bimodal':       [32768, 32768, 2048, 2048, 2048, 2048, 2048, 2048][:world_size],
        'one_long_tail': [32768] + [2048] * (world_size - 1),
    }

    if rank == 0:
        print(f'\n{"=" * 130}')
        print(f'Wan2.1 T2V-14B Real Training Benchmark — Dynamic SP×DP vs Static SP')
        print(f'  Hardware: B300 x{world_size}')
        print(f'  Model: {args.layers} layers, dim={config.dim}, heads={config.num_heads}, '
              f'head_dim={config.head_dim}')
        print(f'  Strategy: {strategy} (same code path for all arms)')
        print(f'  Static: all sequences SP={world_size} (dp=1), sequential')
        print(f'  Dynamic: per-sequence SP size, DP copies parallel within each SP group')
        print(f'  Control: identical model, weights, data, grad sync — ONLY SP×DP scheduling differs')
        print(f'  Measurement: {args.iters} iters, {args.warmup} warmup, event-timed, max-across-ranks')
        print(f'{"=" * 130}')
        print(f'{"Scenario":<18} {"Seqs":>22} {"Tokens":>8} '
              f'{"Static SP=8":>16} {"Dynamic SP×DP":>16} {"Speedup":>8} '
              f'{"Dyn SP Schedule":>24}')
        print(f'{"":<18} {"":>22} {"":>8} '
              f'{"(ms / tok/s)":>16} {"(ms / tok/s)":>16} {"":>8} {"":>24}')
        print('-' * 130)

    speedups = []
    for name, seqs in scenarios.items():
        max_seq = max(seqs)
        if max_seq > 32768:
            if rank == 0:
                print(f'{name:<18} skipped (seq {max_seq} > 32768)', flush=True)
            continue
        if any(s % (8 * 128) != 0 for s in seqs):
            if rank == 0:
                print(f'{name:<18} skipped (seq not divisible by 1024)', flush=True)
            continue

        total_tokens = sum(seqs)

        try:
            # --- Static SP=world_size (sequential, no DP) ---
            t_static = run_static(
                model, gm, seqs, config.dim, device, e, context,
                args.iters, args.warmup, world_group, world_size)
            tps_static = total_tokens / (t_static / 1000.0)

            # --- Dynamic SP×DP (parallel DP copies) ---
            t_dynamic, mbs = run_dynamic(
                model, gm, seqs, config.dim, device, e, context,
                args.iters, args.warmup, world_group, world_size, loader)
            tps_dynamic = total_tokens / (t_dynamic / 1000.0)
        except Exception as exc:
            if rank == 0:
                import traceback
                traceback.print_exc()
                print(f'{name:<18} FAILED — {exc}', flush=True)
            continue

        speedup = t_static / t_dynamic if t_dynamic > 0 else 0
        speedups.append(speedup)

        sp_dist = {}
        for mb in mbs:
            sp_dist[mb.sp_size] = sp_dist.get(mb.sp_size, 0) + 1

        seqs_str = str(seqs)[:22]
        if rank == 0:
            print(f'{name:<18} {seqs_str:>22} {total_tokens:>8} '
                  f'{t_static:>7.1f}ms {tps_static:>6.0f}  '
                  f'{t_dynamic:>7.1f}ms {tps_dynamic:>6.0f}  '
                  f'{speedup:>6.3f}x  {str(sp_dist):>24}')

    if rank == 0 and speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f'{"=" * 130}')
        print(f'Geometric mean speedup (Dynamic SP×DP vs Static SP={world_size}): {geo:.3f}x')
        print(f'{"=" * 130}\n')

    model.destroy_buffers()
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Wan2.1 T2V-14B Dynamic SP×DP vs Static SP benchmark')
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('--seq', type=int, default=8192)
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--bucket-cap-mb', type=float, default=64.0)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--repo-id', default=OFFICIAL_REPO_ID)
    parser.add_argument('--revision')
    parser.add_argument('--synthetic', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    local_checkpoint = None
    if not cli_args.synthetic:
        local_checkpoint = resolve_official_checkpoint(
            cli_args.checkpoint_dir, cli_args.repo_id, cli_args.revision)
    else:
        print('WARNING: using synthetic weights by explicit request')
    launch_port = find_free_port()
    mp.spawn(run, args=(cli_args.num_gpus, launch_port, cli_args, local_checkpoint),
             nprocs=cli_args.num_gpus, join=True)
