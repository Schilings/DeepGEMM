"""Variable-sweep benchmarks for curve plots.

Sweeps:
1. Sequence length (2K → 32K), fixed 8 sequences
2. Number of sequences (1 → 16), fixed 8K each
3. SP size (1, 2, 4, 8), fixed 8K×8

Usage:
    python examples/dynamic_ulysses/bench_sweep.py 8 [--layers 4] [--synthetic]
"""
from __future__ import annotations

import argparse, math, os, sys, time, json
from dataclasses import replace
from datetime import timedelta

import torch, torch.distributed as dist, torch.multiprocessing as mp

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.bench_utils import find_free_port
from wan21.checkpoint import resolve_official_checkpoint, load_and_broadcast_official_parameters, OFFICIAL_REPO_ID
from wan21.config import SPConfig, Wan21Config
from wan21.sp_training import SPWanTransformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh

DYN_DIR = os.path.join(EXAMPLES_DIR, 'dynamic_ulysses')
if DYN_DIR not in sys.path:
    sys.path.insert(0, DYN_DIR)

from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader


def _grid_for_sequence(seq_len):
    for sp in [16 * 128, 8 * 128]:
        if seq_len % sp == 0 and seq_len // sp <= 1024:
            h = 16 if sp == 16 * 128 else 8
            return torch.tensor([[seq_len // sp, h, 128]], dtype=torch.long)
    raise ValueError(f"seq {seq_len} incompatible")


def _max_across_ranks(values, group, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.cpu().tolist()


def reconfigure_sp(model, sp_size, sp_group, seq_len):
    nh = model.blocks[0].self_attn.cfg.num_heads
    hd = model.blocks[0].self_attn.head_dim
    for block in model.blocks:
        attn = block.self_attn
        attn.sp_size = sp_size
        attn.group = sp_group
        attn.sp = replace(attn.sp, sp_size=sp_size, group=sp_group)
        attn.setup_shape(1, seq_len, nh, hd)


def run_one_config(model, gm, sp_size, seq_len, dim, device, e, context,
                   world_group, world_size, iters=3, warmup=1):
    """Run forward+backward for one (sp_size, seq_len) config, return ms."""
    sp_group = gm.get_groups(sp_size).sp_group
    reconfigure_sp(model, sp_size, sp_group, seq_len)
    local_seq = seq_len // sp_size
    grid = _grid_for_sequence(seq_len).to(device)

    g = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(local_seq, dim, device=device, dtype=torch.bfloat16, generator=g)
    grad = torch.randn(local_seq, dim, device=device, dtype=torch.float32,
                       generator=torch.Generator(device=device).manual_seed(5678))

    times = []
    for it in range(warmup + iters):
        model.zero_grad(set_to_none=True)
        xi = x.detach().requires_grad_(True)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(xi, e, grid, context)
        out.backward(grad)
        e2.record()
        torch.cuda.synchronize()
        wall = s.elapsed_time(e2)
        wall_max = _max_across_ranks([wall], world_group, device)[0]
        dist.barrier(world_group)
        if it >= warmup:
            times.append(wall_max)
    return sum(times) / len(times)


def run_dynamic_config(model, gm, loader, seq_lengths, dim, device, e, context,
                       world_group, world_size, iters=3, warmup=1):
    """Run dynamic SP×DP for a list of sequences, return ms."""
    rank = dist.get_rank(world_group)
    seq_sp = []
    for s in seq_lengths:
        sp = loader.assign_sp_size(s)
        if s % sp != 0:
            for try_sp in sorted(gm.get_valid_sp_sizes(), reverse=True):
                if s % try_sp == 0:
                    sp = try_sp
                    break
        seq_sp.append((s, sp))

    groups = {}
    for s, sp in seq_sp:
        key = (sp, s)
        groups[key] = groups.get(key, 0) + 1

    inputs = {}
    grads = {}
    grids = {}
    for s, sp in seq_sp:
        if s not in inputs:
            ls = s // sp
            g = torch.Generator(device=device).manual_seed(1234 + s)
            inputs[s] = {}
            grads[s] = {}
            for try_sp in gm.get_valid_sp_sizes():
                if s % try_sp == 0:
                    tls = s // try_sp
                    inputs[s][try_sp] = torch.randn(tls, dim, device=device,
                                                     dtype=torch.bfloat16, generator=g)
                    g2 = torch.Generator(device=device).manual_seed(5678 + s)
                    grads[s][try_sp] = torch.randn(tls, dim, device=device,
                                                    dtype=torch.float32, generator=g2)
            grids[s] = _grid_for_sequence(s).to(device)

    sorted_keys = sorted(groups.keys(), key=lambda k: (-k[0], -k[1]))

    times = []
    for it in range(warmup + iters):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        s_evt = torch.cuda.Event(enable_timing=True)
        e_evt = torch.cuda.Event(enable_timing=True)
        s_evt.record()

        for (sp_size, seq_len) in sorted_keys:
            count = groups[(sp_size, seq_len)]
            sp_group = gm.get_groups(sp_size).sp_group
            dp_size = world_size // sp_size
            reconfigure_sp(model, sp_size, sp_group, seq_len)
            num_rounds = max(1, math.ceil(count / dp_size))
            for round_idx in range(num_rounds):
                dp_idx = rank // sp_size
                x = inputs[seq_len][sp_size].detach().requires_grad_(True)
                grad_out = grads[seq_len][sp_size]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x, e, grids[seq_len], context)
                out.backward(grad_out)
                dist.barrier(world_group)

        e_evt.record()
        torch.cuda.synchronize()
        wall = s_evt.elapsed_time(e_evt)
        wall_max = _max_across_ranks([wall], world_group, device)[0]
        dist.barrier(world_group)
        if it >= warmup:
            times.append(wall_max)
    return sum(times) / len(times)


def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(world_size)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=2))
    world_group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    config = Wan21Config()
    gm = DynamicSPGroupManager(world_size, group=world_group)
    loader = BalancedDataLoader(world_size)

    torch.manual_seed(42)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = SPWanTransformer(config, SPConfig(sp_size=world_size,
                group=world_group, layout='THD'), args.layers, 'serial')
    finally:
        torch.set_default_dtype(old)
    model.to(device=device, dtype=torch.bfloat16)

    if checkpoint_dir:
        load_and_broadcast_official_parameters(model, checkpoint_dir, world_group,
                                               key_map=model.official_key)
        if rank == 0:
            print(f'Loaded official checkpoint', flush=True)
    else:
        for p in model.parameters():
            dist.broadcast(p.data, src=0, group=world_group)

    ignored = set()
    for block in model.blocks:
        ignored.add(block.modulation)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                                      output_dtype=torch.bfloat16, cast_forward_inputs=False)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    for block in model.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy,
                    ignored_params=ignored or None)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy,
                ignored_params=ignored or None)

    model.setup_shape(1, 8192, config.num_heads, config.head_dim)
    model.train()

    cond_g = torch.Generator(device=device).manual_seed(9999)
    e = torch.randn(1, 6, config.dim, device=device, dtype=torch.float32, generator=cond_g) * 0.01
    context = torch.randn(1, 512, config.dim, device=device, dtype=torch.bfloat16, generator=cond_g) * 0.02

    dim = config.dim
    results = {}

    # ---- Sweep 1: Sequence length (2K → 32K), 8 sequences ----
    sweep1_seqs = [2048, 4096, 8192, 16384, 32768]
    results['sweep_seq_len'] = {'seq_lens': sweep1_seqs, 'static_sp8': [], 'dynamic': []}
    if rank == 0:
        print(f'\n=== Sweep 1: Sequence Length (8 seqs, varying length) ===', flush=True)
        print(f'{"Seq Len":>8} {"Static SP=8 (ms)":>18} {"Dynamic (ms)":>18} {"Speedup":>8}', flush=True)

    for sl in sweep1_seqs:
        # Static SP=8: 8 sequences, sequential
        t_static = 0
        for _ in range(1):
            t = run_one_config(model, gm, world_size, sl, dim, device, e, context,
                               world_group, world_size, iters=3, warmup=1)
            t_static = t * 8  # 8 sequential
        # Dynamic: 8 sequences
        seqs = [sl] * 8
        t_dyn = run_dynamic_config(model, gm, loader, seqs, dim, device, e, context,
                                   world_group, world_size, iters=3, warmup=1)
        sp = t_static / t_dyn if t_dyn > 0 else 0
        results['sweep_seq_len']['static_sp8'].append(t_static)
        results['sweep_seq_len']['dynamic'].append(t_dyn)
        if rank == 0:
            print(f'{sl//1024:>6}K {t_static:>16.1f}ms {t_dyn:>16.1f}ms {sp:>7.2f}x', flush=True)

    # ---- Sweep 2: Number of sequences (1 → 16), 8K each ----
    sweep2_counts = [1, 2, 4, 8, 12, 16]
    results['sweep_num_seqs'] = {'counts': sweep2_counts, 'static_sp8': [], 'dynamic': []}
    if rank == 0:
        print(f'\n=== Sweep 2: Number of Sequences (8K each) ===', flush=True)
        print(f'{"#Seqs":>6} {"Static SP=8 (ms)":>18} {"Dynamic (ms)":>18} {"Speedup":>8}', flush=True)

    for cnt in sweep2_counts:
        # Static: sequential, each SP=8
        t_one = run_one_config(model, gm, world_size, 8192, dim, device, e, context,
                               world_group, world_size, iters=3, warmup=1)
        t_static = t_one * cnt
        # Dynamic
        seqs = [8192] * cnt
        t_dyn = run_dynamic_config(model, gm, loader, seqs, dim, device, e, context,
                                   world_group, world_size, iters=3, warmup=1)
        sp = t_static / t_dyn if t_dyn > 0 else 0
        results['sweep_num_seqs']['static_sp8'].append(t_static)
        results['sweep_num_seqs']['dynamic'].append(t_dyn)
        if rank == 0:
            print(f'{cnt:>5} {t_static:>16.1f}ms {t_dyn:>16.1f}ms {sp:>7.2f}x', flush=True)

    # ---- Sweep 3: SP size (1, 2, 4, 8), 8K single sequence ----
    sweep3_sps = [1, 2, 4, 8]
    results['sweep_sp_size'] = {'sp_sizes': sweep3_sps, 'times': [], 'throughput': []}
    if rank == 0:
        print(f'\n=== Sweep 3: SP Size (single 8K sequence) ===', flush=True)
        print(f'{"SP":>4} {"DP":>4} {"Time (ms)":>12} {"tok/s":>10}', flush=True)

    for sp in sweep3_sps:
        if sp > world_size:
            continue
        t = run_one_config(model, gm, sp, 8192, dim, device, e, context,
                           world_group, world_size, iters=3, warmup=1)
        tps = 8192 / (t / 1000.0)
        results['sweep_sp_size']['times'].append(t)
        results['sweep_sp_size']['throughput'].append(tps)
        if rank == 0:
            dp = world_size // sp
            print(f'{sp:>3} {dp:>3} {t:>10.1f}ms {tps:>8.0f}', flush=True)

    # ---- Sweep 4: Mixed ratio (long:short ratio, 8 total sequences) ----
    sweep4_ratios = [(0, 8), (1, 7), (2, 6), (4, 4), (6, 2), (8, 0)]
    results['sweep_mixed_ratio'] = {'ratios': sweep4_ratios, 'static_sp8': [], 'dynamic': []}
    if rank == 0:
        print(f'\n=== Sweep 4: Long:Short Ratio (8 total, 32K + 2K) ===', flush=True)
        print(f'{"Long":>5} {"Short":>6} {"Static (ms)":>14} {"Dynamic (ms)":>14} {"Speedup":>8}', flush=True)

    for n_long, n_short in sweep4_ratios:
        seqs = [32768] * n_long + [2048] * n_short
        if not seqs:
            continue
        # Static
        t_static = 0
        for s in seqs:
            t = run_one_config(model, gm, world_size, s, dim, device, e, context,
                               world_group, world_size, iters=2, warmup=1)
            t_static += t
        # Dynamic
        t_dyn = run_dynamic_config(model, gm, loader, seqs, dim, device, e, context,
                                   world_group, world_size, iters=2, warmup=1)
        sp = t_static / t_dyn if t_dyn > 0 else 0
        results['sweep_mixed_ratio']['static_sp8'].append(t_static)
        results['sweep_mixed_ratio']['dynamic'].append(t_dyn)
        if rank == 0:
            print(f'{n_long:>4} {n_short:>5} {t_static:>12.1f}ms {t_dyn:>12.1f}ms {sp:>7.2f}x', flush=True)

    if rank == 0:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {out_path}', flush=True)

    model.destroy_buffers()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--repo-id', default=OFFICIAL_REPO_ID)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    ckpt = None
    if not args.synthetic:
        ckpt = resolve_official_checkpoint(args.checkpoint_dir, args.repo_id, None)
    else:
        print('WARNING: synthetic weights')
    port = find_free_port()
    mp.spawn(run, args=(args.num_gpus, port, args, ckpt), nprocs=args.num_gpus, join=True)
