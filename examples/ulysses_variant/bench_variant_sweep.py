"""Throughput sweep: serial vs fused_var across sequence lengths.

Sweeps sequence lengths [2K, 4K, 8K, 16K, 32K] with full 14B model,
measuring FWD/BWD/wall-clock/peak-memory for serial and fused_var.

Usage:
    DG_AG_PUBLISH_SYNC=symm DG_JIT_USE_NVRTC=1 \
    PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
    python3 examples/ulysses_variant/bench_variant_sweep.py 8 \
        --layers 40 --strategies serial,fused_var --sync-mode ddp \
        --checkpoint-dir /path/to/wan2.1-14b
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import timedelta

import torch, torch.distributed as dist, torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.bench_utils import find_free_port
from wan21.checkpoint import (OFFICIAL_REPO_ID, load_and_broadcast_official_parameters,
                               resolve_official_checkpoint)
from wan21.config import SPConfig, Wan21Config
from wan21.grad_sync import sync_replicated_grads
from wan21.sp_training import SPWanTransformer


def _grid_for_sequence(seq_len):
    for spatial in [16 * 128, 8 * 128]:
        if seq_len % spatial == 0 and seq_len // spatial <= 1024:
            h = 16 if spatial == 16 * 128 else 8
            return torch.tensor([[seq_len // spatial, h, 128]], dtype=torch.long)
    raise ValueError(f"seq {seq_len} incompatible")


def _max_across_ranks(values, group, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.cpu().tolist()


def _parameter_sync_sizes(module):
    replicated = sharded = 0
    for p in module.parameters():
        if getattr(p, '_sp_sharded', False):
            sharded += p.numel()
        else:
            replicated += p.numel()
    return replicated, sharded


def _wrap_ddp(module, group, rank, bucket_cap_mb):
    ignored = [name for name, p in module.named_parameters()
               if getattr(p, '_sp_sharded', False)]
    if ignored:
        DDP._set_params_and_buffers_to_ignore_for_model(module, ignored)
    return DDP(module, device_ids=[rank], output_device=rank, process_group=group,
               broadcast_buffers=False, bucket_cap_mb=bucket_cap_mb,
               gradient_as_bucket_view=True, static_graph=True)


def _run_iteration(train_model, raw_model, x_seed, grad_output, e, grid, context,
                   sync_mode, group, bucket_cap_mb, sp, measure):
    raw_model.zero_grad(set_to_none=True)
    x_local = x_seed.detach().requires_grad_(True)
    if measure:
        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        sync_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = train_model(x_local, e, grid, context)
    if measure:
        fwd_end.record()
    output.backward(grad_output)
    if sync_mode == "ddp":
        for p in raw_model.parameters():
            if getattr(p, '_sp_sharded', False) and p.grad is not None:
                p.grad.div_(sp)
    if measure:
        bwd_end.record()
    if sync_mode == "manual":
        sync_replicated_grads(raw_model, group, bucket_cap_mb=bucket_cap_mb)
    if measure:
        sync_end.record()
        torch.cuda.synchronize()
        return (total_start.elapsed_time(fwd_end),
                fwd_end.elapsed_time(bwd_end),
                bwd_end.elapsed_time(sync_end),
                total_start.elapsed_time(sync_end))
    torch.cuda.synchronize()
    return None


def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(world_size)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=4))
    group = dist.group.WORLD
    device = torch.device(f'cuda:{rank}')

    import deep_gemm  # noqa

    config = Wan21Config()
    sp = world_size

    cond_g = torch.Generator(device=device).manual_seed(5678)
    e = torch.randn(1, 6, config.dim, device=device, dtype=torch.float32, generator=cond_g) * 0.01
    context = torch.randn(1, 512, config.dim, device=device, dtype=torch.bfloat16, generator=cond_g) * 0.02

    seq_lengths = [2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    strategies = [s.strip() for s in args.strategies.split(',')]

    results = {'seq_lens': seq_lengths, 'strategies': strategies, 'data': {}}

    for strategy in strategies:
        results['data'][strategy] = {'fwd': [], 'bwd': [], 'total': [],
                                      'wall': [], 'throughput': [], 'peak_mb': []}

    if rank == 0:
        print(f'\n{"="*120}')
        print(f'Ulysses Variant Throughput Sweep — Wan2.1 T2V-14B ({args.layers} layers, official weights)')
        print(f'Hardware: B300 x{world_size}, SP={sp}, sync_mode={args.sync_mode}')
        print(f'Sequence lengths: {seq_lengths}')
        print(f'Strategies: {strategies}')
        print(f'{"="*120}')
        print(f'{"Strategy":<12} {"Seq":>6} {"FWD(ms)":>10} {"BWD(ms)":>10} '
              f'{"Total(ms)":>10} {"Wall(ms)":>10} {"tok/s":>10} {"Peak(MB)":>10}')
        print('-' * 120)

    for seq_len in seq_lengths:
        if seq_len % sp != 0 or (seq_len // sp) % 128 != 0:
            if rank == 0:
                print(f'  seq={seq_len} skipped (not divisible by SP={sp} or 128)')
            for strategy in strategies:
                for key in ['fwd', 'bwd', 'total', 'wall', 'throughput', 'peak_mb']:
                    results['data'][strategy][key].append(None)
            continue

        grid = _grid_for_sequence(seq_len).to(device)
        local_seq = seq_len // sp

        for strategy in strategies:
            torch.manual_seed(42)
            old = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device(device):
                    raw_model = SPWanTransformer(
                        config, SPConfig(sp_size=sp, group=group, layout='THD'),
                        args.layers, strategy)
            finally:
                torch.set_default_dtype(old)
            raw_model.to(device=device)

            if checkpoint_dir:
                load_and_broadcast_official_parameters(
                    raw_model, checkpoint_dir, group, key_map=raw_model.official_key)
            else:
                for p in raw_model.parameters():
                    dist.broadcast(p.data, src=0, group=group)

            raw_model.setup_shape(1, seq_len, config.num_heads, config.head_dim)
            raw_model.train()

            train_model = raw_model
            if args.sync_mode == 'ddp':
                train_model = _wrap_ddp(raw_model, group, rank, args.bucket_cap_mb)

            g = torch.Generator(device=device).manual_seed(1234 + seq_len)
            x_seed = torch.randn(local_seq, config.dim, device=device,
                                  dtype=torch.bfloat16, generator=g)
            grad_output = torch.randn(local_seq, config.dim, device=device,
                                       dtype=torch.float32, generator=g)

            # Warmup
            for _ in range(args.warmup):
                _run_iteration(train_model, raw_model, x_seed, grad_output, e, grid,
                               context, args.sync_mode, group, args.bucket_cap_mb, sp, False)
                dist.barrier(group)

            torch.cuda.reset_peak_memory_stats(device)
            acc = [0.0, 0.0, 0.0, 0.0]
            for _ in range(args.iters):
                dist.barrier(group)
                t = _run_iteration(train_model, raw_model, x_seed, grad_output, e, grid,
                                   context, args.sync_mode, group, args.bucket_cap_mb, sp, True)
                rank_max = _max_across_ranks(t, group, device)
                acc = [a + b for a, b in zip(acc, rank_max)]

            fwd, bwd, sync, total = [v / args.iters for v in acc]
            wall = total
            tps = seq_len / (wall / 1000.0)
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mb += raw_model.sym_buffer_bytes() / 1024**2
            peak_mb = _max_across_ranks([peak_mb], group, device)[0]

            results['data'][strategy]['fwd'].append(fwd)
            results['data'][strategy]['bwd'].append(bwd)
            results['data'][strategy]['total'].append(total)
            results['data'][strategy]['wall'].append(wall)
            results['data'][strategy]['throughput'].append(tps)
            results['data'][strategy]['peak_mb'].append(peak_mb)

            if rank == 0:
                print(f'{strategy:<12} {seq_len//1024:>4}K {fwd:>9.2f} {bwd:>9.2f} '
                      f'{total:>9.2f} {wall:>9.2f} {tps:>9.0f} {peak_mb:>9.0f}',
                      flush=True)

            # Cleanup
            if hasattr(raw_model, 'destroy_buffers'):
                raw_model.destroy_buffers()
            del train_model, raw_model
            torch.cuda.empty_cache()
            dist.barrier(group)

    if rank == 0:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sweep_throughput.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {out_path}')

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--strategies', default='serial,fused_var')
    parser.add_argument('--sync-mode', default='ddp')
    parser.add_argument('--bucket-cap-mb', type=float, default=64.0)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--repo-id', default=OFFICIAL_REPO_ID)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    ckpt = None
    if not args.synthetic:
        ckpt = resolve_official_checkpoint(args.checkpoint_dir, args.repo_id, None)
    port = find_free_port()
    mp.spawn(run, args=(args.num_gpus, port, args, ckpt), nprocs=args.num_gpus, join=True)
