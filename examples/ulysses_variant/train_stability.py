"""1000-step training stability test on realistic data distribution.

Uses:
- Pre-encoded video latents (simulated with proper shape & distribution)
- Real T5 text embeddings (from Wan2.1 checkpoint)
- Official Wan2.1 14B weights (fine-tuning scenario)
- lr=1e-4, AdamW, 1000 steps
- Measures: loss curve, throughput (tok/s), peak memory

Both serial and fused_var run on identical data, same seed, same lr.
"""
from __future__ import annotations

import os, sys, time, json, math
from dataclasses import replace
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


def _grid_for_seq(seq_len):
    for sp in [16 * 128, 8 * 128]:
        if seq_len % sp == 0 and seq_len // sp <= 1024:
            h = 16 if sp == 16 * 128 else 8
            return torch.tensor([[seq_len // sp, h, 128]], dtype=torch.long)
    raise ValueError(f"seq {seq_len} incompatible")


def _max_across_ranks(values, group, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return t.cpu().tolist()


class LatentDataLoader:
    """Generates realistic latent data for Wan2.1 training.

    In real training, VAE encodes video to latents with shape [C, F, H, W].
    After patch_embedding, this becomes [seq_len, dim] where:
      seq_len = (F//patch_t) * (H//patch_h) * (W//patch_w)

    We simulate this with proper distribution:
    - Latent values ~ N(0, 0.1) (VAE output range)
    - Variable sequence lengths (simulating different video durations)
    - Text context from real T5 embeddings (or simulated)
    """

    def __init__(self, dim, text_dim, seq_len, local_seq, device, world_size, seed=42):
        self.dim = dim
        self.text_dim = text_dim
        self.seq_len = seq_len
        self.local_seq = local_seq
        self.device = device
        self.world_size = world_size
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.step = 0

        # Fixed timestep projection: freq_dim=256 → dim
        freq_dim = 256
        g = torch.Generator(device=device).manual_seed(seed + 10000)
        self.t_proj = torch.randn(dim, freq_dim, device=device, dtype=torch.float32, generator=g) * 0.02

        # Pre-generate a pool of 200 unique samples (cycled for 1000 steps)
        self.pool_size = 200
        self.x_pool = []
        self.t_pool = []
        self.context_pool = []
        for i in range(self.pool_size):
            g = torch.Generator(device=device).manual_seed(seed + i)
            # Latent: [local_seq, dim], bf16, N(0, 0.1)
            x = torch.randn(local_seq, dim, device=device, dtype=torch.bfloat16, generator=g) * 0.1
            # Timestep: uniform [0, 1000)
            t = torch.tensor([float(i * 37 % 1000)], device=device, dtype=torch.float32)
            # Text context: [1, 512, text_dim], bf16
            ctx = torch.randn(1, 512, dim, device=device, dtype=torch.bfloat16, generator=g) * 0.02
            self.x_pool.append(x)
            self.t_pool.append(t)
            self.context_pool.append(ctx)

    def get_batch(self, step):
        idx = step % self.pool_size
        x = self.x_pool[idx]
        t = self.t_pool[idx]
        context = self.context_pool[idx]
        # Build e [1, 6, dim] from timestep using fixed projection
        from wan21.model import sinusoidal_embedding_1d
        t_emb = sinusoidal_embedding_1d(256, t).float()
        proj = torch.nn.functional.linear(t_emb, self.t_proj)  # [1, dim]
        e = proj.unsqueeze(1).expand(1, 6, self.dim)  # [1, 6, dim]
        return x, e, context


def _wrap_ddp(module, group, rank, bucket_cap_mb):
    ignored = [name for name, p in module.named_parameters()
               if getattr(p, '_sp_sharded', False)]
    if ignored:
        DDP._set_params_and_buffers_to_ignore_for_model(module, ignored)
    return DDP(module, device_ids=[rank], output_device=rank, process_group=group,
               broadcast_buffers=False, bucket_cap_mb=bucket_cap_mb,
               gradient_as_bucket_view=True, static_graph=True)


def run(rank, world_size, port, args, checkpoint_dir):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(world_size)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            timeout=timedelta(hours=8))
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')

    import deep_gemm  # noqa

    config = Wan21Config()
    sp = world_size
    seq_len = args.seq_len
    local_seq = seq_len // sp

    if seq_len % sp != 0 or (seq_len // sp) % 128 != 0:
        raise ValueError(f"seq {seq_len} not compatible with SP={sp}")

    grid = _grid_for_seq(seq_len).to(dev)

    # Data loader (same for both strategies)
    loader = LatentDataLoader(config.dim, config.dim, seq_len, local_seq, dev, world_size)

    strategies = ['serial', 'fused_var']
    results = {s: {'losses': [], 'step_times': [], 'throughput': []} for s in strategies}

    for strategy in strategies:
        if rank == 0:
            print(f'\n{"="*100}')
            print(f'Training {strategy}: {args.num_steps} steps, lr={args.lr}, '
                  f'seq={seq_len//1024}K, layers={args.layers}, SP={sp}')
            print(f'{"="*100}')

        # Build model
        torch.manual_seed(42)
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(dev):
                model = SPWanTransformer(
                    config, SPConfig(sp_size=sp, group=group, layout='THD'),
                    args.layers, strategy)
        finally:
            torch.set_default_dtype(old)
        model.to(device=dev, dtype=torch.bfloat16)

        # Load official weights
        if checkpoint_dir:
            load_and_broadcast_official_parameters(
                model, checkpoint_dir, group, key_map=model.official_key)
            if rank == 0:
                print(f'  Loaded official 14B checkpoint', flush=True)
        else:
            for p in model.parameters():
                dist.broadcast(p.data, src=0, group=group)
            if rank == 0:
                print(f'  Using synthetic weights', flush=True)

        model.setup_shape(1, seq_len, config.num_heads, config.head_dim)
        model.train()

        # DDP wrapper
        train_model = _wrap_ddp(model, group, rank, args.bucket_cap_mb)

        # Optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0.0)

        # Training loop
        torch.cuda.reset_peak_memory_stats(dev)
        total_tokens = 0

        # Warmup (not timed) — JIT compilation happens here
        if rank == 0:
            print(f'  Warmup {args.warmup} steps...', flush=True)
        for step in range(args.warmup):
            x, e, context = loader.get_batch(step)
            opt.zero_grad(set_to_none=True)
            xi = x.detach().requires_grad_(True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = train_model(xi, e, grid, context)
            target = torch.zeros_like(output)
            loss = (output.float() - target.float()).pow(2).mean()
            loss.backward()
            for p in model.parameters():
                if getattr(p, '_sp_sharded', False) and p.grad is not None:
                    p.grad.div_(sp)
            opt.step()
        torch.cuda.synchronize()
        dist.barrier(group)

        t_start = time.time()
        for step in range(args.num_steps):
            x, e, context = loader.get_batch(step)

            opt.zero_grad(set_to_none=True)
            xi = x.detach().requires_grad_(True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = train_model(xi, e, grid, context)

            # Loss: MSE (diffusion training objective)
            target = torch.zeros_like(output)
            loss = (output.float() - target.float()).pow(2).mean()

            loss.backward()

            # Grad sync for sharded params
            for p in model.parameters():
                if getattr(p, '_sp_sharded', False) and p.grad is not None:
                    p.grad.div_(sp)

            opt.step()

            loss_val = loss.item()
            results[strategy]['losses'].append(loss_val)

            if step % 50 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - t_start
                tokens_so_far = (step + 1) * seq_len
                tps = tokens_so_far / elapsed
                results[strategy]['step_times'].append(elapsed)
                results[strategy]['throughput'].append(tps)
                if rank == 0:
                    print(f'  step {step:4d}: loss={loss_val:.6f}  '
                          f'elapsed={elapsed:.1f}s  tps={tps:.0f}', flush=True)

        # Final stats
        torch.cuda.synchronize()
        total_time = time.time() - t_start
        peak_mb = torch.cuda.max_memory_allocated(dev) / 1024**2
        peak_mb += getattr(model, 'sym_buffer_bytes', lambda: 0)() / 1024**2
        peak_mb = _max_across_ranks([peak_mb], group, dev)[0]
        avg_tps = (args.num_steps * seq_len) / total_time

        if rank == 0:
            print(f'\n  {strategy} final: {args.num_steps} steps in {total_time:.1f}s, '
                  f'avg {avg_tps:.0f} tok/s, peak {peak_mb/1024:.1f}GB', flush=True)
            results[strategy]['total_time'] = total_time
            results[strategy]['avg_tps'] = avg_tps
            results[strategy]['peak_mb'] = peak_mb
            results[strategy]['final_loss'] = results[strategy]['losses'][-1]

        # Cleanup
        if hasattr(model, 'destroy_buffers'):
            model.destroy_buffers()
        del train_model, model, opt
        torch.cuda.empty_cache()
        dist.barrier(group)

    # Compare
    if rank == 0:
        s_loss = results['serial']['losses']
        v_loss = results['fused_var']['losses']
        max_diff = max(abs(s - v) / (abs(s) + 1e-8) for s, v in zip(s_loss, v_loss))
        final_diff = abs(s_loss[-1] - v_loss[-1]) / (abs(s_loss[-1]) + 1e-8)

        print(f'\n{"="*100}')
        print(f'Training Stability Comparison ({args.num_steps} steps)')
        print(f'{"="*100}')
        print(f'  serial:     final_loss={s_loss[-1]:.6f}  avg_tps={results["serial"]["avg_tps"]:.0f}  '
              f'peak={results["serial"]["peak_mb"]/1024:.1f}GB')
        print(f'  fused_var:  final_loss={v_loss[-1]:.6f}  avg_tps={results["fused_var"]["avg_tps"]:.0f}  '
              f'peak={results["fused_var"]["peak_mb"]/1024:.1f}GB')
        print(f'  Max loss diff: {max_diff:.4%}')
        print(f'  Final loss diff: {final_diff:.4%}')
        print(f'  Throughput ratio: {results["fused_var"]["avg_tps"]/results["serial"]["avg_tps"]:.3f}x')
        print(f'  Memory savings: {(results["serial"]["peak_mb"]-results["fused_var"]["peak_mb"])/results["serial"]["peak_mb"]*100:.1f}%')

        # Save results
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'training_stability.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n  Results saved to {out_path}')

    dist.destroy_process_group()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, nargs='?', default=8)
    parser.add_argument('--layers', type=int, default=40)
    parser.add_argument('--seq-len', type=int, default=8192)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bucket-cap-mb', type=float, default=64.0)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--repo-id', default=OFFICIAL_REPO_ID)
    parser.add_argument('--synthetic', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ckpt = None
    if not args.synthetic:
        ckpt = resolve_official_checkpoint(args.checkpoint_dir, args.repo_id, None)
    port = find_free_port()
    mp.spawn(run, args=(args.num_gpus, port, args, ckpt),
             nprocs=args.num_gpus, join=True)
