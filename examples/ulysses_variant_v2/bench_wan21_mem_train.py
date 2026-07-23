"""Multi-layer training memory benchmark: serial vs fused_var vs fused_var_v2.

Measures weights, gradients, FP32 Adam states, saved activations and the shared
symmetric workspace.  v2 has the same memory footprint as v1 (Wo sharded) — the
only difference is the backward computation order, not the tensor sizes.

Usage:
    DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD \\
    python3 examples/ulysses_variant_v2/bench_wan21_mem_train.py 8 40 32768 serial,fused_var,fused_var_v2
"""
import argparse
import os, sys, math
import torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

from wan21.autograd_ops_v2 import finalize_deferred_grads
from wan21.bench_utils import find_free_port
from wan21.checkpoint import (
    OFFICIAL_REPO_ID,
    load_and_broadcast_official_parameters,
    resolve_official_checkpoint,
)
from wan21.config import Wan21Config, SPConfig


def get_strategy(name, cfg, sp_cfg):
    if name == 'serial':
        from wan21.sp.serial import SerialUlysses; return SerialUlysses(cfg, sp_cfg)
    elif name == 'fused':
        from wan21.sp.fused import FusedUlysses; return FusedUlysses(cfg, sp_cfg)
    elif name == 'fused_var':
        from wan21.sp.variant import FusedVariantUlysses; return FusedVariantUlysses(cfg, sp_cfg)
    elif name == 'fused_var_v2':
        from wan21.sp.variant_v2 import FusedVariantV2Ulysses; return FusedVariantV2Ulysses(cfg, sp_cfg)
    raise ValueError(name)


def _official_key(local_name: str) -> str:
    return local_name.replace("layers.", "blocks.").replace(".model.", ".self_attn.")


class MultiLayerModel(nn.Module):
    def __init__(self, cfg, sp_cfg, num_layers, strategy_name):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = get_strategy(strategy_name, cfg, sp_cfg)
            self.layers.append(layer)

    def setup_shape(self, bs, seq, nheads, head_dim):
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer._skip_buffer_creation = True
            layer.setup_shape(bs, seq, nheads, head_dim)
        owner = self.layers[0]
        for i in range(1, len(self.layers)):
            if hasattr(owner, 'sym_post') and owner.sym_post is not None:
                self.layers[i].share_buffers_from(owner)

    def forward(self, x, grid, llseq):
        for layer in self.layers:
            x = layer(x, grid, llseq)
        return x

    def destroy_buffers(self):
        for layer in self.layers:
            layer.destroy_buffers()

    def sym_buf_bytes(self):
        if not self.layers:
            return 0
        workspace = getattr(self.layers[0], 'sym_post', None)
        return 0 if workspace is None else workspace.num_bytes


def run(rank, ng, port, args, checkpoint_dir):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim

    seq = args.seq
    assert seq % (16 * 128) == 0
    grid = torch.tensor([[seq // (16 * 128), 16, 128]], dtype=torch.long)
    bs = 1; llseq = seq // ng; lm = bs * llseq
    g2 = torch.Generator(device=dev).manual_seed(42 + rank)
    X_local = torch.randn(llseq, dim, dtype=torch.bfloat16, device=dev, generator=g2)

    if rank == 0:
        print(f"\n{'='*145}")
        print(f"  Wan2.1 14B Multi-Layer Training Memory Benchmark (v2) — {ng} GPUs, {args.num_layers} layers, seq={seq}")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng} local_hidden={nh//ng*hd}")
        print(f"  SP-only (DP=1, no FSDP across SP) | sym_buf shared across layers | Adam fp32 m/v")
        weight_src = f"official checkpoint: {checkpoint_dir}" if checkpoint_dir else "synthetic random weights"
        print(f"  Weights: {weight_src}")
        print(f"{'='*145}")
        print(f"{'strategy':<16} | {'weights(MB)':>11} {'grads(MB)':>11} {'adam(MB)':>11} {'fwd_peak(MB)':>11} {'bwd_peak(MB)':>11} {'sym_buf(MB)':>11} | {'true_peak(MB)':>13}")
        print('-' * 145)

    for strat_name in args.strategies.split(','):
        sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(dev):
                model = MultiLayerModel(model_cfg, sp_cfg, args.num_layers, strat_name)
        finally:
            torch.set_default_dtype(old_dtype)
        model.to(device=dev)

        if checkpoint_dir is not None:
            loaded, elements = load_and_broadcast_official_parameters(
                model, checkpoint_dir, group, key_map=_official_key,
            )
            if rank == 0:
                print(f"{strat_name}: strictly loaded {loaded} tensors / "
                      f"{elements / 1e9:.3f}B params", flush=True)
        else:
            g = torch.Generator(device=dev).manual_seed(42)
            with torch.no_grad():
                for layer in model.layers:
                    for p in layer.model.parameters():
                        p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
            for p in model.parameters():
                dist.broadcast(p.data, src=0, group=group)

        model.setup_shape(bs, seq, nh, hd)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adam_states = []
        for p in model.parameters():
            if p.requires_grad:
                adam_states.append(torch.zeros_like(p, dtype=torch.float32))
                adam_states.append(torch.zeros_like(p, dtype=torch.float32))

        sym_buf_mb = model.sym_buf_bytes() / 1024 / 1024
        weights_mb = n_params * 2 / 1024 / 1024
        adam_mb = n_params * 4 * 2 / 1024 / 1024
        grads_mb = weights_mb

        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

        X_in = X_local.detach().requires_grad_(True)
        y = model(X_in, grid, llseq)
        torch.cuda.synchronize(dev)
        fwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        gy = torch.randn((lm, dim), dtype=torch.bfloat16, device=dev)
        y.backward(gy)
        finalize_deferred_grads(model)
        torch.cuda.synchronize(dev)
        bwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        true_peak = bwd_peak + sym_buf_mb

        del adam_states

        if rank == 0:
            print(f"{strat_name:<16} | {weights_mb:>11.1f} {grads_mb:>11.1f} {adam_mb:>11.1f} {fwd_peak:>11.1f} {bwd_peak:>11.1f} {sym_buf_mb:>11.1f} | {true_peak:>13.1f}")

        model.destroy_buffers()
        del model, X_in, y, gy
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print('=' * 145 + '\n')
    dist.destroy_process_group()
    os._exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_gpus", type=int, nargs="?", default=8)
    parser.add_argument("num_layers", type=int, nargs="?", default=40)
    parser.add_argument("seq", type=int, nargs="?", default=32768)
    parser.add_argument("strategies", nargs="?", default="serial,fused_var,fused_var_v2")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_args()
    local_checkpoint = None
    if not cli_args.synthetic:
        local_checkpoint = resolve_official_checkpoint(
            cli_args.checkpoint_dir, cli_args.repo_id, cli_args.revision
        )
        print(f"Official checkpoint: {local_checkpoint}")
    else:
        print("WARNING: using synthetic weights by explicit request")
    port = find_free_port()
    print(f"Launching: {cli_args.num_gpus} GPUs, {cli_args.num_layers} layers, "
          f"seq={cli_args.seq}, strategies={cli_args.strategies}")
    mp.spawn(run, args=(cli_args.num_gpus, port, cli_args, local_checkpoint),
             nprocs=cli_args.num_gpus, join=True)
