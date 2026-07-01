"""Multi-layer full training memory benchmark: serial vs fused_var.

Measures REAL peak GPU memory including:
  - Weights + gradients + Adam optimizer states (fp32 m, v)
  - Activations (stored for backward, scale with num_layers)
  - sym_buf (symm_mem, NOT tracked by torch.cuda.max_memory_allocated — added manually)
  - FSDP2 unshard buffers

sym_buf is reused across all layers (allocated once, shared) — the practical approach.

Usage: python examples/bench_wan21_mem_train.py <num_gpus> [num_layers] [seq]
"""

import os, sys, math, argparse
import torch, torch.nn as nn, torch.distributed as dist, torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.config import Wan21Config, SPConfig
from wan21.fsdp2_utils import apply_fsdp2
from wan21.bench_utils import find_free_port


def get_strategy(name, cfg, sp_cfg):
    if name == 'serial':
        from wan21.sp.serial import SerialUlysses; return SerialUlysses(cfg, sp_cfg)
    elif name == 'fused_var':
        from wan21.sp.fused_variant import FusedVariantUlysses; return FusedVariantUlysses(cfg, sp_cfg)
    raise ValueError(name)


class MultiLayerModel(nn.Module):
    """N attention layers chained: x → layer1 → layer2 → ... → layerN → out."""
    def __init__(self, cfg, sp_cfg, num_layers, strategy_name):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = get_strategy(strategy_name, cfg, sp_cfg)
            self.layers.append(layer)

    def setup_shape(self, bs, seq, nheads, head_dim):
        """Set up shapes for all layers; reuse sym_buf across layers (layer 0 owns it)."""
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer._skip_buffer_creation = True
            layer.setup_shape(bs, seq, nheads, head_dim)
        # Share sym_buf from layer 0 to layers 1..N
        owner = self.layers[0]
        for i in range(1, len(self.layers)):
            if hasattr(owner, 'sym_post') and owner.sym_post is not None:
                self.layers[i].share_buffers_from(owner)

    def forward(self, x, grid, llseq):
        for layer in self.layers:
            x = layer(x, grid, llseq)
        return x

    def destroy_buffers(self):
        if self.layers and hasattr(self.layers[0], '_unified_buf') and self.layers[0]._unified_buf is not None:
            self.layers[0]._unified_buf.destroy()
            self.layers[0]._unified_buf = None
            self.layers[0].sym_post = None
            self.layers[0].sym_post_bwd = None
        elif self.layers and hasattr(self.layers[0], 'sym_post'):
            if hasattr(self.layers[0], 'sym_post') and self.layers[0].sym_post is not None:
                self.layers[0].sym_post.destroy()
                self.layers[0].sym_post = None
            if hasattr(self.layers[0], 'sym_post_bwd') and self.layers[0].sym_post_bwd is not None:
                self.layers[0].sym_post_bwd.destroy()
                self.layers[0].sym_post_bwd = None

    def sym_buf_bytes(self):
        """Total sym_buf allocation (not tracked by PyTorch allocator)."""
        total = 0
        owner = self.layers[0]
        if hasattr(owner, '_unified_buf') and owner._unified_buf is not None:
            # Unified: single shared buffer, don't double-count
            total = owner._unified_buf.ag.buffer.numel()
        else:
            if hasattr(owner, 'sym_post') and owner.sym_post is not None:
                total += owner.sym_post.buffer.numel()
            if hasattr(owner, 'sym_post_bwd') and owner.sym_post_bwd is not None:
                total += owner.sym_post_bwd.buffer.numel()
        return total


def run(rank, ng, port, num_layers, seq, strategies):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim

    # Use a real Wan2.1 shape (grid product must be <= seq)
    grid = torch.tensor([[21, 30, 52]], dtype=torch.long)  # 21*30*52 = 32760 ≤ 32768
    bs = 1; llseq = seq // ng; lm = bs * llseq
    g2 = torch.Generator(device=dev).manual_seed(42)
    X_full = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=dev, generator=g2)
    X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, dim).contiguous()

    if rank == 0:
        print(f"\n{'='*145}")
        print(f"  Wan2.1 14B Multi-Layer Training Memory Benchmark — {ng} GPUs, {num_layers} layers, seq={seq}")
        print(f"  dim={dim} nh={nh} hd={hd} sp={ng} local_nh={nh//ng} local_hidden={nh//ng*hd}")
        print(f"  sym_buf reused across layers | FSDP2 (fully_shard) | Adam states (fp32 m,v) allocated")
        print(f"{'='*145}")
        print(f"{'strategy':<12} | {'weights(MB)':>11} {'grads(MB)':>11} {'adam(MB)':>11} {'fwd_peak(MB)':>11} {'bwd_peak(MB)':>11} {'sym_buf(MB)':>11} | {'true_peak(MB)':>13}")
        print('-' * 145)

    for strat_name in strategies:
        sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
        model = MultiLayerModel(model_cfg, sp_cfg, num_layers, strat_name).to(dev)

        # Init weights
        g = torch.Generator(device=dev).manual_seed(42)
        with torch.no_grad():
            for layer in model.layers:
                for p in layer.model.parameters():
                    p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
                layer.model = layer.model.to(torch.bfloat16)
        model.setup_shape(bs, seq, nh, hd)

        # FSDP2: shard Wqkv per layer; ignore model params + Wo_r_local for fused_var
        ignored = set()
        for layer in model.layers:
            ignored |= set(layer.model.parameters())
            if strat_name == 'fused_var':
                ignored |= {layer.Wo_r_local}
        apply_fsdp2(model, group, reshard_after_forward=False, ignored_params=ignored)

        # Count trainable params
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Adam optimizer states (fp32 m, v) — allocate manually to measure real memory
        # (FSDP2 DTensor + Adam.step() has compatibility issues, so we measure states directly)
        adam_states = []
        for p in model.parameters():
            if p.requires_grad:
                adam_states.append(torch.zeros(p.shape, dtype=torch.float32, device=dev))  # exp_avg
                adam_states.append(torch.zeros(p.shape, dtype=torch.float32, device=dev))  # exp_avg_sq

        sym_buf_mb = model.sym_buf_bytes() / 1024 / 1024

        # Weights: bf16 (2 bytes)
        weights_mb = n_params * 2 / 1024 / 1024
        # Adam states: fp32 m+v (4 bytes each, 2 states)
        adam_mb = n_params * 4 * 2 / 1024 / 1024
        # Gradients: bf16 (same size as weights)
        grads_mb = weights_mb

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

        baseline = torch.cuda.memory_allocated(dev) / 1024 / 1024

        # --- FWD ---
        X_in = X_local.detach().requires_grad_(True)
        y = model(X_in, grid, llseq)
        torch.cuda.synchronize(dev)
        fwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        # --- BWD (gradients allocated) ---
        gy = torch.randn((lm, dim), dtype=torch.bfloat16, device=dev)
        y.backward(gy)
        torch.cuda.synchronize(dev)
        bwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        # True peak = PyTorch tracked + sym_buf (not tracked by allocator)
        true_peak = bwd_peak + sym_buf_mb

        # Clean up adam states
        del adam_states

        if rank == 0:
            print(f"{strat_name:<12} | {weights_mb:>11.1f} {grads_mb:>11.1f} {adam_mb:>11.1f} {fwd_peak:>11.1f} {bwd_peak:>11.1f} {sym_buf_mb:>11.1f} | {true_peak:>13.1f}")

        model.destroy_buffers()
        del model, X_in, y, gy
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print('=' * 145 + '\n')
    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    strategies = sys.argv[4].split(',') if len(sys.argv) > 4 else ['serial', 'fused_var']
    port = find_free_port()
    print(f"Launching: {ng} GPUs, {num_layers} layers, seq={seq}, strategies={strategies}")
    mp.spawn(run, args=(ng, port, num_layers, seq, strategies), nprocs=ng, join=True)
