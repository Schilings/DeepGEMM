"""Per-layer memory breakdown: track what's allocated at each stage.

Measures memory delta after each major step to find the hidden memory cost.
"""

import os, sys, math
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


def run(rank, ng, port, num_layers, seq):
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port),
                       'RANK': str(rank), 'WORLD_SIZE': str(ng)})
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=ng)
    group = dist.group.WORLD
    dev = torch.device(f'cuda:{rank}')
    import deep_gemm
    model_cfg = Wan21Config(dim=5120, num_heads=40, head_dim=128)
    dim, nh, hd = model_cfg.dim, model_cfg.num_heads, model_cfg.head_dim

    grid = torch.tensor([[21, 30, 52]], dtype=torch.long)
    bs = 1; llseq = seq // ng; lm = bs * llseq
    g2 = torch.Generator(device=dev).manual_seed(42)
    X_full = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=dev, generator=g2)
    X_local = X_full[:, rank*llseq:(rank+1)*llseq, :].reshape(llseq, dim).contiguous()

    if rank == 0:
        print(f"\n{'='*100}")
        print(f"  Per-Layer Memory Breakdown — {ng} GPUs, {num_layers} layers, seq={seq}")
        print(f"{'='*100}")

    for strat_name in ['serial', 'fused_var']:
        sp_cfg = SPConfig(sp_size=ng, group=group, layout='THD', use_fused_ops=True)
        layers = nn.ModuleList()
        for i in range(num_layers):
            layer = get_strategy(strat_name, model_cfg, sp_cfg)
            layers.append(layer)
        # Wrap in a Module with forward (FSDP2 requires forward())
        class MultiLayer(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            def forward(self, x, grid, llseq):
                for layer in self.layers:
                    x = layer(x, grid, llseq)
                return x
        model = MultiLayer(layers).to(dev)

        g = torch.Generator(device=dev).manual_seed(42)
        with torch.no_grad():
            for layer in model.layers:
                for p in layer.model.parameters():
                    p.data = torch.randn(p.shape, dtype=p.dtype, device=dev, generator=g) / math.sqrt(dim)
                layer.model = layer.model.to(torch.bfloat16)
                layer.setup_shape(bs, seq, nh, hd)
                if strat_name == 'fused_var' and len(model.layers) > 1 and i > 0:
                    layer._skip_buffer_creation = True
                    layer.setup_shape(bs, seq, nh, hd)
                    layer.share_buffers_from(model.layers[0])

        # FSDP2
        ignored = set()
        for layer in model.layers:
            ignored |= set(layer.model.parameters())
            if strat_name == 'fused_var':
                ignored |= {layer.Wo_r_local}
        apply_fsdp2(model, group, reshard_after_forward=False, ignored_params=ignored)

        # Adam states
        adam_states = []
        for p in model.parameters():
            if p.requires_grad:
                adam_states.append(torch.zeros(p.shape, dtype=torch.float32, device=dev))
                adam_states.append(torch.zeros(p.shape, dtype=torch.float32, device=dev))

        # sym_buf size
        sym_mb = 0.0
        if strat_name == 'fused_var':
            sym_mb = model.layers[0].sym_post.buffer.numel() / 1024 / 1024 + model.layers[0].sym_post_bwd.buffer.numel() / 1024 / 1024

        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

        if rank == 0:
            print(f"\n--- {strat_name} ---")
            print(f"  sym_buf: {sym_mb:.1f} MB")

        # FWD
        X_in = X_local.detach().requires_grad_(True)
        y = model(X_in, grid, llseq)
        torch.cuda.synchronize(dev)
        fwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        # BWD
        gy = torch.randn((lm, dim), dtype=torch.bfloat16, device=dev)
        y.backward(gy)
        torch.cuda.synchronize(dev)
        bwd_peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

        # Count per-layer parameter sizes
        per_layer_param_mb = 0
        per_layer_extra_mb = 0  # non-parameter tensors like Wo_t
        for layer in model.layers:
            for p in layer.parameters():
                per_layer_param_mb += p.numel() * p.element_size()
            if hasattr(layer, 'Wo_t') and isinstance(layer.Wo_t, torch.Tensor):
                per_layer_extra_mb += layer.Wo_t.numel() * layer.Wo_t.element_size()

        per_layer_param_mb = per_layer_param_mb / num_layers / 1024 / 1024
        per_layer_extra_mb = per_layer_extra_mb / num_layers / 1024 / 1024

        if rank == 0:
            print(f"  per-layer params: {per_layer_param_mb:.1f} MB")
            print(f"  per-layer extra tensors (Wo_t etc): {per_layer_extra_mb:.1f} MB")
            print(f"  fwd_peak: {fwd_peak:.1f} MB")
            print(f"  bwd_peak: {bwd_peak:.1f} MB")
            print(f"  true_peak (bwd + sym_buf): {bwd_peak + sym_mb:.1f} MB")

        # Clean up
        if strat_name == 'fused_var' and hasattr(model.layers[0], 'sym_post') and model.layers[0].sym_post is not None:
            model.layers[0].sym_post.destroy()
            model.layers[0].sym_post = None
            model.layers[0].sym_post_bwd.destroy()
            model.layers[0].sym_post_bwd = None
        del model, adam_states, X_in, y, gy
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print(f"\n{'='*100}\n")
    dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    ng = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    seq = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    port = find_free_port()
    print(f"Launching: {ng} GPUs, {num_layers} layers, seq={seq}")
    mp.spawn(run, args=(ng, port, num_layers, seq), nprocs=ng, join=True)
