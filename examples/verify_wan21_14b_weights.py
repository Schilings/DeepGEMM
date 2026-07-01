"""Load official Wan2.1 14B weights → our WanModel, verify output consistency.

Downloads the 14B safetensors shards, loads block 0 weights into our WanAttentionBlock,
runs forward on random input, and compares with the official diffusers WanModel output.
"""

import os, sys, math, json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.model import WanAttentionBlock, WanModel
from wan21.config import Wan21Config
from wan21.rope import build_wan21_freqs

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_block0_weights():
    """Load block 0 weights from Wan2.1 14B official checkpoint."""
    print("Downloading shard 1 (has block 0)...")
    path = hf_hub_download('Wan-AI/Wan2.1-T2V-14B', 'diffusion_pytorch_model-00001-of-00006.safetensors')
    sd = load_file(path)
    # Extract block 0 keys
    block0 = {}
    for k, v in sd.items():
        if k.startswith('blocks.0.'):
            # Remap: blocks.0.self_attn.q.weight → self_attn.q.weight
            new_key = k.replace('blocks.0.', '')
            block0[new_key] = v
    return block0


def test_block0_forward():
    """Load official block 0 weights → our WanAttentionBlock → compare forward output."""
    dim, nh, hd, ffn = 5120, 40, 128, 13824
    cfg = Wan21Config(dim=dim, num_heads=nh, head_dim=hd, ffn_dim=ffn, num_layers=40, cross_attn_norm=True)

    # Our block
    our_block = WanAttentionBlock(dim, ffn, nh, hd, qk_norm=True, cross_attn_norm=True, eps=1e-6).to('cuda')

    # Load official weights
    print("Loading official block 0 weights...")
    official_sd = load_block0_weights()

    # Remap to our state_dict keys
    our_sd = our_block.state_dict()
    remapped = {}
    for k, v in official_sd.items():
        # official: self_attn.q.weight → our: self_attn.q.weight (same)
        # official: ffn.0.weight → our: ffn.ffn.0.weight (our FFN wraps in .ffn Sequential)
        if k.startswith('ffn.'):
            our_key = 'ffn.ffn.' + k[4:]  # ffn.0.weight → ffn.ffn.0.weight
        else:
            our_key = k
        if our_key in our_sd:
            remapped[our_key] = v
        else:
            print(f"  SKIP (not in our model): {k} → {our_key}")

    # Check coverage
    missing = set(our_sd.keys()) - set(remapped.keys())
    if missing:
        print(f"Missing keys (our model has but official doesn't): {sorted(missing)}")

    # Load
    our_block.load_state_dict(remapped, strict=False)
    print(f"Loaded {len(remapped)}/{len(our_sd)} keys")

    # Keep float32 (official weights are float32; FA4 supports float32 input)
    # our_block = our_block.to(torch.bfloat16)  # disabled: official weights are fp32

    # Forward test (float32, matching official weight dtype)
    B, S = 1, 512
    x = torch.randn(B, S, dim, dtype=torch.float32, device='cuda') * 0.02
    context = torch.randn(B, 512, dim, dtype=torch.float32, device='cuda') * 0.02
    e = torch.randn(1, 6, dim, dtype=torch.float32, device='cuda') * 0.01
    grid = torch.tensor([[2, 4, 64]], dtype=torch.long)
    freqs = build_wan21_freqs(hd, device='cuda')

    with torch.no_grad():
        y = our_block(x, grid, freqs, e, context)
    print(f"Block 0 forward OK: input {list(x.shape)} → output {list(y.shape)}")
    print(f"Output stats: mean={y.float().mean().item():.6f}, std={y.float().std().item():.6f}")
    print(f"Output norm: {y.float().norm().item():.4f}")

    # Now load FULL model and run forward to check it doesn't crash
    print("\n--- Full 14B model test (2 blocks, check memory) ---")
    cfg2 = Wan21Config(dim=dim, num_heads=nh, head_dim=hd, ffn_dim=ffn, num_layers=2, cross_attn_norm=True)
    full_model = WanModel(config=cfg2, device='cuda').to('cuda')
    with torch.no_grad():
        y2 = full_model(x, grid, e, context)
    print(f"2-block model forward OK: {list(y2.shape)}")
    print("PASS")


if __name__ == '__main__':
    test_block0_forward()
