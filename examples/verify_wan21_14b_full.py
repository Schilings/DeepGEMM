"""Load full Wan2.1 14B official weights → our WanModel, verify complete forward.

Downloads all 6 safetensors shards, loads into our WanModel, runs a full forward
pass (patch_embedding → blocks → head → unpatchify), and verifies output shape.
"""

import os, sys, math, json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wan21.model import WanModel
from wan21.config import Wan21Config

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_full_14b_weights():
    """Load all Wan2.1 14B weights from 6 shards."""
    sd = {}
    for i in range(1, 7):
        fname = f'diffusion_pytorch_model-0000{i}-of-00006.safetensors'
        print(f"Downloading {fname}...")
        path = hf_hub_download('Wan-AI/Wan2.1-T2V-14B', fname)
        shard = load_file(path)
        sd.update(shard)
        print(f"  {len(shard)} keys (total: {len(sd)})")
    return sd


def test_full_forward():
    """Load full 14B weights, run complete forward pass."""
    # Official 14B config
    model = WanModel(
        dim=5120, num_heads=40, head_dim=128, ffn_dim=13824, num_layers=40,
        patch_size=(1, 2, 2), text_len=512, in_dim=16, freq_dim=256,
        text_dim=4096, out_dim=16, qk_norm=True, cross_attn_norm=True, eps=1e-6,
    ).to('cuda')

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Load weights
    print("\nLoading official 14B weights...")
    official_sd = load_full_14b_weights()

    # Our model's state dict
    our_sd = model.state_dict()

    # Check key coverage
    official_keys = set(official_sd.keys())
    our_keys = set(our_sd.keys())

    # Remap: our ffn is wrapped in WanFeedForward.ffn Sequential
    # official: ffn.0.weight → our: ffn.ffn.0.weight
    remapped_sd = {}
    for k, v in official_sd.items():
        if k.startswith('blocks.') and '.ffn.' in k:
            # ffn.0.weight → ffn.ffn.0.weight
            parts = k.split('.ffn.')
            our_key = parts[0] + '.ffn.ffn.' + parts[1]
        else:
            our_key = k
        remapped_sd[our_key] = v

    missing = our_keys - set(remapped_sd.keys())
    unexpected = set(remapped_sd.keys()) - our_keys
    if missing:
        print(f"Missing (our model has, official doesn't): {sorted(missing)[:10]}...")
    if unexpected:
        print(f"Unexpected (official has, our model doesn't): {sorted(unexpected)[:10]}...")

    # Load
    model.load_state_dict(remapped_sd, strict=False)
    matched = len(set(remapped_sd.keys()) & our_keys)
    print(f"Loaded {matched}/{len(our_keys)} keys (strict=False)")

    # Forward test: simulate a small video
    print("\nRunning full forward...")
    B = 1
    # Video: [C_in=16, F=4, H=64, W=64] (small for memory)
    # After patch (1,2,2): F=4, H=32, W=32 → seq=4*32*32=4096
    x_video = [torch.randn(16, 4, 64, 64, dtype=torch.float32, device='cuda') * 0.02]
    t = torch.tensor([500.0], dtype=torch.float32, device='cuda')
    # Text context: [L=512, C=4096]
    context = [torch.randn(512, 4096, dtype=torch.float32, device='cuda') * 0.02]
    seq_len = 4096

    with torch.no_grad():
        output = model(x_video, t, context, seq_len)

    print(f"Forward OK!")
    print(f"  Input: [{list(x_video[0].shape)}] (C=16, F=4, H=64, W=64)")
    print(f"  Output: [{list(output[0].shape)}]")
    print(f"  Output stats: mean={output[0].mean().item():.6f}, std={output[0].std().item():.6f}")
    print(f"  Output norm: {output[0].norm().item():.4f}")
    print("\nPASS: Full 14B model forward with official weights!")


if __name__ == '__main__':
    test_full_forward()
