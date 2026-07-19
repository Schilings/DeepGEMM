"""Strictly load official Wan2.1 T2V-14B block 0 and run forward."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from wan21.checkpoint import (
    OFFICIAL_REPO_ID,
    load_official_parameters,
    resolve_official_checkpoint,
)
from wan21.model import WanAttentionBlock, build_wan21_freqs


def block0_official_key(local_name):
    return "blocks.0." + local_name.replace("ffn.ffn.", "ffn.")


def test_block0_forward(checkpoint_dir):
    dim, num_heads, head_dim, ffn_dim = 5120, 40, 128, 13824
    device = torch.device("cuda")
    with torch.device(device):
        block = WanAttentionBlock(
            dim, ffn_dim, num_heads, head_dim,
            qk_norm=True, cross_attn_norm=True, eps=1e-6,
        )

    loaded, elements = load_official_parameters(
        block, checkpoint_dir, key_map=block0_official_key
    )
    print(f"Strictly loaded block 0: {loaded} tensors / {elements / 1e6:.1f}M parameters")

    batch, sequence = 1, 512
    x = torch.randn(batch, sequence, dim, dtype=torch.float32, device=device) * 0.02
    context = torch.randn(batch, 512, dim, dtype=torch.float32, device=device) * 0.02
    e = torch.randn(batch, 6, dim, dtype=torch.float32, device=device) * 0.01
    grid = torch.tensor([[2, 4, 64]], dtype=torch.long)
    freqs = build_wan21_freqs(head_dim, device=device)

    with torch.no_grad():
        output = block(x, e, grid, freqs, context)
    if output.shape != x.shape:
        raise AssertionError(f"Output shape mismatch: {output.shape} != {x.shape}")
    print(f"Block 0 forward OK: {tuple(x.shape)} -> {tuple(output.shape)}")
    print(
        f"Output mean={output.float().mean().item():.6f}, "
        f"std={output.float().std().item():.6f}, norm={output.float().norm().item():.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = resolve_official_checkpoint(
        args.checkpoint_dir, args.repo_id, args.revision
    )
    test_block0_forward(checkpoint)
