"""Strictly load the complete official Wan2.1 T2V-14B and run forward."""

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
from wan21.model import WanModel


def test_full_forward(checkpoint_dir):
    device = torch.device("cuda")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = WanModel(preset="14B")
    finally:
        torch.set_default_dtype(old_dtype)
    model.to(device=device)

    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {count / 1e9:.3f}B")
    loaded, elements = load_official_parameters(model, checkpoint_dir)
    print(f"Strictly loaded {loaded} tensors / {elements / 1e9:.3f}B parameters")

    x_video = [torch.randn(16, 4, 64, 64, dtype=torch.bfloat16, device=device) * 0.02]
    timestep = torch.tensor([500.0], dtype=torch.float32, device=device)
    context = [torch.randn(512, 4096, dtype=torch.bfloat16, device=device) * 0.02]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x_video, timestep, context, seq_len=4096)

    expected = tuple(x_video[0].shape)
    actual = tuple(output[0].shape)
    if actual != expected:
        raise AssertionError(f"Output shape mismatch: expected {expected}, got {actual}")
    print(f"Forward OK: {expected} -> {actual}")
    print(
        f"Output mean={output[0].mean().item():.6f}, "
        f"std={output[0].std().item():.6f}, norm={output[0].norm().item():.4f}"
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
    test_full_forward(checkpoint)
