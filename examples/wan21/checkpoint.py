"""Strict, streaming loaders for the official Wan2.1 T2V-14B checkpoint."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.distributed as dist


OFFICIAL_REPO_ID = "Wan-AI/Wan2.1-T2V-14B"
INDEX_FILENAME = "diffusion_pytorch_model.safetensors.index.json"


def resolve_official_checkpoint(
    checkpoint_dir: str | None = None,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str | None = None,
) -> str:
    """Return a local checkpoint directory, downloading indexed shards if needed."""
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir).expanduser().resolve()
        if not (path / INDEX_FILENAME).is_file():
            raise FileNotFoundError(f"Missing {INDEX_FILENAME} under {path}")
        return str(path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when --checkpoint-dir is not provided"
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[INDEX_FILENAME, "diffusion_pytorch_model-*.safetensors"],
    )


def _read_weight_map(checkpoint_dir: str) -> dict[str, str]:
    index_path = os.path.join(checkpoint_dir, INDEX_FILENAME)
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    return weight_map


def wan_model_official_key(local_name: str) -> str:
    """Map this repository's wrapped FFN parameter names to official names."""
    return local_name.replace(".ffn.ffn.", ".ffn.")


def load_official_parameters(
    module: torch.nn.Module,
    checkpoint_dir: str,
    key_map: Mapping[str, str] | Callable[[str], str] | None = None,
) -> tuple[int, int]:
    """Strictly stream all parameters required by ``module`` from safetensors.

    Only one tensor is materialized on CPU at a time. Missing keys and shape
    mismatches are fatal; silently retaining random initialization is forbidden.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load Wan2.1 weights") from exc

    parameters = dict(module.named_parameters())
    if not parameters:
        raise ValueError("The target module has no parameters")

    def official_name(local_name: str) -> str:
        if key_map is None:
            return wan_model_official_key(local_name)
        if callable(key_map):
            return key_map(local_name)
        try:
            return key_map[local_name]
        except KeyError as exc:
            raise KeyError(f"No official checkpoint mapping for {local_name}") from exc

    weight_map = _read_weight_map(checkpoint_dir)
    by_shard: dict[str, list[tuple[str, str, torch.nn.Parameter]]] = defaultdict(list)
    missing = []
    for local_name, parameter in parameters.items():
        checkpoint_name = official_name(local_name)
        shard_name = weight_map.get(checkpoint_name)
        if shard_name is None:
            missing.append(f"{local_name} -> {checkpoint_name}")
        else:
            by_shard[shard_name].append((local_name, checkpoint_name, parameter))
    if missing:
        preview = "\n  ".join(missing[:20])
        raise KeyError(f"Official checkpoint is missing {len(missing)} required keys:\n  {preview}")

    loaded_tensors = 0
    loaded_elements = 0
    with torch.no_grad():
        for shard_name in sorted(by_shard):
            shard_path = os.path.join(checkpoint_dir, shard_name)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Missing checkpoint shard {shard_path}")
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                available = set(shard.keys())
                for local_name, checkpoint_name, parameter in by_shard[shard_name]:
                    if checkpoint_name not in available:
                        raise KeyError(f"{checkpoint_name} is absent from {shard_name}")
                    tensor = shard.get_tensor(checkpoint_name)
                    if tuple(tensor.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"Shape mismatch for {local_name}: model={tuple(parameter.shape)}, "
                            f"checkpoint={tuple(tensor.shape)}"
                        )
                    parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
                    loaded_tensors += 1
                    loaded_elements += parameter.numel()
                    del tensor

    if loaded_tensors != len(parameters):
        raise RuntimeError(f"Loaded {loaded_tensors}/{len(parameters)} parameters")
    return loaded_tensors, loaded_elements


def load_and_broadcast_official_parameters(
    module: torch.nn.Module,
    checkpoint_dir: str,
    group,
    key_map: Mapping[str, str] | Callable[[str], str] | None = None,
    src: int = 0,
) -> tuple[int, int]:
    """Load once on ``src`` and broadcast the model to the other local ranks."""
    global_rank = dist.get_rank()
    error = None
    stats = (0, 0)
    if global_rank == src:
        try:
            stats = load_official_parameters(module, checkpoint_dir, key_map)
        except Exception as exc:  # propagate failure before parameter broadcasts
            error = exc
            print(f"Official checkpoint load failed: {exc}", flush=True)

    first_parameter = next(module.parameters())
    status = torch.tensor([error is None], dtype=torch.int32, device=first_parameter.device)
    dist.broadcast(status, src=src, group=group)
    if not status.item():
        if error is not None:
            raise error
        raise RuntimeError("Official checkpoint loading failed on source rank")

    stats_tensor = torch.tensor(stats, dtype=torch.int64, device=first_parameter.device)
    dist.broadcast(stats_tensor, src=src, group=group)
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src, group=group)
    return int(stats_tensor[0].item()), int(stats_tensor[1].item())
