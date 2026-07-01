"""Real FSDP2 (torch.distributed.fsdp.fully_shard) for Wan2.1 SP attention.

FSDP2 is available via `from torch.distributed.fsdp import fully_shard` (composable API).
It uses DTensor internally: params sharded on dim-0, forward pre-hook all-gathers,
backward hooks all-gather params + reduce-scatter gradients. AUTOMATIC — no manual all-reduce.

Key for fused_var: Wo is row-split (N-sharded) → add to `ignored_params` so FSDP
does NOT shard/reduce Wo (its grad is already local). Wqkv is replicated → FSDP
shards it and reduce-scatters its grad in backward automatically.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def get_fsdp2_mesh(group: dist.ProcessGroup) -> DeviceMesh:
    """Create a DeviceMesh for FSDP2 from an existing process group."""
    ng = group.size()
    # DeviceMesh needs a list of global ranks; for a single-node SP group this is 0..ng-1
    mesh = DeviceMesh("cuda", list(range(ng)))
    return mesh


def apply_fsdp2(module: nn.Module, group: dist.ProcessGroup,
                reshard_after_forward: bool = True,
                ignored_params=None) -> nn.Module:
    """Apply FSDP2 to a module using fully_shard (composable API).

    Args:
        module: The module to shard (e.g. UlyssesBase strategy)
        group: Process group for the SP/data-parallel mesh
        reshard_after_forward: If True, free unsharded params after forward (save memory)
        ignored_params: set of params to NOT shard/reduce (e.g. fused_var's Wo_r_local)

    After this, calling module.backward() (or loss.backward()) will AUTOMATICALLY
    reduce-scatter weight gradients via DTensor backward hooks. No manual sync needed.

    For bf16 training, mp_policy keeps params/grads in bf16.
    """
    mesh = get_fsdp2_mesh(group)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    fully_shard(
        module,
        mesh=mesh,
        reshard_after_forward=False,  # Keep params unsharded after forward for backward
        mp_policy=mp_policy,
        ignored_params=ignored_params,
    )
    return module
