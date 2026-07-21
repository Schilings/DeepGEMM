# Dynamic Ulysses SP

Dynamic Sequence Parallelism framework for Wan2.1 training, addressing DP load imbalance in static SP×DP configurations.

## Problem

Static SP×DP (e.g., 4SP×2DP) suffers from **long-tail latency**: when DP groups process different-length sequences, all ranks must wait for the slowest group during gradient sync.

## Solution

Dynamically adjust SP group size per microbatch based on sequence length:
- **Long sequences** → large SP group (e.g., SP=8) for memory efficiency
- **Short sequences** → small SP group (e.g., SP=1) to avoid A2A communication overhead
- **All ranks** participate in gradient sync regardless of SP group size

## Architecture

```
┌─────────────────────────┐
│  DynamicSPGroupManager  │  Pre-creates NCCL groups for SP sizes {1,2,4,8}
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  BalancedDataLoader     │  FLOPs-aware sequence → SP group assignment
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  DynamicUlyssesLayer    │  Runtime SP-size-aware attention (SP=1 pure DP, SP>1 Ulysses A2A)
└───────────┬─────────────┘
            │
┌───────────▼─────────────┐
│  DynamicGradientSync    │  Bucketed AllReduce with token-count scaling
└─────────────────────────┘
```

## Components

| File | Description |
|------|-------------|
| `sp_group_manager.py` | Pre-creates power-of-2 SP/DP NCCL groups |
| `balanced_loader.py` | FLOPs-aware sequence packing and SP assignment |
| `dynamic_ulysses.py` | Runtime SP-size-aware attention layer |
| `grad_sync.py` | Bucketed gradient sync across all ranks |
| `dynamic_trainer.py` | Complete training loop with microbatch scheduling |
| `test_dynamic_sp.py` | Basic functionality tests |
| `test_correctness.py` | Correctness verification (5 tests) |
| `bench_dynamic_sp.py` | Analytical FLOPs+comm model benchmark |
| `bench_train.py` | Real GPU training benchmark |
| `DESIGN.md` | Full design document |

## Usage

```python
from dynamic_ulysses import DynamicSPGroupManager, BalancedDataLoader

# Initialize
gm = DynamicSPGroupManager(world_size=8)
loader = BalancedDataLoader(world_size=8)

# Schedule microbatches
seq_lengths = [32768, 8192, 4096, 2048]
microbatches = loader.schedule(seq_lengths)
# → [Microbatch(sp=4, seq=32768), Microbatch(sp=2, seq=8192), ...]

# Process each microbatch with its SP size
for mb in microbatches:
    info = gm.get_groups(mb.sp_size)
    # Set SP config on model layers
    # Run forward + backward
    # Gradient sync after all microbatches
```

## Benchmark Design — Controlled Experiment

The Dynamic SP benchmark isolates the effect of *dynamic SP selection* by holding
all other variables constant between arms.

### Control Variables (identical across all arms)

| Variable | Value |
|----------|-------|
| Attention implementation | `UlyssesScatterAttn` — single code path for all arms |
| Model weights & shapes | Same `dim=5120, heads=40, head_dim=128, layers=4` |
| Input sequences | Same per scenario |
| DP parallelism model | DP copies run in parallel rounds in ALL arms |

The `UlyssesScatterAttn` implementation handles both SP=1 (A2A is a no-op) and
SP>1 (real A2A scatter/gather) in the same code path, so there is no confounding
from different kernel implementations.

### Independent Variable — SP Scheduling Strategy

| Arm | SP Scheduling |
|-----|---------------|
| Static-SP8 | Every sequence at SP=8, processed sequentially (no DP) |
| Static-SP4×2 | Every sequence at SP=4, 2 DP copies parallel per round |
| Static-SP2×4 | Every sequence at SP=2, 4 DP copies parallel per round |
| Static-SP1×8 | Every sequence at SP=1, 8 DP copies parallel per round (pure DP) |
| **Dynamic** | `BalancedDataLoader` assigns SP per sequence; DP copies within each SP group run in parallel |

The "Best Static" baseline for each scenario is the fastest static arm. Dynamic
SP's speedup is measured against this best-case static baseline, making it a
fair (conservative) comparison.

### Why This Matters

Previous benchmark versions compared Dynamic SP (which used `forward_dp` for
SP=1, a different code path with no A2A) against Static SP=8 (which always used
`forward_sp` with A2A). That conflated two effects:
1. The benefit of dynamic SP selection
2. The benefit of avoiding A2A communication entirely

The new design uses one unified code path, so any performance difference is
attributable solely to the SP scheduling strategy.

### Analytical Model (compute + communication)

See `bench_dynamic_sp.py` for the FLOPs-based analytical model (no GPU needed).

### Real GPU (B300 ×8, 4 layers)

Run: `python examples/dynamic_ulysses/bench_train.py 8`

Output table columns:
- `SP8 / SP4x2 / SP2x4 / SP1x8`: wall-clock (ms) for each static baseline
- `Dynamic`: wall-clock (ms) for dynamic SP
- `Best Static`: fastest static arm name
- `Dyn/Best`: speedup of Dynamic vs best static (>=1.0 means Dynamic wins)

## Research Background

- **ByteScale HDP** (ByteDance): Dynamic mesh, data-aware sharding, balance scheduler
- **Megatron Hybrid CP** (NVIDIA): Pre-created power-of-2 NCCL groups, HybridCPDataLoader

Our approach combines both: Megatron-style pre-created groups + ByteScale-style FLOPs scheduling + DeepGEMM Ulysses fused operators.
