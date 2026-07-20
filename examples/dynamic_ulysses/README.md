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

## Benchmark Results

### Analytical Model (compute + communication)

| Scenario | Static SP=8 | Dynamic | Speedup |
|----------|------------|---------|---------|
| uniform 8K | 0.173s | 0.082s | **2.12x** |
| uniform 32K | 0.369s | 0.368s | 1.00x |
| mixed | 0.315s | 0.471s | 0.67x |
| all short | 0.040s | 0.022s | **1.86x** |

Geometric mean: **+7.4% vs SP=8, +20% vs SP=4**

### Real GPU (B300 ×8, 4 layers)

| Scenario | Static SP=8 | Dynamic | Speedup |
|----------|------------|---------|---------|
| uniform 8K | 9504ms | 3807ms | **2.50x** |

Key insight: SP=8 with short sequences (1K local_seq) has very high A2A communication-to-compute ratio. Dynamic SP avoids this by using SP=2 for 8K sequences.

## Research Background

- **ByteScale HDP** (ByteDance): Dynamic mesh, data-aware sharding, balance scheduler
- **Megatron Hybrid CP** (NVIDIA): Pre-created power-of-2 NCCL groups, HybridCPDataLoader

Our approach combines both: Megatron-style pre-created groups + ByteScale-style FLOPs scheduling + DeepGEMM Ulysses fused operators.
