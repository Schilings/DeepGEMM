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
| `bench_train.py` | Simplified GPU benchmark (UlyssesScatterAttn, multiple static baselines) |
| `bench_wan21_14b.py` | **Primary**: real Wan2.1 T2V-14B training throughput benchmark |
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

### Real Wan2.1 14B Training Benchmark

**Primary benchmark** — measures real training throughput (tokens/s) on the
complete Wan2.1 T2V-14B transformer (40 blocks, official weights).

Run: `python examples/dynamic_ulysses/bench_wan21_14b.py 8`

**Design**: Dynamic SP×DP vs Static SP=8. Both arms use the same model,
weights, data, and gradient sync. The ONLY difference is SP×DP scheduling:

- **Static SP=8**: all sequences use SP=8 (dp=1), processed sequentially
- **Dynamic SP×DP**: each sequence assigned optimal SP size; sequences with
  the same (sp_size, seq_len) run as parallel DP copies (dp_size = world_size / sp_size)

Key insight: for weight gradients, SP all-reduce and DP all-reduce are the
SAME operation (cross-rank gradient aggregation). So SP size can be dynamically
adjusted across the SP×DP process grid.

#### Results (B300 ×8, 40 layers, 14.056B params, official weights)

| Scenario | Tokens | Static SP=8 (tok/s) | Dynamic SP×DP (tok/s) | Speedup | Dyn Schedule |
|----------|-------:|--------------------:|----------------------:|--------:|--------------|
| uniform_8K×8 | 65,536 | 31,804 | 48,548 | **1.527x** | {2: 8} |
| uniform_32K×2 | 65,536 | 36,648 | 38,250 | **1.044x** | {4: 2} |
| mixed | 77,824 | 28,979 | 21,246 | 0.733x | {4:2, 2:4, 1:2} |
| all_short_2K×8 | 16,384 | 8,736 | 39,814 | **4.558x** | {1: 8} |
| bimodal | 77,824 | 25,213 | 38,568 | **1.530x** | {4:2, 1:6} |
| one_long_tail | 47,104 | 19,188 | 23,168 | **1.207x** | {4:1, 1:7} |

**Geometric mean: 1.464x** (Dynamic SP×DP is 46% faster than Static SP=8)

#### Analysis

Dynamic SP×DP wins in 5 out of 6 scenarios:

1. **all_short_2K (4.558x)**: 8 short sequences use SP=1 (pure DP), all 8 GPUs
   process different sequences in parallel. Static SP=8 wastes A2A overhead on
   sequences too short to benefit from sequence parallelism.

2. **uniform_8K (1.527x)**: 8 medium sequences use SP=2 (4 DP copies), 4
   sequences processed in parallel per round vs 8 sequential in static.

3. **bimodal (1.530x)**: 2 long sequences use SP=4, 6 short sequences use SP=1.
   Short sequences finish fast in parallel while long sequences get SP benefit.

4. **one_long_tail (1.207x)**: 1 long sequence uses SP=4, 7 short use SP=1.
   The long sequence is the bottleneck but short sequences parallelize well.

5. **uniform_32K (1.044x)**: 2 long sequences use SP=4 (2 DP copies). Nearly
   break-even — long sequences benefit from large SP, DP parallelism limited.

6. **mixed (0.733x)**: Diverse lengths cause many (sp_size, seq_len) groups,
   each requiring separate rounds with dummy fill. Scheduling overhead dominates.

Control variables (identical for both arms):
- **Model**: `SPWanTransformer` with `SerialUlysses` (same code path)
- **Weights**: official `Wan-AI/Wan2.1-T2V-14B` checkpoint (14.056B params)
- **Data**: same input sequences and conditioning
- **Grad sync**: manual all-reduce across all ranks
- **Total tokens**: identical per scenario

### Simplified Benchmark (for quick iteration)

`bench_train.py` uses a simplified `UlyssesScatterAttn` (4 layers, no FFN/cross-attn)
with multiple static baselines (SP8/SP4×2/SP2×4/SP1×8). Useful for rapid
development testing when the full 14B model is too slow.

Run: `python examples/dynamic_ulysses/bench_train.py 8`

## Research Background

- **ByteScale HDP** (ByteDance): Dynamic mesh, data-aware sharding, balance scheduler
- **Megatron Hybrid CP** (NVIDIA): Pre-created power-of-2 NCCL groups, HybridCPDataLoader

Our approach combines both: Megatron-style pre-created groups + ByteScale-style FLOPs scheduling + DeepGEMM Ulysses fused operators.
