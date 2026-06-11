# GEMM+RS v2: Push-based Single-Kernel Fusion

## Motivation

v1 (Pull-based) achieves 0.65x geo_mean speedup vs NCCL separate, with target scenarios
(large K) reaching 1.0-1.7x. However, the fundamental bottleneck is:

**Comm warps do 8 NVLink P2P reads** (from 8 remote ranks) for each tile's reduce.
These reads have ~200ns latency each, serialized per-rank, and cannot be fully hidden.

Flux (ByteDance, SM90) solves this differently:
- Epilogue **pushes** GEMM output directly to remote ranks' buffers (NVLink write)
- Remote rank's reduce warps read from **local** buffer (data already pushed there)
- NVLink write latency is absorbed into the Epilogue (overlapped with GEMM computation)

## v2 Design: Push-based

### Data Flow

```
Rank A's GEMM produces tile T (belongs to Rank B's chunk):
  1. Epilogue: TMEM → smem → NVLink WRITE to Rank B's reduce_buffer[slot=A][tile_T]
  2. Epilogue: set flag in Rank B's flag array: flags[A][tile_T] = 1
  3. Rank B's Reduce warp: poll LOCAL flags[*][tile_T] for all ranks
  4. When all ranks' flags are set: LOCAL HBM read all partials → FP32 reduce → write output

Key difference from v1:
- v1: Epilogue writes LOCAL buffer → Comm warp P2P READS remote buffers
- v2: Epilogue P2P WRITES remote buffer → Reduce warp READS local buffer
```

### Why Push is Better

1. **NVLink write is in the Epilogue** — overlapped with next tile's GEMM computation
2. **Reduce does only LOCAL reads** — HBM bandwidth (~8 TB/s on B300) >> NVLink (900 GB/s)
3. **No per-rank serial polling** — all data arrives independently, reduce starts when ALL ready

### Symmetric Buffer Layout (v2)

```
Each rank has:
  reduce_buffer[num_ranks][max_tokens_per_rank][hidden]  — partial results FROM each src rank
  ready_flags[num_ranks][num_m_blocks][num_n_blocks]     — per-tile flags FROM each src rank

Rank A writing tile T (dst_rank = B):
  → writes to B's reduce_buffer[slot=A][tile_T_rows][tile_T_cols]
  → sets B's ready_flags[slot=A][m_block][n_block] = 1

Rank B's Reduce warp processing tile T:
  → polls LOCAL ready_flags[slot=0..7][m_block][n_block]
  → when all 8 flags set: read LOCAL reduce_buffer[0..7][tile_rows][tile_cols]
  → FP32 accumulate → write output
```

### Warp Layout (unchanged from v1)

```
W0-3 (128T, 48 regs): Reduce Warps — poll local flags + local HBM read + reduce
W4 (32T, 40 regs): TMA Load A+B
W5 (32T, 40 regs): Reserved
W6 (32T, 40 regs): MMA Issue (UMMA 2x1SM)
W7 (32T, 40 regs): Reserved
W8-11 (128T, 208 regs): Epilogue — TMEM → smem → NVLink PUSH to remote
```

### Key Implementation Changes from v1

1. **Epilogue store target**: `workspace.get_partial_ptr(dst_rank, ...)` → `sym_buffer.map(local_ptr, dst_rank)`
   - Write to REMOTE rank's buffer instead of local
   - `st_rel_sys` the flag in REMOTE rank's flag array

2. **Reduce warp reads**: All from LOCAL buffer
   - No `sym_buffer.map()` needed for data reads
   - Only poll LOCAL flags (no NVLink reads for flags either!)

3. **Flag location**: Stored at DESTINATION rank (not source rank)
   - v1: flags in source rank's buffer, comm warp P2P reads remote flags
   - v2: flags in dest rank's buffer, pushed by source's epilogue

### Expected Performance Impact

- Eliminate 8 NVLink reads per tile in Comm/Reduce phase → LOCAL HBM reads only
- NVLink writes in Epilogue overlap with next tile's GEMM (pipeline)
- Expected: geo_mean 0.65x → 0.8-1.0x (most shapes should improve)
- Target scenarios (large K): 1.7x → 2-3x
