# A2A GEMM (All2All + GEMM) Iteration Log

## Baseline (commit 8814016)

**Geo Mean: 1.239x** | Config: mc2, 256T, launch_bounds(256,1), kNumReadyChunksPerSlot=4, rank-order chunk copies

---

## Iter 1: `__launch_bounds__(256, 2)` (commit 4c500a9) ✅ KEEP

**Direction**: Increase min_blocks_per_SM 1→2 for better occupancy.
**Result**: 1.239x → **1.264x** (+2.0%)

---

## Iter 2a: Skip self-rank barrier polling ❌ REVERTED

**Direction**: Skip `ld_acq_sys` polling for self-rank tiles (data always ready after local_ready_event).
**Result**: 256×2048×2048 FAIL (max_diff=218). Race condition between local_ready_event and flag visibility.
**Lesson**: `local_ready_event` does not guarantee all flag writes are visible to the compute kernel on current_stream.

## Iter 2b: Two-phase barrier polling (ld.cg + ld.acquire.sys) ❌ REVERTED

**Direction**: Spin with cached `ld.cg` then confirm with `ld.acquire.sys`.
**Result**: 1.264x → 1.263x (-0.08%). Extra ld.acquire.sys per flag check is pure overhead; `ld.cg` sees stale L2 values.
**Lesson**: `ld.acquire.sys` is already efficient on NVLink; adding a cached pre-check is counterproductive.

---

## Iter 3: Interleave chunk copies across ranks (commit a458c6a) ✅ KEEP

**Direction**: Change remote copy order from [rank×chunk] to [chunk×rank].
Old: rank A chunk 0,1,2,3 → rank B chunk 0,1,2,3 → ...
New: chunk 0 of all ranks → chunk 1 of all ranks → ...

This lets the kernel see chunk 0 data from multiple ranks sooner, improving comm-compute overlap.

**Result**: 1.264x → **1.344x** (+6.3%)

| Shape | Baseline | Iter 1 | Iter 3 | Total Δ |
|---|---|---|---|---|
| 1024×4096×4096 | 1.335x | 1.380x | 1.425x | +6.7% |
| 1024×7168×4096 | 1.222x | 1.179x | 1.896x | **+55.2%** |
| 2048×4096×7168 | 1.383x | 1.401x | 1.406x | +1.7% |
| 2048×7168×4096 | 1.139x | 1.141x | 1.136x | -0.3% |
| 4096×4096×4096 | 1.243x | 1.449x | 1.419x | +14.2% |
| 4096×7168×4096 | 1.345x | 1.204x | 1.186x | -11.8% |
| 4096×4096×7168 | 1.159x | 1.578x | 1.507x | +30.0% |
| 8192×4096×4096 | 1.582x | 1.560x | 1.357x | -14.2% |
| 8192×7168×4096 | 1.346x | 1.347x | 1.362x | +1.2% |
| 8192×7168×7168 | 1.182x | 1.179x | 1.181x | -0.1% |
| 2048×7168×2048 | 1.091x | 1.073x | 1.853x | **+69.8%** |
| 4096×7168×2048 | 1.202x | 1.211x | 1.187x | -1.2% |
| 16384×7168×4096 | 1.201x | 1.120x | 1.169x | -2.7% |
| 16384×7168×7168 | 1.012x | 1.026x | 1.017x | +0.5% |

**Geo Mean**: 1.239x → 1.264x → **1.344x** | **Avg Fused**: 1105.7T → 1129.3T → 1119.9T

**Key insight**: Interleaving chunk copies dramatically helps small K shapes (2048, 4096) where A2A comm latency
relative to GEMM compute is high. The kernel can start processing chunk 0 from rank (i-1) while chunk 1
data from the same rank is still in flight.

---


## Iter 4: Mixed chunk interleave (chunk 0 interleave + chunk 1+ rank-order) ❌ REVERTED

**Direction**: Combine chunk 0 interleave (for early data) with rank-order for chunk 1+ (for NVLink continuity).
**Result**: 1.344x → 1.289x (-4.1%). Mixed strategy breaks NVLink transfer continuity for later chunks.
**Lesson**: Either fully interleave or fully rank-order; mixing patterns confuses the NVLink scheduler.

## Iter 5: kNumReadyChunksPerSlot 4→2 ❌ REVERTED

**Direction**: Reduce per-slot chunk count from 4 to 2, hoping to reduce flag overhead.
**Result**: Geo Mean 1.150x → 1.144x (-0.5%). Larger chunks reduce comm-compute overlap granularity.
**Lesson**: More fine-grained chunks improve overlap opportunity. Reducing chunks hurts.

## Iter 5b: kNumReadyChunksPerSlot 4→8 ❌ REVERTED

**Direction**: Increase per-slot chunk count to 8 for finer-grained overlap.
**Result**: Geo Mean 1.150x → 0.748x (-35%). 8 flags per rank × 8 ranks = massive memset + flag overhead.
**Lesson**: More chunks ≠ more overlap. Flag management overhead grows faster than overlap benefit.

## Iter 6a: Dual-stream (self on current_stream) ❌ REVERTED

**Direction**: Move self chunk copy + flag to current_stream, remote copies on comm_stream.
**Result**: Geo Mean 1.150x → 0.977x (-15%). Reversed stream dependency increases comm_stream wait time.
**Lesson**: Keep all communication on comm_stream; current_stream should only wait for local_ready_event.

## Iter 6b: Remove #pragma unroll on polling loop ❌ REVERTED

**Direction**: Let compiler decide loop unrolling for the kNumReadyChunksPerSlot polling loop.
**Result**: Geo Mean 1.150x → 0.956x (-17%). Unrolled loop with conditional skip is faster than rolled loop.
**Lesson**:  with branch is optimal for small constant iteration counts.

## Iter 6c: Skip self-rank flag polling ❌ REVERTED

**Direction**: Skip  polling when src_rank == rank_idx (self data always ready).
**Result**: Geo Mean 1.150x → 0.958x (-17%). Branch divergence penalty > polling savings.
**Lesson**: Warp-level conditional skip for self-rank doesn't help due to branch divergence.

## Iter 7: Rank-order copy + batched flag setting (commit 15b6d33) ✅ KEEP

**Direction**: Replace per-chunk flag memset with per-rank batched flag setting, and switch back to rank-order copy.
- Old: chunk-interleave + per-chunk  after each copy → 32 memset calls
- New: rank-order + batched  per rank after all rank's chunks copied → 8 memset calls

Rank-order lets NVLink transfer data more continuously for each source rank.
Batched flags reduce host-side API call overhead by 4x.

**Result**: Geo Mean 1.150x → **1.217x** (+5.8%) | Avg Fused 1140.8T → **1187.9T** (+4.1%)

| Shape | Iter 3 (20iter) | Iter 7 (20iter) | Δ |
|---|---|---|---|
| 1024×4096×4096 | 0.917x | **1.200x** | **+30.9%** |
| 1024×7168×4096 | 0.935x | **1.200x** | +28.3% |
| 2048×4096×7168 | 1.021x | **1.319x** | +29.2% |
| 2048×7168×4096 | 0.965x | **1.122x** | +16.3% |
| 4096×4096×4096 | 1.129x | **1.450x** | +28.4% |
| 4096×7168×4096 | 0.993x | **1.342x** | +35.1% |
| 4096×4096×7168 | 1.120x | **1.499x** | +33.8% |
| 8192×4096×4096 | 1.193x | **1.610x** | +34.9% |
| 8192×7168×4096 | 1.190x | **1.187x** | -0.3% |
| 8192×7168×7168 | 1.074x | **0.990x** | -7.8% |
| 2048×7168×2048 | 0.979x | **1.113x** | +13.7% |
| 4096×7168×2048 | 1.045x | **1.154x** | +10.4% |
| 16384×7168×4096 | 1.068x | **1.040x** | -2.6% |
| 16384×7168×7168 | 0.979x | **0.997x** | +1.8% |

**Geo Mean**: 1.150x → **1.217x** | **Avg Fused**: 1140.8T → **1187.9T**

**Key insight**: Reducing host-side API call overhead (32→8 cudaMemsetAsync) is more impactful than
the theoretical overlap benefit of chunk-interleave. Rank-order + batched flags gives the best of both:
NVLink continuity per-rank AND fewer host-side calls. All shapes now have speedup >= 0.99x.

---

## Summary (CUDA events timing)

| Iter | Geo Mean | Avg Fused TFLOPS | Status |
|------|----------|-------------------|--------|
| Baseline | 1.239x (Python) | 1105.7T (Python) | — |
| Iter 1 | +2.0% | — | ✅ launch_bounds(256,2) |
| Iter 3 | +6.3% (Python) | — | ✅ chunk-interleave |
| Iter 3 (CUDA events) | 1.150x | 1140.8T | Reference |
| **Iter 7** | **1.217x** | **1187.9T** | ✅ **Current best** |

## Iter 8: Merge per-rank chunks into single memcpy (commit dc2bc09) ✅ KEEP

**Direction**: Replace per-chunk memcpy (4 calls/rank) with single per-rank memcpy (1 call/rank).
- Old: rank-order with 4 chunk-sized  per rank + batched flag
- New: rank-order with 1 full-rank  per rank + batched flag
- Total host API calls: 64 (iter 3) → 32 (iter 7) → **16** (iter 8)

**Result**: Geo Mean 1.217x → **1.298x** (+6.7%) | Avg Fused 1187.9T → **1236.7T** (+4.1%)

| Shape | Iter 7 (20iter) | Iter 8 (20iter) | Δ |
|---|---|---|---|
| 1024×4096×4096 | 1.200x | **1.868x** | **+55.7%** |
| 1024×7168×4096 | 1.200x | **1.259x** | +4.9% |
| 2048×4096×7168 | 1.319x | **1.477x** | +12.0% |
| 2048×7168×4096 | 1.122x | **1.169x** | +4.2% |
| 4096×4096×4096 | 1.450x | **1.505x** | +3.8% |
| 4096×7168×4096 | 1.342x | **1.373x** | +2.3% |
| 4096×4096×7168 | 1.499x | **1.634x** | +9.0% |
| 8192×4096×4096 | 1.610x | **1.631x** | +1.3% |
| 8192×7168×4096 | 1.187x | **1.183x** | -0.3% |
| 8192×7168×7168 | 0.990x | **0.992x** | +0.2% |
| 2048×7168×2048 | 1.113x | **1.184x** | +6.4% |
| 4096×7168×2048 | 1.154x | **1.198x** | +3.8% |
| 16384×7168×4096 | 1.040x | **1.030x** | -1.0% |
| 16384×7168×7168 | 0.997x | **1.005x** | +0.8% |

**Key insight**: Merging chunk-level memcpy into rank-level memcpy massively reduces host-side API call
overhead (cudaMemcpyAsync 32→8). The GPU CE can handle large contiguous transfers more efficiently
than many small fragmented ones. All shapes now have speedup >= 0.992x.

---

## Updated Summary (CUDA events timing)

| Iter | Geo Mean | Avg Fused TFLOPS | Status |
|------|----------|-------------------|--------|
| Baseline | 1.239x (Python) | 1105.7T (Python) | — |
| Iter 1 | +2.0% | — | ✅ launch_bounds(256,2) |
| Iter 3 (CUDA events) | 1.150x | 1140.8T | ✅ chunk-interleave |
| Iter 7 | 1.217x | 1187.9T | ✅ rank-order + batched flags |
| **Iter 8** | **1.298x** | **1236.7T** | ✅ **Current best** |

## Multi-GPU Scalability (Iter 8, CUDA events, 10 iters)

| GPU Count | Geo Mean Speedup | Avg Fused TFLOPS |
|-----------|------------------|-------------------|
| **2卡** | **1.238x** | 1158.8T |
| **4卡** | **1.283x** | 1214.0T |
| **8卡** | **1.289x** | 1247.5T |

| Shape | 2卡 | 4卡 | 8卡 | 趋势 |
|-------|-----|-----|-----|------|
| 1024×4096×4096 | 1.820x | 1.938x | 2.098x | ↑ 卡越多收益越大 |
| 1024×7168×4096 | 1.586x | 1.318x | 1.380x | ~持平 |
| 2048×4096×7168 | 1.289x | 1.304x | 1.327x | ↑ |
| 2048×7168×4096 | 1.061x | 1.143x | 1.125x | ~持平 |
| 4096×4096×4096 | 1.317x | 1.391x | 1.395x | ↑ |
| 4096×7168×4096 | 1.106x | 1.105x | 1.199x | ↑ 8卡更优 |
| 4096×4096×7168 | 1.272x | 1.332x | 1.481x | ↑↑ |
| 8192×4096×4096 | 1.347x | 1.393x | 1.628x | ↑↑ |
| 8192×7168×4096 | 1.089x | 1.253x | 1.323x | ↑↑ |
| 8192×7168×7168 | 1.060x | 1.281x | 1.048x | 4卡最优 |
| 2048×7168×2048 | 1.126x | 1.174x | 1.177x | ↑ |
| 4096×7168×2048 | 1.130x | 1.147x | 1.180x | ↑ |
| 16384×7168×4096 | 1.181x | 1.292x | 1.078x | 4卡最优 |
| 16384×7168×7168 | 1.160x | 1.076x | 0.968x | ↓ 卡越多越差 |

**Key findings**:
1. Geo Mean 随卡数增加而提升但边际递减：2卡→4卡 +3.6%, 4卡→8卡 +0.5%
2. 小 shape (M≤4096)：卡越多收益越大，A2A 通信占比大时重叠收益显著
3. 大 shape (M≥8192, N×K 也大)：8卡反而不如 4卡，如 16384×7168×7168 从 1.160x 降到 0.968x
4. 4卡是部分大 shape 的甜蜜点
5. 2卡始终有正收益 (≥1.061x)


## Communication vs Compute Breakdown (Iter 8, CUDA events, 10 iters)

### 8卡关键数据

| Shape | A2A ms | GEMM ms | Fused ms | 理论max(A2A,GEMM) | 实际Overlap% | A2A占比 |
|-------|--------|---------|----------|-------------------|-------------|---------|
| 1024×4096×4096 | 0.172 | 0.164 | 0.242 | 0.172 | 27.9% | 51.3% |
| 8192×4096×4096 | 1.029 | 1.295 | 1.612 | 1.295 | 30.6% | 44.3% |
| 4096×4096×7168 | 0.844 | 1.143 | 1.505 | 1.143 | 24.3% | 42.5% |
| 8192×7168×7168 | 1.917 | 4.438 | 6.039 | 4.438 | 5.0% | 30.2% |
| 16384×7168×7168 | 3.214 | 9.806 | 13.002 | 9.806 | 0.1% | 24.7% |

### 结论

**瓶颈是 A2A 通信**，但不是简单的GEMM 等 A2A 串行，而是：

1. **A2A 占比 > 40% → overlap 好生效** (20-33% overlap)。此时 GEMM 可以在等数据时持续算 self-rank tile
2. **A2A 占比 < 30% → overlap 几乎失效** (0-5%)。GEMM 主导但 fused kernel 的 polling/CE 开销仍存，fused ≈ A2A + GEMM
3. **16384×7168×7168 极端案例**：理论 fused 应 ≈ GEMM=9.8ms，实际 13.0ms。额外 3.2ms = A2A 时间，说明 A2A 完全没被隐藏

### 根因：Flux-style 重叠的结构性局限

- Self-rank tile 可立即计算（数据本地）
- Remote-rank tile 必须等 A2A flag → kernel 在 polling 时**空转**
- Shape 越大 → remote rank tile 越多 → 等待比例越高 → overlap 效果越差
- 8卡 A2A 绝对时间更长 (7 vs 3 远端) + flag 开销更大 (28 vs 12 flags) → 更难 overlap

