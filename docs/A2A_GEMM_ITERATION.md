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
