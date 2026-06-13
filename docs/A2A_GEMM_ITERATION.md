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

