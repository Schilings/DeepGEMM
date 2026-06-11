# AKO4ALL Iteration Log — GEMM+RS Kernel Optimization

## Summary

| Iter | Direction | Geo Mean Speedup | Best Shape | Status |
|------|-----------|-----------------|------------|--------|
| baseline | multicast=2, current code | 0.357x | 0.525x (16384×7168×7168) | ✅ |
| 1 | Distributed ready flag polling (4 warps) | 0.362x | 0.543x | ✅ +1.4% |
| 2 | __nanosleep in polling (reverted) | 0.361x | 0.534x | ❌ no improvement |
| 3 | Parallel TMA store (128 threads) | 0.388x | 0.538x | ✅ +7.2% |
| 4 | kNumTMAStoreStages 2→3 | 0.418x | 0.786x | ✅ +7.7% |
| 5 | Pre-compute base ptr in reduce (reverted) | 0.385x | 0.513x | ❌ -7.9% regression |
| 6 | Remove unused Comm smem buffer (+1 pipeline stage) | 0.487x | 1.694x | ✅✅ +16.5%! First >1x! |
| 7 | kNumTMAStoreStages 3→2, pipeline 7→8 stages | 0.601x | 2.196x | ✅✅✅ +23.4%! |
| 8 | STORE_BLOCK_N=128 + 1 store stage (reverted) | 0.528x | 1.328x | ❌ -12.1% |
| 9 | Replace TMA 1D store with vectorized global STG | 0.654x | 2.087x | ✅✅ +8.8%! 6/21 wins! |

## Analysis after Iter 1-2

**Key insight**: The bottleneck is NOT Comm warp efficiency. It's the GEMM pipeline throughput.

Evidence:
- Separate timing ≈ pure GEMM time (NCCL RS only ~4μs due to NVLink 900 GB/s)
- Fused kernel = 2.1x slower than standard GEMM for same compute
- Comm warp optimizations (polling parallelism, nanosleep) have <2% impact

**Root cause**: Epilogue writes to partial buffer using per-row TMA 1D bulk copies
(128 serial TMA store operations per tile). This is much slower than standard GEMM's
TMA 2D store epilogue. The slow Epilogue blocks the TMEM pipeline, causing MMA to
stall waiting for tmem_empty_barriers.

**Next priority (P0)**: Replace per-row TMA 1D store with TMA 2D store for partial buffer.
This requires:
1. Creating a TMA tensor map descriptor for the partial buffer (in JIT runtime)
2. Using `cute::SM90_TMA_STORE_2D::copy` in Epilogue instead of per-row ptx::tma_store_1d
3. Adjusting partial buffer layout to match TMA 2D requirements (swizzle/alignment)

Expected impact: 2x Epilogue speedup → overall geo_mean from 0.36x to ~0.6-0.7x

## Baseline

- **Date**: 2026-06-11
- **Kernel**: `sm100_bf16_gemm_rs.cuh` (multicast=2 enabled)
- **Config**: 8× B300 SXM6, 148 SMs, cluster_dim=2
- **Correctness**: 6/6 ALL PASS
- **Performance**: Geo Mean = 0.357x vs NCCL separate (GEMM + reduce_scatter)
- **Peak fusion TFLOPS**: ~600 (vs ~1100 standard GEMM)
- **Key metrics**:
  - K=7168 shapes: 0.45-0.53x (best overlap)
  - N=7168, K=2048 shapes: 0.23-0.24x (worst, comm-bound)

---
