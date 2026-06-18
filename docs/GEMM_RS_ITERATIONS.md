# AKO4ALL Iteration Log — GEMM+RS Kernel Optimization

> ⚠️ 历史迭代日志：包含不同机器与阶段的实验结果。
> 当前可复现结论请优先查看 `docs/PROGRESS.md`。

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
| 10 | Reduce CD stages 2→1 (reverted) | 0.540x | 2.324x | ❌ -17.4% (need double-buffer for TMEM→smem) |
| 11 | Remove __threadfence_system (use release-acquire) | 0.62x | ~3.7x | ✅ neutral/slight improvement |
| 12 | **v2 PUSH-BASED** (Flux-inspired architecture change) | **0.686x** | 0.93x | ✅✅✅ +5% (stable, no warmup noise) |
| 13 | Remove __threadfence_system in push epilogue | **0.732x** | 0.88x | ✅ +6.7%, worst 0.54x |
| 14 | Direct TMEM→remote (bypass smem, reverted) | 0.469x | 0.656x | ❌ -36% (smem staging needed for parallelism) |
| 15 | **Round-robin interleaved tile scheduling** | **0.970x** | 1.12x | ✅✅✅ +33%! 13/21 shapes >1x! |
| 16 | Self-rank direct output write (bypass partial) | **0.982x** | 1.16x | ✅ +1.2%, K=4096 best 1.16x |
| 17 | TMA async store, single-thread issue (reverted) | N/A | 0.741x | ❌ 128 per-row tma_store_1d by 1 thread too slow |
| 18 | **TMA async store, multi-thread parallel issue** | **1.004x** | 1.11x | ✅✅✅ BREAKS 1.0x! K=2048: 0.78→0.84-0.95x |

## Profiling 发现 (iter 18 后)

**Fused GEMM 比标准 GEMM 慢 46.6%**（4096×4096×4096: 1169T vs 1714T）

| 因素 | 标准 GEMM | Fused GEMM-RS | 影响 |
|------|-----------|---------------|------|
| 总线程 | 256T (8 warps) | 384T (12 warps) | 128T Comm 占 4 warp slots |
| 寄存器/线程 | 64512/256=252 | 64512/384=168 | **33% less regs → register spilling** |
| Epilogue store | TMA 2D (1次) | Per-row TMA 1D (128次) | Store 效率较低 |
| 额外同步 | 无 | nvlink_barrier ×2 | Grid-level sync overhead |
| Warp scheduler | 8 warps (全给 GEMM) | 12 warps (4 idle Comm) | Scheduler bandwidth 被稀释 |

**最大嫌疑**：`__launch_bounds__(384, 1)` 导致编译器给每线程只分配 168 寄存器（vs 标准 252），
register spilling 严重拖慢 GEMM pipeline。这是 fusion kernel 性能不及预期的根本原因。

**优化方向**：减少 register spilling，或减少总线程数（但之前 288T 实验 0.23x 说明需要保持编译器行为一致）。

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
