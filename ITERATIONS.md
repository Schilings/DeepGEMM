# AKO4ALL Iteration Log — GEMM+RS Kernel Optimization

## Summary

| Iter | Direction | Geo Mean Speedup | Best Shape | Status |
|------|-----------|-----------------|------------|--------|
| baseline | multicast=2, current code | 0.357x | 0.525x (16384×7168×7168) | ✅ |
| 1 | Distributed ready flag polling (4 warps) | 0.362x | 0.543x | ✅ +1.4% |

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
