# A2A GEMM (All2All + GEMM) Iteration Log

## Baseline (commit 8814016)

Flux-style A2A GEMM: host-side CE DMA + compute-only kernel, per-chunk barrier polling.

**8 GPU, 10 iters:**

| Shape (M/rank × N × K) | Fused TFLOPS | Sep TFLOPS | Speedup |
|---|---|---|---|
| 1024×4096×4096 | 584.7T | 438.0T | 1.335x |
| 1024×7168×4096 | 1011.5T | 827.4T | 1.222x |
| 2048×4096×7168 | 1084.0T | 783.6T | 1.383x |
| 2048×7168×4096 | 1221.3T | 1072.0T | 1.139x |
| 4096×4096×4096 | 1092.6T | 878.9T | 1.243x |
| 4096×7168×4096 | 1275.8T | 948.3T | 1.345x |
| 4096×4096×7168 | 965.8T | 833.4T | 1.159x |
| 8192×4096×4096 | 1327.3T | 839.1T | 1.582x |
| 8192×7168×4096 | 1281.0T | 951.9T | 1.346x |
| 8192×7168×7168 | 1131.5T | 957.3T | 1.182x |
| 2048×7168×2048 | 1000.2T | 916.6T | 1.091x |
| 4096×7168×2048 | 1267.5T | 1054.3T | 1.202x |
| 16384×7168×4096 | 1203.2T | 1001.5T | 1.201x |
| 16384×7168×7168 | 1033.9T | 1021.3T | 1.012x |

**Geo Mean: 1.239x | Avg Fused: 1105.7T | Avg Sep: 894.6T**

Config: mc2, block 128×128×64, num_stages=7, 256T (128 non-epi + 128 epi), kNumReadyChunksPerSlot=4

### Analysis

- **Strong shapes**: N=4096, M≥8192 → 1.3-1.6x (A2A comm overhead is significant, overlap pays off)
- **Weak shapes**: K=7168 → 1.01-1.18x (GEMM compute dominates, A2A is ~5% overhead, hard to overlap)
- **Weakest**: 16384×7168×7168 = 1.012x (almost no A2A overhead to save)

### Optimization directions to try

1. **PDL (Pipeline DSL) tuning** — AG GEMM iter 1 got +1.5% from PDL
2. **Barrier polling backoff** — `__nanosleep(200)` when polling (reduce SM power waste)
3. **Chunk copy pipelining** — overlap multiple chunk copies on comm_stream
4. **Launch bounds tuning** — `__launch_bounds__(256, 2)` for more occupancy
5. **kNumReadyChunksPerSlot tuning** — 4→2 or 4→8
6. **Communication order tuning** — try different ring order for remote pulls

---

## Iterations

