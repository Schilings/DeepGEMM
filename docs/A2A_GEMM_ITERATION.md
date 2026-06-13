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

Config: mc2, block 128×128×64, num_stages=7, 256T, launch_bounds(256,1), kNumReadyChunksPerSlot=4

---

## Iter 1: `__launch_bounds__(256, 2)` (commit 4c500a9)

**Direction**: Increase min_blocks_per_SM from 1→2 to improve SM occupancy.

**Result**: Geo Mean 1.239x → **1.264x** (+2.0%) ✅ KEEP

| Shape | Baseline | Iter 1 | Delta |
|---|---|---|---|
| 1024×4096×4096 | 1.335x | 1.380x | +3.4% |
| 1024×7168×4096 | 1.222x | 1.179x | -3.5% |
| 2048×4096×7168 | 1.383x | 1.401x | +1.3% |
| 2048×7168×4096 | 1.139x | 1.141x | +0.2% |
| 4096×4096×4096 | 1.243x | 1.449x | **+16.6%** |
| 4096×7168×4096 | 1.345x | 1.204x | -10.5% |
| 4096×4096×7168 | 1.159x | 1.578x | **+36.2%** |
| 8192×4096×4096 | 1.582x | 1.560x | -1.4% |
| 8192×7168×4096 | 1.346x | 1.347x | +0.1% |
| 8192×7168×7168 | 1.182x | 1.179x | -0.3% |
| 2048×7168×2048 | 1.091x | 1.073x | -1.6% |
| 4096×7168×2048 | 1.202x | 1.211x | +0.7% |
| 16384×7168×4096 | 1.201x | 1.120x | -6.7% |
| 16384×7168×7168 | 1.012x | 1.026x | +1.4% |

**Geo Mean**: 1.239x → 1.264x | **Avg Fused**: 1105.7T → 1129.3T

**Analysis**: Mixed results—big wins on N=4096 shapes, some losses on N=7168 and large shapes. Overall geo_mean improvement is modest but positive. The launch_bounds(256,2) may be restricting register usage, benefiting compute-bound (small N) shapes but hurting communication-bound (large N/K) shapes where more registers help with barrier polling.

---

