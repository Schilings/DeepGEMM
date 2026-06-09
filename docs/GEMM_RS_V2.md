# GEMM+RS V2: Pull-based Single-Kernel Fusion with Tile-Level Overlap

## 概述

V2 是对原始 GEMM+RS 实现的重大重构，借鉴了 ByteDance Flux 和 DeepSeek MegaMoe 的设计思想，
在 NVIDIA Blackwell (SM100) 架构上实现了真正的**计算-通信 tile 级流水线重叠**。

## 核心改进

| 维度 | V1 (Push + PDL) | V2 (Pull-based) |
|------|-----------------|-----------------|
| Kernel 数量 | 2 (GEMM + Reduce) | 1 (全融合) |
| 通信方向 | Push (写远端) | Pull (读远端) |
| 通信模型 | Symmetric Push O(N) | All-to-1 Pull O((N-1)/N) |
| Overlap 粒度 | 无真正 overlap | Tile 级流水线 |
| 同步模型 | 全局 nvlink_barrier | Per-tile ready flag |
| 通信带宽效率 | 非 optimal | Bandwidth-optimal (= NCCL) |

## 架构设计

### Warp 分工 (320 threads = 10 warps)

```
┌─────────────────────────────────────────────────────────────────┐
│ Warp 0 (32T): TMA Load — 加载 A/B tiles 到共享内存              │
│ Warp 1 (32T): MMA Issue — 执行 UMMA FMA → TMEM accumulator     │
│ Warp 2-3 (64T): Epilogue — TMEM → smem → local partial buffer  │
│                            + per-tile ready flag signaling       │
├─────────────────────────────────────────────────────────────────┤
│ Warp 4-7 (128T): Comm — Pull-based Reduce-Scatter              │
│                  - Poll per-tile ready flags from ALL ranks     │
│                  - NVLink P2P Read (pull remote partial)        │
│                  - FP32 accumulate → write final output         │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

```
时间 →
                  Tile 0         Tile 1         Tile 2         ...
                  ──────         ──────         ──────
GEMM Warps:      [compute]      [compute]      [compute]
                       │              │              │
Epilogue:        [TMEM→smem→    [TMEM→smem→    [TMEM→smem→
                  local buf]     local buf]     local buf]
                  set flag_0     set flag_1     set flag_2
                       │              │              │
                       ↓              ↓              ↓
Comm Warps:      (waiting)      poll flag_0 →   poll flag_1 →
                               pull + reduce   pull + reduce
                               → output[0]     → output[1]
```

### M 维 Swizzle 调度

Rank i 计算 tile 的顺序：
1. 先计算属于 rank (i+1) 的 chunk
2. 再计算属于 rank (i+2) 的 chunk
3. ...
4. 最后计算属于自己 rank i 的 chunk

这确保了接收方的 Comm Warps 能尽早开始拉取数据。

### 同步机制

1. **GEMM → Epilogue**: tmem_full/tmem_empty barriers (经典流水线)
2. **Epilogue → Comm (跨 rank)**: Per-tile ready flag + `__threadfence_system`
3. **Comm poll**: `ld_acq_sys` 自旋等待远端 ready flag
4. **Kernel 结束**: nvlink_barrier 确保所有 rank 完成 pull，然后 reset flags

## 通信量分析

对于 N 个 rank，每个 rank 的输出大小为 `M_per_rank × N_dim`：

- **V1 (Symmetric Push)**: 每个 rank 向 N-1 个 peer 各推送一份完整 chunk
  - 总通信量 = `(N-1) × M_per_rank × N_dim` per rank
  - 全系统 = `N × (N-1) × chunk_size`

- **V2 (All-to-1 Pull)**: 每个 rank 从 N-1 个 peer 各拉取一份 chunk
  - 总通信量 = `(N-1) × M_per_rank × N_dim` per rank
  - 但关键区别：**每个 rank 只读取自己需要的数据**（无冗余传输）
  - 等同于 NCCL ring reduce-scatter 的 bandwidth-optimal 通信量

## 文件结构

```
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs_v2.cuh  — 核心 kernel 实现
csrc/jit_kernels/impls/sm100_bf16_gemm_rs_v2.hpp            — JIT runtime
csrc/jit_kernels/heuristics/gemm_rs.hpp                     — get_gemm_rs_v2_config()
csrc/apis/gemm_rs.hpp                                       — C++ API: bf16_gemm_rs_v2_nt()
deep_gemm/gemm_rs/__init__.py                               — Python API: bf16_gemm_rs_v2_nt()
tests/test_gemm_rs_v2.py                                    — 正确性测试
benchmarks/bench_gemm_rs_v2.py                              — 性能基准测试
```

## 运行测试

```bash
# 正确性测试 (2 GPUs)
python tests/test_gemm_rs_v2.py 2

# 正确性测试 (8 GPUs)
python tests/test_gemm_rs_v2.py 8

# 性能基准测试
python benchmarks/bench_gemm_rs_v2.py 2 20
python benchmarks/bench_gemm_rs_v2.py 8 20
```

## 预期改进

相比 V1：
- **小 batch (2 GPU)**: 接近或超越 V1（省去 reduce kernel launch 开销）
- **多 GPU (4-8)**: 显著优于 V1（bandwidth-optimal + tile overlap）
- **大矩阵**: 计算密集时 overlap 效果最好，预期接近纯 GEMM 时延

相比 Separate (GEMM + NCCL RS)：
- **所有 shape**: 预期接近或更好（无 kernel launch 间隙 + 计算通信重叠）
