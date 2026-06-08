# GEMM-RS Benchmark 报告

## 测试环境

| 项目 | 规格 |
|------|------|
| GPU | 2× NVIDIA B300 SXM6 (SM100) |
| 互联 | NVLink |
| Driver | CUDA 12.x |
| 测试迭代 | 20 次取平均 |
| 日期 | 2026-06-08 |

## 对比方案

| 方案 | 描述 |
|------|------|
| **Separate（基线）** | 标准 GEMM (`bf16_gemm_nt` / `fp8_gemm_nt`) + NCCL `reduce_scatter_tensor` |
| **Fused（融合）** | `bf16_gemm_rs_nt` / `fp8_gemm_rs_nt`（GEMM + NVLink push + PDL reduce） |

## 测试 Shape

Shape 格式为 `total_M × N × K`（`total_M = tokens_per_rank × num_ranks`）。

覆盖从 inference 小 batch 到 training 大矩阵的典型 MoE/Dense 配置。

## 结果

### BF16

| total_M × N × K | Separate (μs) | Fused (μs) | Separate TFLOPS | Fused TFLOPS | Speedup |
|-----------------|:-------------:|:-----------:|:---------------:|:------------:|:-------:|
| 512×512×1024 | 51.4 | **34.6** | 10.4 | 15.5 | **1.49x** |
| 512×1024×2048 | 45.2 | **43.8** | 47.5 | 49.1 | **1.03x** |
| 1024×2048×4096 | **48.0** | 98.9 | 357.7 | 173.7 | 0.49x |
| 2048×2048×4096 | **57.9** | 171.8 | 593.3 | 200.0 | 0.34x |
| 4096×2048×4096 | **86.9** | 308.5 | 790.8 | 222.7 | 0.28x |
| 8192×4096×4096 | **253.5** | 1198.3 | 1084.2 | 229.4 | 0.21x |
| 8192×7168×2048 | **307.9** | 2016.8 | 781.0 | 119.3 | 0.15x |
| 8192×2048×7168 | **194.4** | 634.6 | 1237.2 | 379.0 | 0.31x |

### FP8（BF16 通信）

| total_M × N × K | Separate (μs) | Fused (μs) | Separate TFLOPS | Fused TFLOPS | Speedup |
|-----------------|:-------------:|:-----------:|:---------------:|:------------:|:-------:|
| 512×512×1024 | 65.6 | **42.2** | 8.2 | 12.7 | **1.56x** |
| 512×1024×2048 | 66.1 | **50.2** | 32.5 | 42.8 | **1.32x** |
| 1024×2048×4096 | **63.2** | 101.7 | 271.8 | 168.9 | 0.62x |
| 2048×2048×4096 | **74.3** | 171.7 | 462.4 | 200.2 | 0.43x |
| 4096×2048×4096 | **76.1** | 304.4 | 902.7 | 225.7 | 0.25x |
| 8192×4096×4096 | **196.0** | 1155.8 | 1402.5 | 237.8 | 0.17x |
| 8192×7168×2048 | **250.4** | 1989.0 | 960.7 | 120.9 | 0.13x |
| 8192×2048×7168 | **141.1** | 596.0 | 1704.9 | 403.5 | 0.24x |

### FP8（FP32 通信）

| total_M × N × K | Separate (μs) | Fused (μs) | Separate TFLOPS | Fused TFLOPS | Speedup |
|-----------------|:-------------:|:-----------:|:---------------:|:------------:|:-------:|
| 512×512×1024 | 65.6 | **48.9** | 8.2 | 11.0 | **1.34x** |
| 512×1024×2048 | 66.1 | **63.2** | 32.5 | 34.0 | **1.05x** |
| 1024×2048×4096 | **63.2** | 160.6 | 271.8 | 106.9 | 0.39x |
| 2048×2048×4096 | **74.3** | 292.8 | 462.4 | 117.4 | 0.25x |
| 4096×2048×4096 | **76.1** | 554.2 | 902.7 | 124.0 | 0.14x |
| 8192×4096×4096 | **196.0** | 2215.4 | 1402.5 | 124.1 | 0.09x |
| 8192×7168×2048 | **250.4** | 3816.9 | 960.7 | 63.0 | 0.07x |
| 8192×2048×7168 | **141.1** | 1111.5 | 1704.9 | 216.4 | 0.13x |

### 汇总

| 方案 | Geometric Mean Speedup | Min | Max |
|------|:---------------------:|:---:|:---:|
| BF16 fused vs separate | 0.403x | 0.15x | **1.49x** |
| FP8 fused (BF16 comm) vs separate | 0.403x | 0.13x | **1.56x** |
| FP8 fused (FP32 comm) vs separate | 0.248x | 0.07x | **1.34x** |

## 分析

### 融合 kernel 有优势的场景（M ≤ 512）

在 **小 batch / 低延迟** 场景下（`tokens_per_rank ≤ 256`），融合 kernel 提供 **1.3x–1.6x** 加速：

- **核心原因**：小矩阵时 kernel launch overhead 和 NCCL 初始化延迟占比大。融合方案省去一次独立通信调用，将计算与 NVLink push 完全 overlap。
- **典型用途**：MoE inference serving（每个 expert 只有 64~512 tokens）。

### 融合 kernel 劣势的场景（M ≥ 1024）

在 **大矩阵** 场景下，分离方案反而更快（高达 5–7x）：

| 原因 | 说明 |
|------|------|
| NCCL reduce_scatter 高度优化 | Ring/Tree 算法，只传输 1/N 数据量，bandwidth-optimal |
| 融合 kernel 通信模式非最优 | Symmetric push：每个 rank 向所有其他 rank 推送 partial，总通信量 = (N-1)/N × full_output |
| GEMM 主体 occupancy 受限 | 融合 kernel 的 epilogue warps (W4-W7) 占了 128 线程，大矩阵下 compute-bound 时不如纯 GEMM 高效 |
| NCCL 可以做 compute-comm overlap | 分离方案中 NCCL 可以利用 Copy Engine 与 SM 并行 |

### FP32 通信额外开销

FP32 通信比 BF16 通信额外多约 20–30% 延迟（NVLink 带宽翻倍），在大矩阵下进一步恶化。

## 结论与建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| MoE inference（M_per_rank ≤ 256） | ✅ `gemm_rs_nt` 融合 | 1.3–1.6x 加速，低延迟 |
| MoE inference 小 expert（M ≤ 128） | ✅ `gemm_rs_nt` 融合 | 融合避免 launch overhead |
| 训练 / 大 batch（M_per_rank ≥ 512） | ❌ 用分离方案 | NCCL RS 更高效 |
| 需要 FP32 精度通信 | 根据 M 大小选择 | 小 M 用融合 FP32 comm，大 M 用分离 |

### 优化方向（TODO）

1. **改进通信拓扑**：当前 symmetric push 不是 bandwidth-optimal。考虑 ring-based push 或利用 NVLink multicast 做 single-shot reduce。
2. **大矩阵 occupancy**：减少 epilogue 线程数或使用 persistent kernel 方案提升 SM utilization。
3. **GEMM 内核效率**：融合 kernel 的 GEMM 部分 TFLOPS 远低于独立 GEMM（~200 vs 800+ TFLOPS），需要 profiling 确认瓶颈是 epilogue 延迟还是 TMA 带宽。
4. **Compute-comm overlap 改进**：考虑 CTA 级别的 pipelining，让部分 CTA 做 GEMM、部分做通信。

## 如何运行 Benchmark

```bash
cd /root/.local/codebuddy/DeepGEMM

# 2 GPU, 默认 20 iterations
python benchmarks/bench_gemm_rs.py 2

# 8 GPU, 50 iterations（更稳定）
python benchmarks/bench_gemm_rs.py 8 50
```

输出格式：
```
════════════════════════════════════════════════════════════════════════════════
  GEMM-RS Benchmark: 2 GPUs, 20 iterations per measurement
════════════════════════════════════════════════════════════════════════════════

       Shape (M×N×K) │             Method             │  Time (μs) │   TFLOPS │  Speedup
────────────────────────────────────────────────────────────────────────────────────────
        512×512×1024 │    BF16 separate (gemm+RS)     │       51.4 │     10.4 │ baseline
                     │    BF16 fused (gemm_rs_nt)     │       34.6 │     15.5 │    1.49x
                     │     FP8 separate (gemm+RS)     │       65.6 │      8.2 │ baseline
                     │     FP8 fused (BF16 comm)      │       42.2 │     12.7 │    1.56x
                     │     FP8 fused (FP32 comm)      │       48.9 │     11.0 │    1.34x
──────────────────────────────────────────────────────────────────────────────────────────
  ...
```

> **注意**：不要用 `torchrun`。脚本内部用 `mp.spawn` 管理多进程。
