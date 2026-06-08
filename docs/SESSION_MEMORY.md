# DeepGEMM GEMM-RS 开发会话记忆

> **最后更新**: 2026-06-08
> **当前分支**: `main`
> **环境**: 8× B300 SXM6 NVLink 互联

---

## 📌 项目概述

DeepGEMM 的 **GEMM-RS (GEMM + Reduce-Scatter)** 融合 kernel，目标是在多 GPU NVLink 互联环境下，将 GEMM 计算与 ReduceScatter 通信重叠，实现 MoE 推理中的通信掩盖。

支持两种数据类型：
- **BF16** — 主力 kernel，已稳定可用 ✅
- **FP8** — 有 pre-existing bug，待修复 ❌

---

## 🏗️ 架构设计：方案 B（当前实现）

### 核心思路

**两阶段 PDL 分离架构 + 统一 barrier 同步**

```
┌─────────────────────────────────────────────────────────┐
│ 阶段1: GEMM + NVLink Push Kernel                        │
│                                                         │
│  Ring 调度: rank i → 先算 rank(i+1) chunk → push远端     │
│            → 再算 rank(i+2) → push ...                  │
│            → 最后算自己的 chunk → 写本地 partial buf       │
│                                                         │
│  Epilogue: TMEM → smem → TMA bulk store / global store  │
│                                                         │
│  ★ 所有 tile 完成后（仅一次）:                            │
│    tma_store_wait(0) → __threadfence_system()           │
│    → nvlink_barrier（跨 rank 同步）                      │
└─────────────────────────────────────────────────────────┘
            │ PDL (Programmatic Dependent Launch)
            ↓
┌─────────────────────────────────────────────────────────┐
│ 阶段2: Reduce Epilogue Kernel                           │
│                                                         │
│  cudaGridDependencySynchronize() 等待阶段1              │
│  直接读 partial buffer（无需轮询 ready flag）            │
│  element-wise 累加 → 写 output                          │
└─────────────────────────────────────────────────────────┘
```

### 方案 B vs 旧方案（per-tile fence + ready flag）

| 维度 | 旧方案 | 方案 B |
|------|--------|--------|
| 同步粒度 | 每个 tile 写完都做 `__threadfence_system` + `st_rel_sys(ready_flag)` | 整个 kernel 只做一次 `__threadfence_system` + `nvlink_barrier` |
| Reduce kernel 等待 | 自旋轮询 `ld_acq_sys(ready_flag)` | PDL 依赖 → 进入即读 |
| 线程开销 | 需要设置/清零 ready flag | 无 flag 操作 |
| 初始化 | 需要清零所有 ready flag + barrier | 仅需一个初始 barrier |

### 关键设计决策

1. **Dynamic block_m**: 根据 wave count 动态选择 32/64/128（参考 mega_moe 的 heuristic）
2. **num_stages 不加 cap**: 去掉 `min(..., 8)` 限制，让 smem 容量自行决定
3. **BF16 epilogue 用 TMA bulk store**: 通过 smem → TMA CE 写远端
4. **FP8 epilogue 用 global store**: 逐行写（FP8 → BF16 转换后写）

---

## 📊 当前性能状态

### BF16 Benchmark (8 GPU, 20 iters)

| Shape (M×N×K) | Separate (μs) | Fused (μs) | Speedup |
|:---|:---|:---|:---|
| 2048×512×1024 | 51.0 | 36.3 | **1.40x** ✨ |
| 2048×1024×2048 | 48.4 | 50.7 | 0.96x |
| 4096×2048×4096 | 106.0 | 187.0 | 0.57x |
| 8192×2048×4096 | 195.9 | 214.7 | 0.91x |
| 16384×2048×4096 | 295.2 | 307.2 | 0.96x |
| 32768×4096×4096 | 1016.6 | 1101.3 | 0.92x |
| 32768×7168×2048 | 1225.1 | 1737.7 | 0.71x |
| 32768×2048×7168 | 772.3 | 807.7 | 0.96x |

**分析**:
- 小 shape（communication-bound，MoE 常见）: fused 优势明显 **1.40x**
- 大 shape（compute-bound）: fused 略慢 4~8%，主要开销来自 epilogue global store + 全局 fence
- 4096×2048×4096 和 7168 场景下 fused 较慢，可能是 dynamic block_m heuristic 不够优化

---

## ✅ 正确性验证结果

| 测试 | 2 GPU | 8 GPU |
|------|-------|-------|
| BF16 GEMM-RS | ✅ PASS (max_diff=0.0) | ✅ PASS (max_diff=0.0) |
| FP8 GEMM-RS (BF16 comm) | ✅ PASS (max_diff=0.0) | ✅ PASS (max_diff=0.0) |
| FP8 GEMM-RS (FP32 comm) | ✅ PASS (max_diff=0.0) | ✅ PASS (max_diff=0.0) |

---

## 🐛 已知问题

### ~~FP8 Kernel `cudaErrorIllegalAddress`~~ ✅ 已修复！

- **根因**: `get_pipeline_config_for_gemm_rs()` 在计算 shared memory 大小时，
  漏了 FP8 特有的 SFA/SFB 缓冲区和 `with_sf_full` barriers 开销
- **修复**: 为 `get_pipeline_config_for_gemm_rs()` 添加 `is_fp8` 参数，
  FP8 路径正确计入 `smem_sfa_per_stage + smem_sfb_per_stage + 3 barriers/stage`
- **验证**: 2 GPU 和 8 GPU FP8 测试全部 PASS (max_diff=0.000000)

### FP8 Fused 性能问题（待优化）

- **现象**: FP8 fused 比 separate 慢很多（0.07x ~ 0.72x）
- **根因**: FP8 epilogue 使用逐元素 global store（不是 TMA bulk store）
  - 每个线程逐个写 4 bytes 到远端 NVLink 内存
  - 对于大 shape 严重串行化
- **优化方向**: 
  - 改为 TMA store（需要 FP32→BF16/FP32 转换后写入 smem，再 TMA bulk store）
  - 或者仿照 BF16 kernel 的 TMEM→smem→TMA store 流水线

---

## 📁 关键文件路径

```
deep_gemm/include/deep_gemm/impls/
├── sm100_bf16_gemm_rs.cuh       # BF16 GEMM kernel (方案B已应用) ✅
├── sm100_fp8_gemm_rs.cuh        # FP8 GEMM kernel (方案B已应用，但有pre-existing bug)
├── sm100_reduce_epilogue.cuh    # Reduce Epilogue kernel (BF16/FP8共用)

csrc/jit_kernels/impls/
├── sm100_bf16_gemm_rs.hpp       # BF16 JIT 配置 (含 dynamic block_m)
├── sm100_fp8_gemm_rs.hpp        # FP8 JIT 配置

deep_gemm/include/deep_gemm/comm/
├── barrier.cuh                  # nvlink_barrier / grid_sync 实现

deep_gemm/include/deep_gemm/layout/
├── gemm_rs.cuh                  # GemmRSWorkspace 布局定义
├── sym_buffer.cuh               # SymmetricBuffer（NVLink 映射）

tests/
├── test_gemm_rs_bf16.py         # BF16 正确性测试
├── test_gemm_rs_fp8.py          # FP8 正确性测试
├── test_gemm_rs_comm_modes.py   # 通信模式测试

benchmarks/
├── bench_gemm_rs.py             # 性能对比 benchmark (支持 SKIP_FP8=1)
```

---

## 🔄 Git 提交历史（方案B相关）

```
33520d5 feat(fp8_gemm_rs): apply Plan B to FP8 kernel + add SKIP_FP8 bench option
8054773 feat(gemm_rs): Plan B - remove per-tile fence, add kernel-end barrier + dynamic config
80bdb23 bench: add 2/4/8 GPU results to GEMM-RS benchmark report
91825a6 bench: add GEMM-RS benchmark script and performance report
```

---

## 🚀 后续优化方向（按优先级）

### P0: FP8 Epilogue 性能优化
- **TMA Store**: 将 FP8 epilogue 从逐元素 global store 改为 TMEM→smem→TMA bulk store
  - 参考 BF16 kernel 的实现（smem staging + TMA 1D store per row）
  - 需要在 smem 中做 FP32→comm_dtype 转换后再 TMA 写出
- **当前状态**: FP8 fused 只有小 shape 接近 separate 性能（0.72x），大 shape 慢 5~10x

### P1: 大 Shape 性能优化
- **Heuristics 调优**: 4096×2048×4096 等 shape 的 block_m 选择优化
- **Persistent kernel**: 考虑 persistent thread block 减少 launch overhead
- **BF16 大 shape**: 目前 0.57x~0.71x 的 shape 需要更好的 tile 分配策略

### P2: 功能完善
- 支持更多 MoE shape 组合
- 与 mega_moe 的深度整合
- 多节点支持（跨 NVSwitch domain）

---

## 💡 运行命令速查

```bash
# 安装
pip install -e . --no-build-isolation

# BF16 正确性测试
python tests/test_gemm_rs_bf16.py 2    # 2 GPU
python tests/test_gemm_rs_bf16.py 8    # 8 GPU

# FP8 正确性测试（当前有 bug）
python tests/test_gemm_rs_fp8.py 2

# Benchmark
python benchmarks/bench_gemm_rs.py 8 20           # 完整 benchmark
SKIP_FP8=1 python benchmarks/bench_gemm_rs.py 8 20  # 跳过 FP8

# 清除 JIT 缓存（修改 .cuh 后需要）
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs_nt.*
rm -rf ~/.deep_gemm/cache/kernel.sm100_fp8_gemm_rs_nt.*
rm -rf ~/.deep_gemm/cache/kernel.sm100_fp8_reduce_epilogue.*
```

---

## ⚙️ 环境信息

- **GPU**: 8× NVIDIA B300 SXM6 (NVLink 互联)
- **CUDA**: SM100 (Blackwell)
- **Python 包**: `deep_gemm` (editable install)
- **JIT 缓存**: `~/.deep_gemm/cache/`
