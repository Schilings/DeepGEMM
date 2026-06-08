# BF16 GEMM + Reduce-Scatter (GEMM-RS) Kernel

## 概述

`bf16_gemm_rs_nt` 是一个融合了矩阵乘法（GEMM）和 Reduce-Scatter 通信的 SM100（B300）专用内核。它在计算 `D = A @ B^T` 的同时，将结果在多 GPU 间进行 reduce-scatter 操作，避免了单独通信的开销。

## 硬件要求

- **至少 2 张 NVIDIA B300 (SM100) GPU**
- 支持 NVLink 互联（用于 reduce-scatter 通信）

## 环境准备

```bash
# 1. 确保已安装 PyTorch（需支持 CUDA）
python -c "import torch; print(torch.__version__)"

# 2. 安装 DeepGEMM（开发模式）
cd /root/.local/codebuddy/DeepGEMM
pip install -e . --no-build-isolation
```

> **注意**：必须加 `--no-build-isolation`，因为 `setup.py` 依赖当前环境中的 `torch`。

## 运行测试

```bash
cd /root/.local/codebuddy/DeepGEMM
python tests/test_gemm_rs_bf16.py
```

测试会自动用 `torch.multiprocessing.spawn` 启动 2 个进程（每个进程占 1 张 GPU）。

### 预期输出

```
============================================================
BF16 GEMM-RS Test: 2 GPUs
  M_per_rank=256, K=1024, N=512
============================================================

>>> Phase 1: Warm-up (JIT compilation)...
>>> Phase 2: Second run for consistency check...
  Consistency check: max_diff=0.000000
  ✅ Kernel produces consistent results across runs
>>> Phase 3: Comparing with reference (bf16_gemm + reduce_scatter)...

============================================================
Results:
  Max abs diff:  0.000000
  Mean abs diff: 0.000000
  ✅ PASS — BF16 GEMM-RS matches reference!
    [Rank 0] max_diff=0.000000, mean_diff=0.000000
    [Rank 1] max_diff=0.000000, mean_diff=0.000000
============================================================
Test complete.
============================================================
```

## 测试逻辑

| 阶段 | 说明 |
|------|------|
| Phase 1 | Warm-up，触发 JIT 编译内核 |
| Phase 2 | 连续两次调用，验证结果一致性 |
| Phase 3 | 与参考实现（`bf16_gemm_nt` + `reduce_scatter_tensor`）对比 |

参考实现：先用标准 `bf16_gemm_nt` 做完整 GEMM，再用 `torch.distributed.reduce_scatter_tensor` 做 reduce-scatter。两者结果应 bit-exact 一致。

## 核心文件

| 文件 | 说明 |
|------|------|
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh` | CUDA 内核实现 |
| `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp` | JIT 编译入口，传递模板参数 |
| `csrc/jit_kernels/heuristics/gemm_rs.hpp` | 启发式配置（tile size、multicast 等） |
| `tests/test_gemm_rs_bf16.py` | 测试脚本 |

## 内核模板参数

```cpp
template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t SMEM_EPILOGUE_N,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          typename cd_dtype_t>
```

- `kSwizzleAMode/B/CD`：Swizzle 模式，控制 shared memory 布局
- `kNumMulticast`：TMA multicast 数量（1 或 2）
- `kIsMulticastOnA`：multicast 作用于 A 还是 B
- `kSwapAB`：是否交换 A/B 输入
- `kWithAccumulation`：是否累加到已有输出

## 常见问题

### Q: `ModuleNotFoundError: No module named 'deep_gemm'`
安装包：`pip install -e . --no-build-isolation`

### Q: `RuntimeError: CUDA error: no kernel image is available for execution on the device`
内核仅支持 SM100（B300），不支持 A100/H100 等旧架构。

### Q: 测试卡住不动
检查是否有足够的 GPU（`nvidia-smi`），以及 NVLink 连接是否正常。
