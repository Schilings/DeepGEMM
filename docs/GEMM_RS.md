# GEMM + Reduce-Scatter (GEMM-RS) Kernel

## 概述

DeepGEMM 提供融合了矩阵乘法（GEMM）和 Reduce-Scatter 通信的 SM100（B300）专用内核，在计算 `D = A @ B^T` 的同时将结果在多 GPU 间进行 reduce-scatter 操作，避免了单独通信的开销。

支持两种数据类型：

| API | 输入精度 | 说明 |
|-----|---------|------|
| `bf16_gemm_rs_nt` | BF16 × BF16 | 标准 BF16 矩阵乘 + Reduce-Scatter |
| `fp8_gemm_rs_nt` | FP8 × FP8 | FP8 矩阵乘（含 Scale Factor）+ Reduce-Scatter |

两者共享相同的**两阶段 PDL 架构**：

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段1: GEMM + NVLink Push                                    │
│                                                              │
│  调度顺序: rank i 先计算发往 rank i+1 的 chunk → push       │
│           → rank i+2 → push → ... → 最后写自己的 chunk      │
│  N 次计算掩盖 N-1 次通信                                     │
│  Epilogue: TMEM → registers → global store 到远端 partial   │
└──────────────────────────────────────────────────────────────┘
             │ PDL (Programmatic Dependent Launch)
             ↓
┌──────────────────────────────────────────────────────────────┐
│ 阶段2: Reduce Epilogue                                       │
│                                                              │
│  cudaGridDependencySynchronize() 等待阶段1完成               │
│  从 partial buffer 读取各 rank 的数据                        │
│  element-wise 累加 → 写 output                               │
└──────────────────────────────────────────────────────────────┘
```

## 硬件要求

- **至少 2 张 NVIDIA B300 (SM100) GPU**
- NVLink 互联（用于 reduce-scatter 通信）

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

# BF16 GEMM-RS 测试（默认 2 GPU）
python tests/test_gemm_rs_bf16.py 2

# FP8 GEMM-RS 测试（默认 2 GPU）
python tests/test_gemm_rs_fp8.py 2

# 8 GPU 测试
python tests/test_gemm_rs_bf16.py 8
python tests/test_gemm_rs_fp8.py 8
```

测试用 `torch.multiprocessing.spawn` 启动多进程（每进程占 1 张 GPU）。**不要用 `torchrun`**。

### 预期输出（BF16）

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
============================================================
```

### 预期输出（FP8）

```
============================================================
FP8 GEMM-RS Test: 2 GPUs
  M_per_rank=256, K=1024, N=512, gran_k=128
============================================================

>>> Phase 1: Warm-up (JIT compilation)...
>>> Phase 2: Second run for consistency check...
  Consistency check: max_diff=0.000000
  ✅ Kernel produces consistent results across runs
>>> Phase 3: Comparing with reference (fp8_gemm_nt + reduce_scatter)...

============================================================
Results:
  Max abs diff:  0.000000
  Mean abs diff: 0.000000
  ✅ PASS — FP8 GEMM-RS matches reference!

>>> Phase 4: Testing with FP32 communication dtype...
  FP32 comm: max_diff=0.000000, mean_diff=0.000000
  ✅ FP32 communication matches FP32-path reference!
============================================================
```

## 测试逻辑

### BF16 测试

| 阶段 | 说明 |
|------|------|
| Phase 1 | Warm-up，触发 JIT 编译内核 |
| Phase 2 | 连续两次调用，验证结果一致性 |
| Phase 3 | 与参考实现（`bf16_gemm_nt` + `reduce_scatter_tensor`）对比 |

### FP8 测试

| 阶段 | 说明 |
|------|------|
| Phase 1 | Warm-up，触发 JIT 编译内核 |
| Phase 2 | 连续两次调用，验证结果一致性 |
| Phase 3 | 与参考实现（`fp8_gemm_nt` + FP32 manual reduce-scatter）对比（BF16 通信） |
| Phase 4 | 测试 FP32 通信模式，与 FP32 路径参考对比 |

参考实现：先用标准 GEMM 做完整矩阵乘，再用 all_gather + FP32 手动 reduce-scatter。两者结果应 bit-exact 一致。

## API 参考

### `bf16_gemm_rs_nt`

```python
deep_gemm.bf16_gemm_rs_nt(
    y,                    # [tokens_per_rank, N], output (BF16)
    a,                    # [total_tokens, K], input (BF16)
    b,                    # [N, K], weight NT layout (BF16)
    sym_buffer,           # GemmRSSymmBuffer
    num_tokens_per_rank,  # 当前实际 token 数
    compiled_dims='nk',   # JIT 编译维度
    reduce_in_fp32=True,  # reduce 阶段是否用 FP32 累加
)
```

### `fp8_gemm_rs_nt`

```python
deep_gemm.fp8_gemm_rs_nt(
    y,                    # [tokens_per_rank, N], output (BF16)
    (a_fp8, a_sf),        # (FP8 tensor, scale factor)
    (b_fp8, b_sf),        # (FP8 tensor, scale factor)
    sym_buffer,           # GemmRSSymmBuffer
    num_tokens_per_rank,  # 当前实际 token 数
    recipe=(1, 1, 128),   # (gran_m, gran_n, gran_k)
    compiled_dims='nk',   # JIT 编译维度
    comm_dtype='bf16',    # 通信精度: 'bf16' 或 'fp32'
    reduce_in_fp32=True,  # reduce 阶段是否用 FP32 累加
)
```

### `get_symm_buffer_for_gemm_rs`

```python
sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
    group,                    # ProcessGroup
    num_max_tokens_per_rank,  # 最大 token 数（会自动对齐）
    hidden,                   # N 维度
    out_dtype=torch.bfloat16, # 输出 dtype
    comm_dtype=None,          # 通信 dtype (None=BF16, torch.float32=FP32)
)
```

### 通信精度选择

| `comm_dtype` | NVLink 带宽 | 精度 | 适用场景 |
|-------------|------------|------|---------|
| `'bf16'`（默认） | 1× | 略有截断 | 训练（推荐） |
| `'fp32'` | 2× | 完整 FP32 | 需要高精度的场景 |

> **注意**：`comm_dtype` 控制的是 NVLink 传输的数据精度。GEMM 内部累加器始终为 FP32，差异仅在于 TMEM→global store 时是否做 FP32→BF16 截断。

## 内核架构

### Warp 分工（256 线程 = 8 warps）

| Warp | BF16 职责 | FP8 职责 |
|------|----------|----------|
| W0 | TMA 加载 A/B | TMA 加载 A/B + SFA/SFB |
| W1 | MMA (UMMA) | SF→TMEM + MMA (MXF8F6F4) |
| W2 | Idle | SF warp transpose (UTCCP) |
| W3 | Idle | Idle |
| W4-W7 | Epilogue（TMEM→reg→remote store） | Epilogue（TMEM→reg→remote store） |

### 设计优势（相比旧版内嵌 RS warps 方案）

1. **去掉 128 个 RS 线程**（384→256），所有线程用于计算和通信
2. **Reduce kernel 不需自旋等待**，PDL 保证数据已就绪
3. **简化的 epilogue**：直接 TMEM → registers → global store
4. **支持 `comm_dtype_t` 选择通信精度**（BF16 省带宽 / FP32 保精度）

### FP8 特有逻辑

- TMA 加载 Scale Factor A/B (SFA/SFB) 到 shared memory
- Warp 2 执行 SF warp transpose（UTCCP 所需布局变换）
- Warp 1 通过 UTCCP 拷贝 SF 到 TMEM，再调用 `SM100_MMA_MXF8F6F4_SS::fma`
- SM100 要求 UE8M0 格式的 scale factor（`use_ue8m0=True`）
- 支持 `gran_k=32` 或 `gran_k=128`

## 核心文件

| 文件 | 说明 |
|------|------|
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh` | BF16 GEMM-RS + Reduce Epilogue CUDA 内核 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_rs.cuh` | FP8 GEMM-RS CUDA 内核（Push Only） |
| `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp` | BF16 JIT 编译入口 |
| `csrc/jit_kernels/impls/sm100_fp8_gemm_rs.hpp` | FP8 JIT 编译入口（两阶段启动） |
| `csrc/jit_kernels/heuristics/gemm_rs.hpp` | 启发式配置（tile size、multicast 等） |
| `deep_gemm/gemm_rs/__init__.py` | Python API 层 |
| `tests/test_gemm_rs_bf16.py` | BF16 测试脚本 |
| `tests/test_gemm_rs_fp8.py` | FP8 测试脚本 |

## 内核模板参数

### BF16 GEMM-RS Kernel

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
```

### FP8 GEMM-RS Kernel

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kGranK,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
```

FP8 版本额外多一个 `kGranK` 参数，控制 Scale Factor 的 K 维度粒度。

### Reduce Epilogue Kernel（BF16/FP8 共用）

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads,
          typename cd_dtype_t,
          typename comm_dtype_t,
          bool kReduceInFP32 = true>
```

- `kReduceInFP32`：是否在 FP32 精度下执行 reduce 累加

## 运行 Benchmark

对比融合 GEMM-RS 与分离方案（GEMM + NCCL reduce_scatter）的端到端吞吐量：

```bash
cd /root/.local/codebuddy/DeepGEMM

# 2 GPU, 默认 20 iterations
python benchmarks/bench_gemm_rs.py 2

# 8 GPU, 50 iterations（更稳定的测量）
python benchmarks/bench_gemm_rs.py 8 50
```

Benchmark 覆盖多种 shape（M=256~4096, N/K=512~7168），输出每种配置的延迟（μs）、TFLOPS 和相对加速比。

详细结果与分析见 [BENCHMARK_GEMM_RS.md](./BENCHMARK_GEMM_RS.md)。

> **结论**：融合 kernel 在小 batch（M_per_rank ≤ 256）下提供 1.3–1.6x 加速（适合 MoE inference），大矩阵下应使用分离方案。

## 常见问题

### Q: `ModuleNotFoundError: No module named 'deep_gemm'`
安装包：`pip install -e . --no-build-isolation`

### Q: `RuntimeError: CUDA error: no kernel image is available for execution on the device`
内核仅支持 SM100（B300），不支持 A100/H100 等旧架构。

### Q: 测试卡住不动
检查是否有足够的 GPU（`nvidia-smi`），以及 NVLink 连接是否正常。

### Q: FP8 测试报 `sf.size(-2) == ceil_div(mn, gran_mn)` 断言失败
确保传递了正确的 `recipe` 参数。`per_token_cast_to_fp8` 生成的 SF 是 per-token 的（`gran_mn=1`），所以 `recipe` 必须设为 `(1, 1, gran_k)`，而非默认的 `(1, 128, 128)`。

### Q: 如何选择 `comm_dtype`？
- 大多数训练场景用默认 BF16 即可（省一半 NVLink 带宽）
- 如果 reduce 后精度损失不可接受（如长序列、大 hidden），切换为 FP32

### Q: 不要用 `torchrun`
测试脚本内部用 `mp.spawn` 管理多进程，用 `torchrun` 会导致进程数翻倍。直接 `python tests/test_gemm_rs_*.py <num_gpus>` 即可。
