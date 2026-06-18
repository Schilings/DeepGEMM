# GEMM-RS 当前设计（2026-06-18 版本）

> 本文档描述**当前代码形态与近期重构方向**，不是历史方案合集。

---

## 1. 当前实现形态

### 1.1 主路径

- 算子：`bf16_gemm_rs_nt`（以及 `bf16_gemm_rs_nt_v3`）
- 形态：SM100 上的 GEMM + RS 融合路径，依赖 symmetric buffer + per-tile ready flag。
- 当前可用性：
  - quick correctness（2 GPU）可跑通
  - benchmark 路径仍在稳定化

### 1.2 关键模块

- Kernel/impl：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
- JIT runtime：`csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`
- 启发式：`csrc/jit_kernels/heuristics/gemm_rs.hpp`
- Python 入口：`deep_gemm/gemm_rs/__init__.py`
- 测试：`tests/test_gemm_rs.py` / `tests/test_gemm_rs_quick.py`
- 基准：`benchmarks/bench_gemm_rs.py`

---

## 2. 本轮已落地的工程修复

### 2.1 JIT/NVRTC 可用性修复

- 在无 `nvcc` 机器上明确走 `DG_JIT_USE_NVRTC=1`。
- 补齐 NVRTC 编译需要的 include 路径（含 `cutlass`、`cccl`）。
- NVRTC 失败时 fail-fast，输出明确错误信息。
- 修复 PTXAS 参数拼接格式问题（避免非法参数合并）。

### 2.2 运行时与多进程行为修复

- `dist.init` 后先 `torch.cuda.set_device(local_rank)`，再设置默认设备到 `cuda:{local_rank}`，降低多进程默认设备错绑风险。

### 2.3 benchmark 诊断能力补充

- 支持通过环境变量限制 shape 范围与单 shape 运行。
- 支持逐迭代同步模式用于异步报错定位。

---

## 3. 现阶段设计判断（不被历史方案绑死）

### 3.1 保留当前主线用于“跑通 + 建基线”

在 benchmark 稳定前，不进行大规模结构重写，优先：
1. 固化可复现测试链路
2. 定位 benchmark lifecycle 崩溃点
3. 拿到稳定基线

### 3.2 重构方向（Flux-style）

一旦基线稳定，优先考虑：
- **Compute/Comm 进一步解耦**（偏向双核或更清晰流水边界）
- 把 RS 从 kernel 细节提升为可替换组件
- 减少收尾阶段对 symmetric memory 析构时序的敏感性

---

## 4. 编译与运行（当前可信命令）

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force

# quick correctness
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs_quick.py 2

# benchmark（当前用于复现/定位）
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

可选诊断变量：
- `DG_BENCH_MAX_SHAPES`
- `DG_BENCH_SINGLE_SHAPE`
- `DG_BENCH_SYNC_EACH_ITER`

---

## 5. 近期里程碑（建议）

1. 先把 `DG_BENCH_SINGLE_SHAPE=256,512,1024` 跑稳。
2. 扩到 3~5 个 shape，形成可重复性能样本。
3. 之后再进入 AKO4ALL 迭代与更大规模结构优化。