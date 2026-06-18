# DeepGEMM GEMM-RS 进度（单一事实来源）

> 最后更新：2026-06-18 04:16
> 适用分支：`main`
> 最近关键提交：`05a4716`（本地环境修复与基准诊断）

---

## 当前目标

在当前环境上先完成：
1. **稳定跑通 GEMM-RS 正确性链路**（已完成 2 GPU quick）
2. **恢复可复现 benchmark**（当前仍阻塞）
3. 在可复现 benchmark 基础上，继续 AKO4ALL 性能迭代

---

## 当前真实状态（以本机实测为准）

### 已完成 ✅

- `deep_gemm._C` 可编译并可导入。
- 在无 `nvcc` 环境下，`DG_JIT_USE_NVRTC=1` 路径已可用。
- `2 GPU quick` 正确性测试通过：
  - 命令：`DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs_quick.py 2`
  - 结果：`bf16_gemm_nt` 与 `bf16_gemm_rs_nt` 均 PASS。
- 已完成并推送的关键修复：
  - NVRTC include 路径补齐（`cutlass`/`cccl`）
  - NVRTC 编译失败 fail-fast（输出明确错误）
  - PTXAS 参数拼接修复（避免 `Unknown option`）
  - `sm100_bf16_gemm*.cuh` 补充 `cuda_device_runtime_api.h`
  - 多进程默认设备绑定修复（`dist.py`）
  - benchmark 诊断开关与 shape 限制能力补充

### 阻塞中 🔴

- `benchmarks/bench_gemm_rs.py` 在本机 2 GPU 上仍出现：
  - `torch.AcceleratorError: CUDA error: unspecified launch failure`
  - 错误触发位置经常表现为 `dist.barrier()` / `symmetric_memory` 析构阶段
- 该问题**不等于 quick correctness 失败**；当前更像是 benchmark 生命周期/异步错误曝光点问题。

---

## 本轮（2026-06-18）关键提交记录

### 1) `9ef2a29`
`fix(jit): make nvrtc path work on sm10.3 and fail-fast on compile errors`

### 2) `05a4716`
`wip(runtime): preserve nvrtc/bench diagnostics and dist fixes`

包含文件：
- `csrc/jit/compiler.hpp`
- `csrc/jit/device_runtime.hpp`
- `deep_gemm/utils/dist.py`
- `benchmarks/bench_gemm_rs.py`

---

## 当前推荐运行方式（本机）

### 环境准备

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force
```

### 最小可用验证

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs_quick.py 2
```

### benchmark（当前会失败，保留用于复现）

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

可用诊断开关：
- `DG_BENCH_MAX_SHAPES=1`：仅跑前 1 个 shape
- `DG_BENCH_SINGLE_SHAPE=256,512,1024`：只跑指定 shape
- `DG_BENCH_SYNC_EACH_ITER=1`：逐迭代同步便于定位异步错误

---

## 下一步执行清单（接班即做）

1. 在 benchmark 每阶段后显式 `torch.cuda.synchronize()` 并打印 shape 级里程碑，定位**首个失败点**。
2. 将 `sym_buffer.destroy()/barrier` 的调用顺序与异常兜底再收敛为单一路径，避免析构期放大错误。
3. 在 `DG_BENCH_SINGLE_SHAPE=256,512,1024` 下先跑通 2 GPU 5 iters，拿第一版稳定基线。
4. 基线稳定后再扩 shape 集，最后恢复 AKO4ALL 迭代。

---

## 重要说明

- 历史文档中存在“已超过 1.0x / 全量 shape 已稳定”等旧结论，与本机当前状态不一致。
- **后续以本文件 + 最新 commit 为准**，其余文档视作历史参考。