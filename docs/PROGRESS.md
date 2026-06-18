# DeepGEMM GEMM-RS 进度（唯一主线）

> 最后更新：2026-06-18 04:28
> 分支：`main`
> 口径：仅保留当前上线主线 `bf16_gemm_rs_nt`

---

## 当前结论（本机实测）

- **不是全项目故障**：其它融合算子可正常运行。
- **主线 GEMM-RS 正确性已稳定**：`tests/test_gemm_rs.py 2` 通过（6/6）。
- **benchmark 脚本已收敛为主线对比**：仅比较
  - `separate` = `bf16_gemm_nt + reduce_scatter_tensor`
  - `main fused` = `bf16_gemm_rs_nt`

---

## 已验证通过 ✅

### 多卡正确性

- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
  - 结果：**6/6 PASS**
- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_a2a_gemm.py 2`
  - 结果：**6/6 PASS**
- `DG_JIT_USE_NVRTC=1 AG_GEMM_SHAPE_LIMIT=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_ag_gemm.py 2`
  - 结果：**1/1 PASS**

### 主线 benchmark（收敛后）

- 单 shape（`256,512,1024`，2 GPU，5 iter）
  - **主线 fused ≈ 1.53x**（约 9.5T vs 6.2T）
- 前 3 个标准 shape（2 GPU，3 iter）
  - **geo mean ≈ 1.038x**
  - 平均 TFLOPS：fused 1036.7T vs separate 998.5T

---

## 当前代码状态

- `benchmarks/bench_gemm_rs.py` 已去除历史多版本比较逻辑，仅保留唯一主线评测路径。
- JIT/NVRTC 路径可用（无 `nvcc` 环境按 `DG_JIT_USE_NVRTC=1` 运行）。

---

## 推荐运行命令（接班即用）

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

DG_BENCH_MAX_SHAPES=3 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3
```

---

## 下一步（正在执行）

1. 扩展到 5~8 个 shape，建立稳定小样本基线。
2. 基于主线脚本做参数与调度层优化。
3. 每一轮性能收益都落盘到本文件并立即 commit + push。
