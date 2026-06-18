# DeepGEMM GEMM-RS 进度（唯一主线）

> 最后更新：2026-06-18 04:38
> 分支：`main`
> 口径：仅保留当前上线主线 `bf16_gemm_rs_nt`

---

## 当前结论（本机实测）

- **不是全项目故障**：其它融合算子可正常运行。
- **主线 GEMM-RS 正确性稳定**：`tests/test_gemm_rs.py 2` 通过（6/6）。
- **Megatron SP 主线方向已切换到中大 shape**：迭代评估以 `1024/2048/4096` 档、`N/K=4096/7168` 组合为主。
- **benchmark 脚本已收敛为主线对比**：仅比较
  - `separate` = `bf16_gemm_nt + reduce_scatter_tensor`
  - `main fused` = `bf16_gemm_rs_nt`
- **学习基准**：`flux` 的 GEMM-RS 已在 H 卡稳定上线，作为策略参考；本项目在 B 卡上做可复现实测适配。

---

## 已验证通过 ✅

### 多卡正确性

- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
  - 结果：**6/6 PASS**
- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_a2a_gemm.py 2`
  - 结果：**6/6 PASS**
- `DG_JIT_USE_NVRTC=1 AG_GEMM_SHAPE_LIMIT=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_ag_gemm.py 2`
  - 结果：**1/1 PASS**

### 主线 benchmark（Megatron SP 中大 shape）

- 运行命令（2 GPU，5 iter）：
  - `DG_BENCH_MAX_SHAPES=10 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 5`
- 覆盖 shape：`1024~4096` token/rank 区间（中大 shape 主集）
- 结果摘要：
  - **geo mean speedup = 1.076x**
  - **Best = 1.18x**（`4096x7168x4096`）
  - **Worst = 0.96x**（`2048x7168x2048`）
  - 平均 TFLOPS：fused **1102.5T** vs separate **1018.5T**
- 分组表现：
  - `N=7168`：**1.069x**（5 shapes）
  - `K=7168`：**1.065x**（4 shapes）
  - `M/rank>=2048`：**1.078x**（8 shapes）

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

DG_BENCH_MAX_SHAPES=10 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## 下一步（正在执行）

1. 以 `2048x7168x2048` 为第一短板做定向优化（保持 correctness 不退化）。
2. 围绕 Megatron SP 中大 shape（`1024/2048/4096`）做持续迭代。
3. 每轮性能收益落盘到本文件并立即 `commit + push`。
4. 对照 `flux` GEMM-RS 的稳定思路，优先复用“已证实有效”的调度/通信重叠策略。