# DeepGEMM GEMM-RS 进度（唯一主线）

> 最后更新：2026-06-18 04:52
> 分支：`main`
> 口径：仅保留当前上线主线 `bf16_gemm_rs_nt`

---

## 当前结论（本机实测）

- **主线 GEMM-RS 正确性稳定**：`tests/test_gemm_rs.py 2` 通过（6/6）。
- **benchmark shape 口径已切换为用户指定 13 shape**，并新增重点 5 shape 的单独汇总。
- **Megatron SP 主线优化**继续围绕中大 shape。
- **学习基准**：`flux` GEMM-RS（H 卡稳定上线）作为方法学参考；本项目在 B 卡上做可复现实测适配。

---

## 已验证通过 ✅

### 多卡正确性

- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
  - 结果：**6/6 PASS**

### 主线 benchmark（用户指定 13 shape，2 GPU，4 iter）

运行：

- `MASTER_PORT=29681 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 4`

结果摘要：

- **geo mean speedup = 1.103x**
- **Best = 1.19x**
- **Worst = 0.96x**（`2048x7168x2048`）
- 平均 TFLOPS：fused **1196.1T** vs separate **1078.7T**
- `User focus medium/large`（5 shape）子集：**1.158x**

### 重点 5 shape 复测（2 GPU，6 iter）

运行：

- `MASTER_PORT=29682 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 6`

结果摘要：

- **geo mean speedup = 1.149x**
- **Best = 1.18x**
- **Worst = 1.04x**
- 平均 TFLOPS：fused **1310.2T** vs separate **1142.0T**

---

## 当前代码状态

- `benchmarks/bench_gemm_rs.py` 已切到用户指定 shape 口径。
- 新增支持：
  - `DG_BENCH_FOCUS_ONLY=1`（只跑重点 5 shape）
  - `DG_BENCH_SHAPES="M,N,K;..."`（显式 shape 列表）
- 当前主线 heuristic 已包含中大 shape 的 multicast 选择分支（持续验证中）。

---

## 推荐运行命令（接班即用）

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

MASTER_PORT=29681 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 4

MASTER_PORT=29682 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 6
```

---

## 下一步（正在执行）

1. 围绕 `K=7168` 弱势点（如 `4096x4096x7168`）继续做定向优化。
2. 保持重点 5 shape 子集为主目标集合，优先提升其几何均值。
3. 每轮收益落盘并立即 `commit + push`（防止实例中断导致丢进度）。