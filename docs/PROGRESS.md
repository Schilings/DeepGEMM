# DeepGEMM GEMM-RS 进度（唯一主线）

> 最后更新：2026-06-18 04:56
> 分支：`main`
> 口径：仅保留当前上线主线 `bf16_gemm_rs_nt`

---

## 当前结论（本机实测）

- **主线 GEMM-RS 正确性稳定**：`tests/test_gemm_rs.py 2` 通过（6/6）。
- **benchmark 口径已固定为用户指定 13 shape**，并单独追踪重点 5 个中大 shape。
- 本轮迭代以 Megatron SP 中大 shape 为主目标；`flux` GEMM-RS（H 卡）作为方法学参考，在 B 卡做适配验证。

---

## 已验证通过 ✅

### 多卡正确性

- `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
  - 结果：**6/6 PASS**

### 主线 benchmark（指定 13 shape）

运行（最新回归）：

- `MASTER_PORT=29685 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 3`

结果摘要：

- **geo mean speedup = 1.102x**
- **Best = 1.20x**
- **Worst = 0.97x**（`2048x7168x2048`）
- 平均 TFLOPS：fused **1176.1T** vs separate **1062.6T**
- `User focus medium/large`（5 shape）子集：**1.158x**

### 重点 5 shape 复测（定向目标）

运行（最新复测）：

- `MASTER_PORT=29708 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 4`

结果摘要：

- **geo mean speedup vs torch-native = 1.166x**
- **geo mean speedup vs deepgemm-separate = 1.163x**
- **Best vs torch-native = 1.22x**
- **Worst vs torch-native = 1.04x**（`4096x4096x7168`）
- 平均 TFLOPS：fused **1306.2T** vs separate **1124.7T** vs torch-native **1123.4T**

---

## 本轮调优动作（已生效）

- 在 `csrc/jit_kernels/heuristics/gemm_rs.hpp` 新增 K-heavy 中大 shape 的 multicast 选择分支：
  - `k>=7168 && n<=4096 && m_per_rank<=4096` 时倾向 `multicast=1`
- 单点验证 `4096x4096x7168`：速度从约 `1.04x` 提升到约 `1.06x`（8 iter 复测）。

---

## 当前代码状态

- `benchmarks/bench_gemm_rs.py` 已固定为用户指定 shape 口径。
- 新增支持：
  - `DG_BENCH_FOCUS_ONLY=1`（只跑重点 5 shape）
  - `DG_BENCH_SHAPES="M,N,K;..."`（显式 shape 列表）

---

## 推荐运行命令（接班即用）

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

MASTER_PORT=29685 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3

MASTER_PORT=29684 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## 下一步（正在执行）

1. 继续压低 `K=7168` 弱势点（重点 `4096x4096x7168`）。
2. 保持重点 5 shape 为主目标集合，优先提升其几何均值。
3. 每轮收益落盘并立即 `commit + push`（防止实例中断丢进度）。