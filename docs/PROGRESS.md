# DeepGEMM GEMM-RS 进度（唯一主线）

> 最后更新：2026-06-18 07:18
> 分支：`main`
> 口径：仅保留当前上线主线 `bf16_gemm_rs_nt`

---

## 当前结论（本机实测）

- **【架构重构完成 · 正确性达标 · 性能待优化】** 主线 `bf16_gemm_rs_nt` 已重构为
  **真·Flux pull 式 dual-kernel**（GEMM 256T 无 comm warps + epilogue 纯本地 scatter write；
  独立 RS reduce kernel 从远端 pull）。详见 `GEMM_RS_DESIGN.md` / `GEMM_RS_ITERATION.md`(Iteration 3)。
- **正确性**：`tests/test_gemm_rs.py 2` → **6/6 PASS，max_diff=0.0**（逐元素精确匹配参考）。
  （修复了一处 nvlink_barrier 死锁：移除了与对端信号竞争的 per-call barrier memset。）
- **性能（2 GPU，3 iter，13 shape）**：geo_mean **0.584x vs torch / 0.582x vs sep**，
  fused 628.5T vs sep 1065.2T —— **慢于旧 push v3（~1.10x）**。
  根因：pull reduce 用朴素标量 P2P 读 + 与 GEMM 抢 SM；需改为 TMA 流水线 fetch（Flux `Sm90ReduceScatterDma` 风格）。
- **高性能回退**：旧 push dual-kernel 仍可用 `DG_GEMM_RS_IMPL=v3`（或 `push`）。

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

- `MASTER_PORT=29729 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python benchmarks/bench_gemm_rs.py 2 4`

结果摘要：

- **geo mean speedup vs torch-native = 1.174x**
- **geo mean speedup vs deepgemm-separate = 1.161x**
- **Best vs torch-native = 1.24x**
- **Worst vs torch-native = 1.05x**（`4096x4096x7168`）
- 平均 TFLOPS：fused **1304.6T** vs separate **1126.1T** vs torch-native **1115.1T**

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

MASTER_PORT=29728 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3

MASTER_PORT=29729 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 4
```

---

## 下一步（正在执行）

1. **排查 pull 路径首轮 2-GPU 运行期错误**：抓首屏真实报错（kernel assert / nvlink barrier timeout / IMA），
   定位 pull 同步或寻址问题，跑通 `tests/test_gemm_rs.py 2`（目标 6/6）。
2. 跑通后用 `bench_gemm_rs.py` 重测 pull 基线，与旧 push(v3) 对比；验证 GEMM 吞吐是否因消除
   寄存器 spilling 而显著提升。
3. 继续压低 `K=7168` 弱势点（重点 `4096x4096x7168`）。
4. 每轮收益落盘并立即 `commit + push`（防止实例中断丢进度）。