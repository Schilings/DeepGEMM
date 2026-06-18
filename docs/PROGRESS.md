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
- **性能（2 GPU，13 shape）**：geo_mean **0.835x vs torch / 0.836x vs sep**，fused 平均 **906T**
  （本会话起点 0.606x / 660T）。单 shape 0.77~0.87x（最差点从 0.51x 提升到 0.77x）。
  - **Iteration 5（高 MLP reduce 重写）**：reduce 原为延迟/occupancy-bound（MLP≈2），重写为预计算固定
    远端基址 + 每线程 `kUnroll=8` 批量发射 P2P load（MLP 高）。→ 0.733x / 814T。
  - **Iteration 6**：确认 SM carveout 对 SM-based reduce 是零和死胡同（`DG_RS_REDUCE_SMS` 保留默认 0）。
  - **Iteration 7（reduce grid 过订阅 ×2）**：reduce 在 GEMM 后跑、SM 全空，每 SM 多 block → 更多 warp
    并发发射 P2P load → 更高 NVLink 有效带宽。`DG_RS_REDUCE_MULT` 默认 2。→ **0.835x / 906T**。
  详见 `GEMM_RS_ITERATION.md`(Iteration 5/6/7)。
- **架构收敛**：既然对齐 Flux（单机 RS = pull），**已删除 push 路径**（旧 `v3`/compute kernel、
  `gemm_rs_compute` API、相关 test/bench）。主线唯一实现 = pull，`DG_GEMM_RS_IMPL` 开关已移除。
- 下一步（冲 >1.0x）：把跨卡搬运从 SM/LSU 移到 **TMA 引擎**（`cp.async.bulk` 异步 fetch 远端连续段 →
  smem，SM 仅做加法），并与 GEMM 真正共驻 overlap（对齐 Flux `Sm90ReduceScatterDma`）。

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