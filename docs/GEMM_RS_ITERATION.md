# GEMM-RS 迭代记录

## 背景

本文件用于记录 `bf16_gemm_rs_nt` 的**连续迭代过程**（改动、验证、结论），口径对齐 `A2A/AG` 迭代文档。

目标：

1. 面向 `SM100` 的单机（intra-node）`Reduce-Scatter + GEMM` 持续优化；
2. 以 `flux` 的设计思路作为参考，但不被既有实现绑定；
3. 每轮迭代必须包含：**改动点 -> 正确性/性能结果 -> 去留结论**。

---

## 当前基线（承接 `PROGRESS.md`）

- 主线：`bf16_gemm_rs_nt`
- 正确性：`tests/test_gemm_rs.py 2` 通过（6/6）
- 最近基线（13 shape）：geo mean 约 `1.10x`（fused vs separate）
- 最近重点 5 shape：geo mean 约 `1.17x`（vs torch-native）

> 说明：详细命令与最新统一口径以 `docs/PROGRESS.md` 为准；本文件只记录“迭代动作与结论”。

---

## 2026-06-18：Iteration 1 — 对齐 AG 风格，尝试 A/B 分离加载（失败回退）

### Change

文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`

- 参考 `AG GEMM`，尝试把 RS kernel 的 load 结构由
  - `Warp4: A+B unified load`
  改为
  - `Warp4: Load A`
  - `Warp5: Load B`
- 对应修改 stage barrier producer 计数（从单 producer 调整为双 producer）。

### Result

- 编译可过，但在实测中出现 launch failure / 不稳定行为。
- 该版本未形成稳定可复现收益。

### Verdict

- **回退**到稳定的单 load warp 版本（`A+B unified`）。
- 结论：当前 RS 的 A/B 分离加载不能直接照搬 AG，需要结合 RS 通信阶段重新设计同步协议。

---

## 2026-06-18：Iteration 2 — 通信轮询 scope 优化（保留）

### Change

文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`

- 在 ready flag 轮询中按来源 rank 分流：
  - self rank：`ld_acq`（GPU scope）
  - remote rank：`ld_acq_sys`（system scope）
- 保留 2-rank hot path 的并行轮询结构（warp0/warp1 分摊）。

### Result

- 内核恢复稳定可运行。
- 单点与 focus-only 复测显示该改动为**低风险小收益/中性偏正**，无正确性回退。

### Verdict

- **保留**。
- 结论：在不改变主 pipeline 的前提下，先吃 memory scope 精简收益是性价比最高的路径。

---

## 2026-06-18：Flux 参考结论（用于后续迭代方向）

- `flux` 的 RS 路径中，TMA 主要用于**本地可见缓冲区搬运/读取**；
- 跨 rank 通信仍主要依赖 `nvshmem put/get + barrier/signal`（不是“跨 GPU 直接 TMA 通信”）。

这意味着后续在 `DeepGEMM` 上可优先尝试：

1. 保持跨 rank 通信路径稳定；
2. 把“本地 reduce/fetch 阶段”做成更强的 TMA 化与流水解耦；
3. 再评估是否值得重构成更接近 flux 的双核或职责拆分模型。

---

## 2026-06-18：Iteration 3 — 重构为真·Flux pull 式 dual-kernel（进行中）

### Change

把主线 `sm100_bf16_gemm_rs.cuh` 从「单 kernel push + in-kernel reduce」彻底重构为
**真·Flux pull 式 dual-kernel**，对齐 Flux 的
`Sm90AuxStoreReduceScatter`(GEMM 写本地 scatter buffer) + `Sm90ReduceScatterDma`(独立 kernel 从远端 pull)：

- `impls/sm100_bf16_gemm_rs.cuh`（Kernel 1, GEMM compute）：
  - 256T(8 warps)，**移除 comm warps**（旧版 384T → 寄存器 spilling 的根因被消除）；
  - epilogue **纯本地 scatter write**：每个 tile 按 `dst_rank` 写到本地 `slot[dst_rank]`，
    置**本地** per-tile ready flag（`st_rel_sys`，system scope 可被 peer 读到）；
  - GEMM epilogue **不再跨 NVLink**（这是相对 push 式 v3 的核心收益）。
- `impls/sm100_rs_reduce.cuh`：新增 `kPullBased` 模板分支：
  - rank R 对自身 chunk 每个 tile，poll 各 src rank 的**远端** `flag[R][m][n]`（`sym_buffer.map` 到 src）；
  - FP32 累加各 src 的**远端** `slot[R]`（map P2P，self 映射回本地）→ 写 output；
  - 读完后远端 reset flag（Flux `wait_eq_reset` 语义）。
- `jit_kernels/impls/sm100_bf16_gemm_rs.hpp`：改为 dual-kernel pull 编排
  （compute_stream 跑 GEMM + comm_stream 跑 pull reduce + event 同步），复用 `GemmRSComputeConfig`。
- `jit_kernels/impls/sm100_rs_reduce.hpp`：把 `pull_based` 透传到 Args/generate_impl（push v3 路径不受影响）。
- `deep_gemm/gemm_rs/__init__.py`：默认路由切到 pull；保留 `DG_GEMM_RS_IMPL=v3/push` 旧 push 路径可选。

跨迭代正确性：GEMM 起始 `nvlink_barrier` + host 端 `cudaStreamSynchronize(comm_stream)`/event 门控，
保证「上一轮所有 rank 的 reduce(含远端 reset) 完成 → 本轮才 set flag」，杜绝 stale flag 读。

### Result

- C++ 扩展编译通过，代码已 commit & push。
- **Bug 修复**：首轮 2-GPU 测试死锁（nvlink_barrier timeout）。根因是 host 端 per-call
  `cudaMemsetAsync` 清零 barrier 区与对端 GEMM 的 in-flight NVLink 信号写竞争
  （`cudaStreamSynchronize(comm_stream)` 延迟某 rank 的 memset → 把对端已送达的信号清掉 → 死锁）。
  `comm::nvlink_barrier` 本身是 phase/sign 自复位协议，buffer 创建时已 `zero_()`，
  故 per-call memset 既不必要又有害 → **移除 memset**。
- **正确性**：`tests/test_gemm_rs.py 2` → **6/6 PASS，max_diff=0.0**（与参考逐元素精确一致）。
- **性能（2 GPU，3 iter，指定 13 shape）**：geo_mean **0.584x vs torch / 0.582x vs sep**，
  fused 平均 **628.5T** vs separate 1065.2T。**明显慢于旧 push v3（~1.10x）**。

### Verdict

- **正确性达标，性能回退（需优化）**。
- 根因（性能）：真·Flux pull 的速度优势依赖 Flux `Sm90ReduceScatterDma` 的 **TMA 流水线 fetch**；
  当前 RS reduce 用**朴素标量 P2P 读**（每元素串行读 num_ranks 个远端 slot），且 reduce 与 GEMM
  双流**抢占 SM/带宽**。而 push v3 把跨卡传输放在 GEMM epilogue 的 TMA async store 中被计算掩盖、
  reduce 只读本地，所以更快。
- 下一步（perf）：把 pull reduce 改造为 **TMA 流水线 fetch+reduce**（对齐 Flux `Sm90ReduceScatterDma`：
  远端 → smem 的 producer/consumer 流水），或改进 GEMM/reduce 的 SM 划分与重叠；
  在此之前，主线高性能路径仍可用 `DG_GEMM_RS_IMPL=v3`（push）回退。

---

## 2026-06-18：Iteration 4 — 收敛到 pull，删除 push 路径

### Change

既然 Flux 单机 RS = pull（`Sm90ReduceScatterDma` 为 TMA 远端 fetch；push 仅用于跨节点 ring），
主线收敛为唯一 pull 实现，**删除 push 路径**：

- 删除 `impls/sm100_bf16_gemm_rs_compute.cuh`、`jit_kernels/impls/sm100_bf16_gemm_rs_compute.hpp`、
  `apis/gemm_rs_compute.hpp`、孤立的 `heuristics/gemm_rs.hpp`、`*.cuh.bak`；
- 删除 v3 test/bench：`tests/test_gemm_rs_v3.py`、`quick_bench_v3.py`、
  `benchmarks/bench_gemm_rs_v3.py`、`benchmarks/bench_gemm_a2a_pdl.py`；
- `sm100_rs_reduce.cuh/.hpp` 收敛为 **pull-only**（移除 `kPullBased`/`pull_based` 开关）；
- `python_api.cpp` 移除 `gemm_rs_compute` 注册；`gemm_rs/__init__.py` 移除 `bf16_gemm_rs_nt_v3` 与
  `DG_GEMM_RS_IMPL` 开关，`deep_gemm/__init__.py` 移除其导出。

### Result

- 编译通过；`tests/test_gemm_rs.py 2` → **6/6 PASS, max_diff=0.0**（清理后正确性不变）。

### Verdict

- **保留**。主线唯一 = 真·Flux pull。性能优化转入下一步（TMA 流水线 fetch）。

---

## 下一步计划

1. **核心**：把 pull RS reduce 从朴素标量 P2P 读改造为 **TMA 流水线 fetch+reduce**
   （远端→smem 的 producer/consumer，对齐 Flux `Sm90ReduceScatterDma`），并优化 GEMM/reduce 的 SM 划分与重叠；
2. 继续 `K=7168` 弱势点（`4096x4096x7168`）定向优化；
3. 每轮迭代按本文件模板沉淀：**Change / Result / Verdict**。

---

## 本轮涉及文件

- `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
- `deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh`
- `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`
- `csrc/jit_kernels/impls/sm100_rs_reduce.hpp`
- `deep_gemm/gemm_rs/__init__.py`
- `docs/PROGRESS.md`
- `docs/GEMM_RS_DESIGN.md`
- `docs/SESSION_MEMORY.md`
- `docs/FLUX_GEMM_RS_STUDY.md`
- `docs/GEMM_RS_ITERATION.md`
