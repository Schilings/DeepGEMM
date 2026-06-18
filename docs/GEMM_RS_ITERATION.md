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

## 下一步计划

1. 在稳定内核上继续做 `K=7168` 弱势点定向优化（尤其 `4096x4096x7168`）；
2. 设计“RS 本地缓冲 TMA fetch”可开关原型（先 2-rank 路径）；
3. 每轮迭代都按本文件模板沉淀：**Change / Result / Verdict**。

---

## 本轮涉及文件

- `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
- `docs/PROGRESS.md`
- `docs/FLUX_GEMM_RS_STUDY.md`
- `docs/GEMM_RS_ITERATION.md`
