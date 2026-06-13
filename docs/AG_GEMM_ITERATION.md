# AG GEMM 迭代记录

## 背景

本轮开始单独迭代 `AG GEMM`（All-Gather + GEMM）算子，目标是把当前 `SM100` 实现从 **单 kernel 内自带 NVLink ring-push 通信**，改成更接近 `Flux` 的形式：

- **通信放到 kernel 外**（host-side / comm stream）
- **GEMM kernel 内不再主动搬运跨卡数据**
- kernel 只负责：
  - 按 tile / chunk 等待 ready flag
  - TMA load A/B
  - UMMA
  - epilogue store

这样做的目标是：

1. 减少 kernel 内为通信保留的 warp 资源
2. 把跨卡搬运切到 `cudaMemcpyAsync` / CE 路线，避免 SM 直接参与通信
3. 为后续做更细粒度的 `chunk-ready` overlap 打基础

---

## Baseline 结论（旧实现）

旧版 `deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh` 的特点：

- 采用 **单 kernel 融合**
- `W0-W3` 是 AG warps，在 kernel 内做 peer-mapped NVLink push
- GEMM warp 组随后从本地 `slot_x` 里 TMA 读取数据
- `slot_state[src_rank]` 只有粗粒度 ready 语义，本质上是“整 rank 数据 ready”

和 `Flux` 相比，差异在于：

- `Flux` 的通信不在 GEMM kernel 里
- `Flux` 让 kernel **只做等待 + 计算**
- `Flux` 的 overlap 依赖更细粒度的 ready / barrier 机制，而不是 kernel 内自己做 ring push

---

## 2026-06-12：AKO4ALL 迭代启动

- 按 `docs/RULE.md` 要求切换到 AKO4ALL 方式继续推进 `AG GEMM`
- 不使用 `solution/` 或 `opt/` 分支，直接在 `main` 原文件上迭代
- 使用项目原生 test / benchmark 驱动，而不是 AKO4ALL 默认 bench scaffold
- 当前先补 `tests/test_ag_gemm.py` 与 `benchmarks/bench_ag_gemm.py`，然后基于其结果继续优化

---

## 2026-06-12：Phase 1 — Flux 风格重构骨架

### 已完成

#### 1. BF16 workspace ready-flag 改为显式 chunk 语义

文件：`deep_gemm/include/deep_gemm/layout/bf16_ag_gemm.cuh`

- 新增 `kNumReadyChunksPerSlot = 4`
- `slot_state` 区域大小改为 `num_slots * 4 * sizeof(uint32_t)` 的显式语义
- 为了和稳定的 `BF16 A2A` 对称内存骨架保持一致，`local_x` 预留区扩成 `num_ranks` 份 token buffer，`slot_x` 基址相应后移
- `get_slot_state_ptr(slot_idx, chunk_idx)` 支持按 slot + chunk 访问

#### 2. BF16 AG GEMM kernel 改为 compute-only 形态

文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh`

核心变化：

- 去掉旧版 kernel 里的 AG warp 通信阶段
- 删除 startup 时的 `slot_state` 清零 + `nvlink_barrier`
- 允许 `kNumAGThreads = 0`
- 保留 GEMM warp / epilogue warp 流水
- 在 load A 之前按 `src_rank + local_m` 计算当前 tile 依赖的 chunk 区间
- 对 chunk-ready flag 做 system-scope polling：ready 后再发起 TMA load A
- 调度顺序改成 **从本 rank 的 M chunk 开始**，更接近 Flux 的 local-first 计算顺序

#### 3. Host-side 通信编排接入独立通信流

文件：`csrc/jit_kernels/impls/sm100_bf16_ag_gemm.hpp`

- 新增 `launch_bf16_ag_gemm_comm(...)`
- 在高优先级 comm stream 上：
  - `cudaMemsetAsync` 清零本地 `slot_state`
  - 用显式 local `cudaMemcpyAsync` / remote `cudaMemcpyPeerAsync` 把各 rank 的 `local_x` 拷到本地 `slot_x[src_rank]`
  - 每个 chunk 拷完后用 `cudaMemsetAsync` 置位本地 ready flag
- 先完成 self chunk，再 record `local_ready_event`
- GEMM 主 stream 只等待 `local_ready_event`，然后立刻启动 compute kernel
- 远端 rank 的 chunk 继续在 comm stream 后台搬运，kernel 侧按 ready flag 自旋等待

#### 4. Launch 配置去掉 AG warps

文件：`csrc/jit_kernels/heuristics/ag_gemm.hpp`

- `num_ag_threads: 128 -> 0`
- 当前 BF16 AG GEMM launch 从 `384T` 变为 `256T`

#### 5. API 串联更新

文件：`csrc/apis/ag_gemm.hpp`

- `bf16_ag_gemm_nt(...)` 现在把 `sym_buffer` 本体传给新的 runtime，以便 host 侧通信编排直接访问本地对称 buffer

---

## 当前状态

这是一个 **第一阶段结构性改造**，重点是把旧的“in-kernel AG”拆开，建立 `Flux-style AG` 的骨架。

### 已实现能力

- 通信与计算的职责已经拆开
- kernel 已具备按 chunk-ready 等待的能力
- host 已具备独立 comm stream 的准备逻辑
- 调度顺序已从 rank-local M chunk 开始，便于 overlap

### 还未完成 / 待验证

1. **功能正确性测试**
   - 目前仓库里还没有单独的 `test_ag_gemm.py`
   - 下一步需要补 2/4/8 GPU correctness test

2. **性能 benchmark**
   - 需要新增 `bench_ag_gemm.py`
   - 对比对象：
     - `dist.all_gather + bf16_gemm_nt`
     - 新版 `bf16_ag_gemm_nt`

3. **更细粒度 chunk 策略**
   - 目前固定 `4 chunks / rank`
   - 后续可按 `M_per_rank / BLOCK_M` 自适应 chunk 数量

4. **调度细化**
   - 当前是 local-first 的简单 rank-offset M swizzle
   - 后续可以按 Flux 的思想继续做更精细的 block 排序

5. **通信后端扩展**
   - 目前先走 `cudaMemcpyAsync + stream write value`
   - 后续可对比：
     - 单 stream pull
     - 多 stream / staged copies
     - CE 并发上限

---

## 后续计划

### Phase 2

- 继续定位 BF16 AG 首轮 cold-start 的根因，目标是去掉 correctness test 里的显式 warmup launch
- 验证 chunk-ready 等待逻辑在更多形状与更多轮次下没有死锁 / 越界 / 顺序问题
- 去掉测试脚本里的临时 debug 开关或把它们整理成正式 debug 模式

### Phase 3

- 基于现有 `benchmarks/bench_ag_gemm.py` 继续扩形状与 GPU 数
- 和 `Flux` / `separate all_gather + gemm` 做 baseline 对比
- 评估 `256T compute-only` 相比旧 `384T` kernel 的吞吐变化；当前 geo mean：`2 GPU 0.952x` / `4 GPU 0.953x` / `8 GPU 0.946x`

### Phase 4

- 继续优化 chunk 大小、copy 顺序、rank swizzle
- 看是否需要引入更细粒度 flag 或多阶段 local-ready

---

---

## 2026-06-12：Phase 2 — 启用 comm-compute overlap

### 问题

Phase 1 虽然实现了通信外移 + per-chunk barrier，但 host 端在 launch kernel 前：
1. `cudaStreamWaitEvent(comm_done_event)` — 等待**全部**远程数据拷贝完成
2. `ready_chunk_rows = runtime_m_per_rank; num_ready_chunks = 1` — 使所有 tile poll chunk 0

**结果**：通信和计算串行，无 overlap。对比 Flux 参考，这是关键缺陷。

### 修复（仅 host 端 13 行删除，kernel 无需改动）

文件：`csrc/jit_kernels/impls/sm100_bf16_ag_gemm.hpp`

```diff
- // Wait for all communication to complete before kernel launch.
- DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(current_stream.stream(), comm_done_event, 0));
- ready_chunk_rows = static_cast<uint32_t>(runtime_m_per_rank);
- num_ready_chunks = 1;
+ // ✅ OVERLAP: kernel launches after local_ready_event (local data ready),
+ //    remote chunks still copying on comm_stream → kernel polls per-chunk barrier
```

`launch_bf16_ag_gemm_comm` 已经 wired `current_stream` wait `local_ready_event` 后才返回 → kernel 在 local data 就绪后立即 launch。`ready_chunk_rows` / `num_ready_chunks` 使用真实值。

### Overlap 机制（SM100）

```
Copy Stream:         copy local → record local_ready → copy remote_0 → copy remote_1 → ...
                     barrier[self]=1                barrier[0]=1       barrier[1]=1

Compute Stream:      wait(local_ready) → launch kernel
                     kernel: poll barrier[0] → TMA load → UMMA → epilogue
                             poll barrier[1] → TMA load → UMMA → epilogue
                             ...
```

Kernel 内 `ptx::ld_acq_sys(slot_state[src_rank][chunk])` 自旋等待，CE DMA 完成后 `cudaMemsetAsync(flag=1)` 唤醒。

### 测试验证

- 正确性：2/4/8 GPU, basic + extended 全量 shapes ✅ ALL PASS
- Benchmark (4 GPU, large training shapes): Geo Mean 0.920x（vs 旧 0.930x）
  - 大 shape 下 comm <2% 总时间，barrier polling 微量开销抵消了 overlap 收益
  - 架构正确性优先于短期 benchmark

---

## 2026-06-12：Flux 参考分析

深入分析了 Flux 项目的 BF16 + SM90 + 单机 AG GEMM 设计，输出参考文档 `docs/AG_GEMM_FLUX_REFERENCE.md`，核心发现：

1. **通信与计算并行方式**：
   - CE DMA（`cudaMemcpyAsync` + `CUStreamWriteValue`）在 Copy Stream 搬运数据
   - GEMM kernel 在 Compute Stream 上逐 tile 通过 `SystemBarrier::wait_eq` 等待 barrier flag
   - CE 和 SM 是不同的硬件单元，真正并行执行

2. **Barrier 粒度**：per data_chunk（M/world_size 行），不等全矩阵 AG 完成即可开始计算

3. **调度优化**：Tile Swizzle 将 tile 顺序按 rank 做 M 方向偏移，使每个 rank 优先计算本地数据对应的 tile（local-first）

4. **与 DeepGEMM 当前实现的对应**：Phase 1 重构已覆盖 Flux 的核心设计要点（通信外移、per-chunk barrier、local-first swizzle）。后续细化方向：chunk 自适应数量、双 consumer warpgroup 交替、SM100 barrier slot 细节

---

## 本轮涉及文件

- `docs/AG_GEMM_ITERATION.md`
- `docs/AG_GEMM_FLUX_REFERENCE.md`
- `deep_gemm/include/deep_gemm/layout/bf16_ag_gemm.cuh`
- `deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh`
- `csrc/jit_kernels/impls/sm100_bf16_ag_gemm.hpp`
- `csrc/jit_kernels/heuristics/ag_gemm.hpp`
- `csrc/apis/ag_gemm.hpp`

---

## Iteration 1 — Enable PDL (Programmatic Dependent Launch)

### Change

File: csrc/jit/device_runtime.hpp

- Changed enable_pdl default from false to true
- Rationale: LaunchArgs constructor defaults enable_pdl=true, but LaunchRuntime::launch() overrides with device_runtime->get_pdl() which was false. PDL was effectively disabled for all kernels including AG GEMM.

### Results (8 GPU, 10 iters)

| Shape (M/rank x N x K) | Sep TFLOPS | Fus TFLOPS | Speedup |
|---|---|---|---|
| 4096x4096x4096 | 859T | 1254T | 1.46x |
| 4096x7168x4096 | 1082T | 1297T | 1.20x |
| 4096x7168x7168 | 1077T | 1162T | 1.08x |
| 6144x4096x4096 | 869T | 1297T | 1.49x |
| 6144x7168x4096 | 1104T | 1243T | 1.13x |
| 6144x7168x7168 | 1092T | 1107T | 1.01x |
| 8192x4096x4096 | 869T | 1316T | 1.52x |
| 8192x7168x4096 | 1108T | 1245T | 1.12x |
| 8192x7168x7168 | 1067T | 1085T | 1.02x |
| 10240x7168x4096 | 1102T | 1192T | 1.08x |
| 10240x7168x7168 | 1054T | 1057T | 1.00x |
| 12288x7168x4096 | 1106T | 1158T | 1.05x |
| 12288x7168x7168 | 1024T | 1068T | 1.04x |
| 16384x7168x4096 | 1096T | 1126T | 1.03x |
| 16384x7168x7168 | 990T | 1064T | 1.07x |
| 20480x7168x4096 | 1076T | 1132T | 1.05x |
| 20480x7168x7168 | 965T | 1044T | 1.08x |

- **Geo Mean: 1.133x** (baseline: 1.135x) — within noise
- **17/17 wins** (same as baseline)

### Analysis

PDL enables programmaticStreamSerializationAllowed on kernel launch, allowing the next kernel to start while the current one is still finishing. However, AG GEMM is a single persistent kernel with its own comm-compute overlap pipeline, so PDL does not provide meaningful benefit in microbenchmarks. The change is kept because:
1. It aligns with LaunchArgs original design intent (default enable_pdl=true)
2. It does not regress performance
3. It may help in multi-kernel training loops where AG GEMM is called back-to-back

### Verdict: NEUTRAL — kept, move to next optimization

---

## Iteration 2 — Pipeline stages 7->8 (remove SF from smem calc)

### Change

File: csrc/jit_kernels/heuristics/ag_gemm.hpp

- Removed smem_sfa_per_stage + smem_sfb_per_stage from smem_per_stage calculation
- BF16 AG GEMM does not use MX scaling factors, so SF smem was overcounted
- This allowed num_stages to go from 7 to 8

### Results (8 GPU, 10 iters)

- Geo Mean: 1.116x (down from 1.133x baseline) — REGRESSION
- 17/17 wins

### Analysis

8 pipeline stages hurt small shapes (K=4096) due to longer pipeline fill/drain overhead:
- 4096x4096x4096: 1.46x -> 1.31x (-0.15)
- 8192x4096x4096: 1.52x -> 1.47x (-0.05)
- 12288x7168x7168: 1.04x -> 1.01x (-0.03)

K=7168 shapes were roughly unchanged or slightly improved. The tradeoff is not favorable.

### Verdict: REGRESSION — reverted, try next direction

---

## Iteration 3 — Barrier polling with __nanosleep backoff

### Change

File: deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh

- Added __nanosleep(200) with exponential backoff to chunk-ready polling loop
- First 128 iterations: busy-wait, then: nanosleep(200ns) between polls
- Rationale: reduce SM overhead and L2 pressure during long waits for remote chunks

### Results (8 GPU, 10 iters)

- Geo Mean: 1.126x (down from 1.133x) — slight REGRESSION
- 17/17 wins

### Analysis

The nanosleep backoff did not help. The polling is done by a single thread (elect_one) per CTA, so SM overhead from busy-waiting is minimal. Chunk wait times are short enough (sub-microsecond for most shapes) that the busy-spin is more efficient than introducing nanosleep overhead. The 200ns sleep adds latency to the fast path without sufficient benefit on the slow path.

### Verdict: REGRESSION — reverted, try next direction
