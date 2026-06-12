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

- 跑通 `4/8 GPU` correctness
- 验证 chunk-ready 等待逻辑没有死锁 / 越界 / 顺序问题
- 去掉测试脚本里的临时 debug 开关或把它们整理成正式 debug 模式

### Phase 3

- 基于现有 `benchmarks/bench_ag_gemm.py` 继续扩形状与 GPU 数
- 和 `Flux` / `separate all_gather + gemm` 做 baseline 对比
- 评估 `256T compute-only` 相比旧 `384T` kernel 的吞吐变化；当前 `2 GPU` geo mean 为 `0.952x`

### Phase 4

- 继续优化 chunk 大小、copy 顺序、rank swizzle
- 看是否需要引入更细粒度 flag 或多阶段 local-ready

---

## 本轮涉及文件

- `docs/AG_GEMM_ITERATION.md`
- `deep_gemm/include/deep_gemm/layout/bf16_ag_gemm.cuh`
- `deep_gemm/include/deep_gemm/impls/sm100_bf16_ag_gemm.cuh`
- `csrc/jit_kernels/impls/sm100_bf16_ag_gemm.hpp`
- `csrc/jit_kernels/heuristics/ag_gemm.hpp`
- `csrc/apis/ag_gemm.hpp`
