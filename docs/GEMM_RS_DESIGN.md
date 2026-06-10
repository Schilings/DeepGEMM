# GEMM+RS: Pull-based Single-Kernel Fusion with Tile-Level Overlap

## 概述

DeepGEMM 的 GEMM-RS 融合 kernel，在 NVIDIA Blackwell (SM100) 架构上实现**计算-通信 tile 级流水线重叠**。
借鉴 ByteDance Flux 和 DeepSeek MegaMoe 的设计思想。

> **状态**: 代码已完成，待多卡环境测试验证。

## 背景与动机

之前尝试过 Push+PDL 两阶段方案，验证后发现性能不可接受：
- Symmetric Push O(N) 通信量随 rank 数线性增长
- 无真正 overlap：全部 GEMM 完成才通信
- 8GPU 下仅 NCCL 分离方案的 0.21x

当前方案的目标：
- 通信量降到 bandwidth-optimal: O((N-1)/N)
- 实现 tile 粒度的计算-通信重叠
- 单 kernel 完成所有工作，消除 inter-kernel 开销

## 核心设计

| 维度 | 说明 |
|------|------|
| Kernel 数量 | 1 (全融合) |
| 通信方向 | Pull (读远端) |
| 通信模型 | All-to-1 Pull O((N-1)/N) |
| Overlap 粒度 | Tile 级流水线 |
| 同步模型 | Per-tile ready flag |
| 通信带宽效率 | Bandwidth-optimal (= NCCL) |
| Reduce 方式 | Comm Warps 内融合 FP32 reduce |

## 架构设计

### Warp 分工 (384 threads = 12 warps)

```
┌─────────────────────────────────────────────────────────────────┐
│ Warp 0-3 (128T, 48 regs): Comm Warps — Pull-based RS           │
│   - Per-rank pipelined reduce (Flux-style ring order)           │
│   - Poll per-tile ready flags from each rank                    │
│   - NVLink P2P Read (pull remote partial)                       │
│   - FP32 accumulate → write final output                        │
├─────────────────────────────────────────────────────────────────┤
│ Warp 4 (32T, 40 regs): TMA Load A                              │
│   - TMA multicast load A tiles → smem (2-CTA)                  │
├─────────────────────────────────────────────────────────────────┤
│ Warp 5 (32T, 40 regs): TMA Load B                              │
│   - TMA multicast load B tiles → smem (2-CTA)                  │
├─────────────────────────────────────────────────────────────────┤
│ Warp 6 (32T, 40 regs): MMA Issue                               │
│   - 单 warp 发射 UMMA FMA (Blackwell: 1 warp 驱动 TC)          │
├─────────────────────────────────────────────────────────────────┤
│ Warp 7 (32T, 40 regs): Reserved                                │
│   - TMEM allocation + keep alignment                            │
├─────────────────────────────────────────────────────────────────┤
│ Warp 8-11 (128T, 208 regs): Epilogue Warps                     │
│   - TMEM → smem → local partial buffer (TMA bulk store)         │
│   + per-tile ready flag signaling (fence_system + st_rel)       │
└─────────────────────────────────────────────────────────────────┘
```

### 寄存器预算 (SM100 Max = 64512)

```
48 × 128 (comm) + 40 × 128 (non-epi) + 208 × 128 (epilogue)
= 6144 + 5120 + 26624 = 37888  ← 充裕！剩余 26624 可用于 occupancy
```

### 数据流（Tile 级流水线）

```
时间 →
                  Tile 0         Tile 1         Tile 2         ...
                  ──────         ──────         ──────
GEMM Warps:      [compute]      [compute]      [compute]
                       │              │              │
Epilogue:        [TMEM→smem→    [TMEM→smem→    [TMEM→smem→
                  local buf]     local buf]     local buf]
                  set flag_0     set flag_1     set flag_2
                       │              │              │
                       ↓              ↓              ↓
Comm Warps:      (waiting)      poll flag_0 →   poll flag_1 →
                               pull + reduce   pull + reduce
                               → output[0]     → output[1]
```

### M 维 Swizzle 调度

Rank i 计算 tile 的顺序：
1. 先计算属于 rank (i+1) 的 chunk → 写本地 partial + set flag
2. 再计算属于 rank (i+2) 的 chunk
3. ...
4. 最后计算属于自己 rank i 的 chunk

**目的**：接收方的 Comm Warps 能尽早开始从 peer 拉取数据，最大化 overlap 窗口。

### 同步机制

| 同步点 | 机制 | 说明 |
|--------|------|------|
| GEMM → Epilogue | `tmem_full` / `tmem_empty` barriers | TMEM 流水线（经典模式） |
| Epilogue → Comm (跨 rank) | `__threadfence_system()` + `st_rel_sys(flag, 1)` | Per-tile ready flag |
| Comm poll (跨 rank) | `ld_acq_sys(flag)` 自旋等待 | 检测远端 tile 就绪 |
| Kernel 结束 | `nvlink_barrier` | 确保所有 rank 完成 pull 后再 reset flags |

### 通信量分析

对于 N 个 rank，每个 rank 的输出大小为 `M_per_rank × N_dim`：

- **旧方案 (Symmetric Push)**: 总通信量 = `N × (N-1) × chunk_size` (全系统)
- **当前方案 (All-to-1 Pull)**: 总通信量 = `N × (N-1) × chunk_size / N` = `(N-1) × chunk_size`
  - 等同于 NCCL ring reduce-scatter 的 bandwidth-optimal 通信量

## 从 Flux 学到的关键设计思想

### 1. Per-Rank Pipelined Reduce（逐 Rank 流水 Reduce）

Flux 的核心创新：**不等 ALL ranks ready，而是逐个 rank 处理**。

```
for each tile in my_chunk:
    acc = 0 (FP32)
    for src_rank in ring_order(rank_idx):
        wait(src_rank's ready flag for this tile)
        load src_rank's partial → smem/regs
        reduce: acc += partial
    store acc → final output
```

**优势**：
- 减少等待时间：某个 rank ready 了就立即开始处理
- Ring order 匹配 M-Swizzle：rank(i+1) 是我们最先计算完的 chunk 所属 rank，
  所以从 rank(i+1) 开始 pull 时它最可能已经 ready

### 2. Warp Specialization + Register Reconfig

Flux 的 4 种 Producer Warp 角色：
- Mainloop Load / Epilogue Load / RS Fetch / RS Reduce

我们的对应关系：
- Load A (W4) / Load B (W5) / MMA (W6) / Reserved (W7) — GEMM 侧
- Comm W0-W3 = RS Fetch + Reduce（合并为统一的 pull+reduce 循环）
- Epilogue W8-W11 = 写本地 partial + 设 flag

### 3. Per-tile Flag Granularity

Flux 使用 128B 对齐的 per-tile flag，通过 NVLink P2P 映射的全局内存：
- `GenericSystemBarrier` 使用 system-scope 原子操作
- 我们使用 `ld_acq_sys` / `st_rel_sys` PTX 指令直接操作

### 4. TMA for Remote Fetch

Flux 在 Hopper 上用 TMA 从远端拉取数据到 SMEM，然后在 SMEM 中做 reduce。

当前我们的实现用 **global load (uint4 vectorized)** 做 P2P read。
**后续优化方向**：改用 TMA 1D bulk load 从远端拉取（硬件异步，更高效）。

## 从 MegaMoe 学到的关键设计模式

### 1. Dispatch Warp 通信模型

MegaMoe 的 Dispatch Warps (W0-W3) 负责 NVLink 通信：
- 每个 warp 独立工作，有自己的 smem buffer + mbarrier
- Round-robin 分配 token 到不同 warp
- TMA 1D load from remote → smem → TMA 1D store to local

我们的 Comm Warps 类似设计：
- 4 个 warp (W0-W3) 共同处理 tile-level pull + reduce
- 通过 NamedBarrier 做 warp 间同步

### 2. 非均匀寄存器分配

MegaMoe 使用 `warpgroup_reg_dealloc<N>()` / `warpgroup_reg_alloc<N>()`：
- 通信 warp: 48 regs（轻量，主要做地址计算和原子操作）
- MMA warp: 40 regs（仅发射指令，不需要大量状态）
- Epilogue warp: 208 regs（需要大量寄存器做 TMEM → convert → store）

### 3. 多级 Barrier 体系

```
full_barriers[kNumStages]        : TMA Load → MMA (smem 流水线)
empty_barriers[kNumStages]       : MMA → TMA Load (smem 回收)
tmem_full_barriers[kNumEpiStages] : UMMA → Epilogue (TMEM 就绪)
tmem_empty_barriers[kNumEpiStages]: Epilogue → UMMA (TMEM 回收)
```

Phase 翻转机制避免 barrier reset：
```cpp
stage_idx = (stage_idx + 1) % kNumStages;
phase ^= stage_idx == 0;  // 满一圈翻转
```

### 4. 2-CTA UMMA + TMA Multicast

- 2 个 CTA 共同计算一个 tile（cluster_size=2）
- A 矩阵 TMA multicast：一次 HBM 读取写入 2 个 CTA 的 smem
- 等效 HBM 读带宽翻倍（对 compute-bound 场景尤其有效）

## 文件结构

```
=== 核心实现 ===
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh   — 核心 kernel (~750行)
csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp              — JIT runtime + launch
csrc/jit_kernels/heuristics/gemm_rs.hpp                    — get_gemm_rs_config()
csrc/apis/gemm_rs.hpp                                      — C++ API: bf16_gemm_rs_nt()
deep_gemm/gemm_rs/__init__.py                              — Python API: bf16_gemm_rs_nt()

=== 测试 ===
tests/test_gemm_rs.py                                      — 正确性测试 (多shape, 2/4/8 GPU)
benchmarks/bench_gemm_rs.py                                — 对比: Fused vs GEMM+NCCL分离

=== 基础设施 ===
deep_gemm/include/deep_gemm/comm/barrier.cuh               — nvlink_barrier / grid_sync
deep_gemm/include/deep_gemm/layout/gemm_rs.cuh             — GemmRSWorkspace 布局
deep_gemm/include/deep_gemm/layout/sym_buffer.cuh          — SymmetricBuffer (NVLink 映射)
deep_gemm/include/deep_gemm/ptx/ld_st.cuh                  — PTX st_rel_sys / ld_acq_sys
```

## Python API

```python
import deep_gemm

# 创建对称缓冲区
sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
    group,                    # ProcessGroup
    num_max_tokens_per_rank,  # 最大 token 数
    hidden,                   # N 维度
    out_dtype=torch.bfloat16,
    comm_dtype=None,          # None=BF16, torch.float32=FP32
)

# GEMM+RS
deep_gemm.bf16_gemm_rs_nt(
    y,                    # [tokens_per_rank, N], output
    a,                    # [total_tokens, K], input (BF16)
    b,                    # [N, K], weight NT layout (BF16)
    sym_buffer,           # GemmRSSymmBuffer
    num_tokens_per_rank,  # 当前实际 token 数
    compiled_dims='nk',   # JIT 编译维度
)
```

## 运行测试

```bash
cd /workspace/codebuddy/DeepGEMM

# 清除 JIT 缓存
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# 正确性测试 (2 GPUs, basic shapes)
python tests/test_gemm_rs.py 2

# 正确性测试 (8 GPUs, all shapes including extended)
python tests/test_gemm_rs.py 8 --all

# 性能基准测试
python benchmarks/bench_gemm_rs.py 2 20
python benchmarks/bench_gemm_rs.py 8 30
```

> **注意**：不要用 `torchrun`，脚本内部用 `mp.spawn` 管理多进程。

## 调试指南

### 如果正确性不过

1. **检查 per-tile flag**: 加 `printf` 在 comm warps 看 flag 是否被正确设置
2. **检查 M-Swizzle**: 确认每个 rank 的 tile 计算顺序和 flag 索引对应
3. **检查 FP32 reduce**: 确认累加器初始化为 0，以及 rank 自己的 partial 也被加入
4. **单步验证**: 先固定 `num_ranks=2`，关闭 M-Swizzle 测试基本逻辑

### 如果 Hang（死锁）

1. **Comm Warps 永远等不到 flag**: 可能是 Epilogue 没有正确 `st_rel_sys`
2. **nvlink_barrier 卡住**: 可能某 rank 提前结束 → 检查所有 rank 是否执行相同数量的 barrier
3. **TMA Load 超时**: 远端内存地址错误 → 检查 `sym_buffer_ptrs` 的正确性

### 如果性能不达预期

1. **nsys profile**: `nsys profile python benchmarks/bench_gemm_rs.py 2 5`
2. **检查 SM 占用率**: 384 线程可能限制 occupancy → 考虑减少 Comm Warps
3. **检查 NVLink 利用率**: `nvidia-smi nvlink -i 0 -s` 查看 NVLink 吞吐
4. **Comm Warps 瓶颈**: 如果 pull 延迟高，考虑用 TMA Load 代替手动 global load

## 后续优化方向

### P1: TMA Load for Pull（高优先）
用 TMA 的硬件异步 bulk load 代替 Comm Warps 的手动 global load。
- 优势：TMA 引擎独立于 SM 运行，不占用 CUDA core 时间
- 实现：Comm Warps 只负责 poll flag + issue TMA + wait completion + reduce

### P2: Comm Warps Pipeline Depth
当前 Comm Warps 对每个 src_rank 是串行的（poll → load → reduce）。
优化：引入 2-stage pipeline：
- Stage 0: 正在 reduce rank_k 的数据
- Stage 1: 同时 poll + prefetch rank_{k+1} 的数据

### P3: Adaptive Warp Split
根据 compute/comm ratio 动态分配 Comm vs Epilogue warp 数量：
- Compute-heavy（大 K、小 M）：减少 Comm warps，增加 GEMM 吞吐
- Comm-heavy（大 M、小 K、多 rank）：增加 Comm warps，加快 pull 速度

### P4: FP8 版本
扩展到 FP8 输入（需要 SF 处理逻辑 + UTCCP）。

### P5: Multi-step Ring
参考 NCCL ring 的流水线，进一步减少延迟：
- 不是每个 rank 从所有 peer pull，而是 ring 传递 + 逐步 reduce
- 适用于 8+ GPU 场景
