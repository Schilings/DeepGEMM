# BF16 GEMM-RS 两阶段算子详解

> 面向 Blackwell (SM100) 架构的 BF16 GEMM + Reduce-Scatter 融合算子  
> 采用两阶段分离架构：GEMM+Push kernel + Reduce Epilogue kernel（通过 PDL 串联）

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [阶段1: GEMM + NVLink Push Kernel](#2-阶段1-gemm--nvlink-push-kernel)
3. [阶段2: Reduce Epilogue Kernel](#3-阶段2-reduce-epilogue-kernel)
4. [Host 端调度与 PDL 流水线](#4-host-端调度与-pdl-流水线)
5. [Persistent Kernel 调度机制](#5-persistent-kernel-调度机制)
6. [辅助基础设施](#6-辅助基础设施)
7. [完整源码](#7-完整源码)

---

## 1. 整体架构概览

### 1.1 问题背景

在 EP (Expert Parallelism) 场景中，每个 rank 持有完整输入 A (shape: `[M_total, K]`) 和不同 expert 的权重 B (shape: `[K, N]`)。GEMM 计算完成后，需要在 rank 间做 Reduce-Scatter：每个 rank 只保留输出 Y 的一部分行（按 M 维切分），同时将自己算出的属于其他 rank 的部分推送过去。

### 1.2 两阶段分离架构

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段1: GEMM + NVLink Push (sm100_bf16_gemm_rs_nt_impl)      │
│                                                              │
│  调度顺序: rank i 先计算发往 rank i+1 的 chunk → push       │
│           再计算 rank i+2 → push                            │
│           ...                                                │
│           最后计算自己 rank i 的 → 直接写本地 partial buf     │
│                                                              │
│  N 次计算掩盖 N-1 次通信                                     │
│  Epilogue: TMEM → smem → cp.async.bulk 写到远端 partial buf  │
│  完成后设置 ready flag (st_rel_sys)                          │
└──────────────────────────────────────────────────────────────┘
            │
            │ PDL (Programmatic Dependent Launch)
            ↓
┌──────────────────────────────────────────────────────────────┐
│ 阶段2: Reduce Epilogue (sm100_bf16_reduce_epilogue_impl)     │
│                                                              │
│  cudaGridDependencySynchronize() 等待阶段1完成               │
│  从 partial buffer 读取各 rank 的数据                        │
│  element-wise 累加 → 写 output                               │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 为什么分两个 Kernel？

| 方案 | 缺陷 |
|------|------|
| **单 kernel + RS Warp 内嵌** | GEMM 结束时 RS warp 自旋等待所有 tile 完成，浪费线程资源；epilogue 写远端与下一轮 MMA 无法流水 |
| **单 kernel + 末尾 reduce** | Reduce 阶段只有部分线程工作，其余线程空转 |
| **两阶段 PDL（我们的方案）** | ✅ GEMM 线程全部用于计算+通信，无空转；Reduce kernel 进入时数据已就绪，无需自旋；两阶段通过 PDL 重叠 |

### 1.4 数据布局：SymBuffer + GemmRSWorkspace

```
SymBuffer (对称缓冲区):
  每个_rank_持有同一块 NVLink 可达的内存区域
  通过 offsets[] 数组实现本地指针→远端指针的映射

GemmRSWorkspace 内存布局:
  ┌─────────────────────────────────┐ offset 0
  │ Barrier Signal (32 bytes)        │ ← 跨_rank_同步信号
  ├─────────────────────────────────┤ offset 32
  │ Partial Buffer                   │ ← 各_rank_计算的 GEMM 部分结果
  │   [rank_0][m][n]                 │    大小 = num_ranks * M * N * elem_size
  │   [rank_1][m][n]                 │
  │   ...                            │
  │   [rank_N-1][m][n]              │
  ├─────────────────────────────────┤
  │ Ready Flags                      │ ← 标记某个 (m_block, n_block) 的 partial 结果已就绪
  │   [rank_0][m_block][n_block]     │    大小 = num_ranks * num_m_blocks * num_n_blocks * 4
  │   [rank_1][m_block][n_block]     │
  │   ...                            │
  └─────────────────────────────────┘ 16-byte 对齐
```

---

## 2. 阶段1: GEMM + NVLink Push Kernel

### 2.1 Kernel 签名与模板参数

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,     // 128, 128, 64
          uint32_t kNumStages,                                        // 软件流水线级数
          uint32_t kNumNonEpilogueThreads,                            // 128 (Warp0+Warp1)
          uint32_t kNumEpilogueThreads,                               // 128 (Warp2+Warp3)
          uint32_t kNumSMs, uint32_t kNumRanks,                       // GPU SM 数, rank 数
          typename cd_dtype_t>                                         // BF16 或 FP32
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_nt_impl(
    const uint32_t shape_m_per_rank,      // 最大 M per rank（workspace 分配依据）
    const uint32_t runtime_m_per_rank,     // 实际 M per rank（本次计算的实际 token 数）
    const uint32_t shape_n,                // N 维度
    const uint32_t shape_k,                // K 维度
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,  // 对称缓冲区
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,        // A 矩阵 TMA 描述符
    const __grid_constant__ cute::TmaDescriptor tensor_map_b);       // B 矩阵 TMA 描述符
```

### 2.2 Warp 分工

```
┌──────────────────────────────────────────────────┐
│ 256 threads = 4 warps                            │
│                                                  │
│ Warp 0 (TMA Load Warp):  TMA 加载 A、B 到 smem   │
│ Warp 1 (MMA Issue Warp): 发射 UMMA FMA 指令      │
│ Warp 2 (Epilogue Warp 0): TMEM→smem→远端写入      │
│ Warp 3 (Epilogue Warp 1): TMEM→smem→远端写入      │
│                                                  │
│ 注意: BF16 路径没有 RS Warp                       │
│       (FP8 路径有 128 个 RS 线程，旧架构)          │
└──────────────────────────────────────────────────┘
```

### 2.3 初始化阶段

```cpp
// ── 常量定义 ──
constexpr uint32_t BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 64;
constexpr uint32_t UMMA_M = 128, UMMA_N = 128, UMMA_K = 16;
constexpr uint32_t STORE_BLOCK_M = 128;
constexpr uint32_t STORE_BLOCK_N = 128 / sizeof(cd_dtype_t);  // BF16: 64, FP32: 32
constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;       // 128 threads 做 epilogue
constexpr uint32_t kNumEpilogueStages = 2;  // TMEM 双缓冲
constexpr uint32_t kNumTMAStoreStages = 2;  // smem CD 双缓冲
```

**Shared Memory 布局：**

```
smem_buffer:
  ┌─────────────────────────────────┐
  │ smem_cd[0]                      │ STORE_BLOCK_M * STORE_BLOCK_N * elem_size
  │ smem_cd[1]                      │ (双缓冲，2 stages)
  ├─────────────────────────────────┤
  │ smem_a[0..kNumStages-1]         │ LOAD_BLOCK_M * BLOCK_K * 2 per stage
  ├─────────────────────────────────┤
  │ smem_b[0..kNumStages-1]         │ LOAD_BLOCK_N * BLOCK_K * 2 per stage
  ├─────────────────────────────────┤
  │ full_barriers[kNumStages]       │ TMA load 完成信号
  │ empty_barriers[kNumStages]      │ smem stage 空闲信号
  │ tmem_full_barriers[2]           │ TMEM 写入完成信号
  │ tmem_empty_barriers[2]          │ TMEM 空闲信号 (128 threads arrive)
  │ tmem_ptr                        │ TMEM 分配指针
  └─────────────────────────────────┘
```

### 2.4 跨 Rank 同步：Ready Flags 清零 + NVLink Barrier

```cpp
// 所有线程协作清零 ready flags
for (uint32_t i = sm_idx * kNumThreads + thread_idx;
     i < kNumRanks * workspace.get_num_m_blocks_per_rank() * workspace.get_num_n_blocks();
     i += kNumSMs * kNumThreads) {
    workspace.get_ready_ptr()[i] = 0;
}

// 跨 rank barrier: 确保所有 rank 都完成清零后才开始计算
comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, 41>(
    workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);
```

**NVLink Barrier 原理：**
1. 先做 grid sync（所有 SM 达成一致）
2. SM 0 的前 kNumRanks 个线程通过 `red_add_rel_sys` 向各 rank 发送信号
3. SM 0 的 thread 0 自旋等待信号计数 == kNumRanks
4. 再做一次 grid sync 通知所有 SM barrier 完成

### 2.5 Persistent Block 调度：Rank Wave 机制

这是 GEMM-RS 与普通 GEMM 的核心区别。普通 GEMM 的 tile 调度不关心数据属于哪个 rank，而 GEMM-RS 需要按照 rank 分组调度，以实现通信-计算重叠。

```cpp
auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, 
                          uint32_t& n_block_idx, uint32_t& iter_idx) {
    if (block_idx >= num_m_blocks * num_n_blocks)
        return false;
    
    // 将一维 block_idx 分解为 (rank_wave, local_m_block, n_block)
    const uint32_t m_rank_wave = block_idx / (num_m_blocks_per_rank * num_n_blocks);
    const uint32_t rem = block_idx - m_rank_wave * num_m_blocks_per_rank * num_n_blocks;
    const uint32_t local_m_block_idx = rem / num_n_blocks;
    n_block_idx = rem - local_m_block_idx * num_n_blocks;
    
    // Wave 0 → rank i+1, Wave 1 → rank i+2, ..., Wave N-1 → self
    const uint32_t dst_rank = (m_rank_wave + 1 < kNumRanks) ?
        (rank_idx + m_rank_wave + 1) % kNumRanks : rank_idx;
    m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;
    
    block_idx += kNumSMs;  // round-robin 步进
    ++ iter_idx;
    return true;
};
```

**调度示意图（4 rank 示例，rank 0 视角）：**

```
所有 M blocks 按 rank 分组：
  ┌──────────┬──────────┬──────────┬──────────┐
  │ rank=1   │ rank=2   │ rank=3   │ rank=0   │
  │ (push)   │ (push)   │ (push)   │ (local)  │
  └──────────┴──────────┴──────────┴──────────┘
   Wave 0     Wave 1     Wave 2     Wave 3

计算 rank=1 的 tile 时，epilogue 通过 NVLink push 到 rank 1
计算 rank=2 的 tile 时，epilogue 通过 NVLink push 到 rank 2
...
计算 rank=0 (self) 的 tile 时，直接写本地 partial buffer

关键: 先算远程 → 通信先启动；最后算本地 → 不需要通信
结果: N 次计算自然掩盖 N-1 次 NVLink 通信
```

**与普通 GEMM Scheduler 的对比：**

| 特征 | 普通 GEMM Scheduler | GEMM-RS get_next_block |
|------|-------------------|----------------------|
| 步进 | `(++iter) * kNumSMs + blockIdx.x` | `blockIdx.x + iter * kNumSMs` (等价) |
| tile 归属 | 直接映射 (m_block, n_block) | 额外映射到 dst_rank |
| 调度顺序 | 无特定顺序 | 按 rank wave 排序：远程优先 |
| Swizzle | 有 L2 cache swizzle | 暂无（后续可优化） |

### 2.6 Warp 0: TMA Load Pipeline

```cpp
if (warp_idx == 0 and cute::elect_one_sync()) {
    uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
    while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
        const uint32_t global_m = m_block_idx * BLOCK_M;
        const uint32_t n_idx = n_block_idx * BLOCK_N;
        const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
        
        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
            // 等待当前 stage 的 smem 空闲
            empty_barriers[stage_idx]->wait(phase ^ 1);
            
            // TMA 异步加载 A 和 B
            tma::copy(&tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, global_m, 1);
            tma::copy(&tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
            
            // 通知 barrier 期望收到多少字节
            full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
        }
    }
}
```

**流水线状态机：**
```
Stage 0 ──[empty]──▶ TMA Load A+B ──[full]──▶ MMA 消费 ──▶ [empty] ──▶ ...
Stage 1 ──[empty]──▶ TMA Load A+B ──[full]──▶ MMA 消费 ──▶ [empty] ──▶ ...
...
Stage N-1 ──[empty]──▶ ...

phase 位: 每次 stage_idx 归零时翻转，用于 barrier 的 wait/arrive 配对
```

### 2.7 Warp 1: MMA Issue (UMMA FMA)

```cpp
else if (warp_idx == 1 and is_leader_cta) {
    // 构建 UMMA 指令描述符和 smem 描述符
    auto instr_desc = cute::UMMA::make_instr_desc<...>();
    auto a_desc = make_umma_desc<...>(smem_a[0], 0, 0);
    auto b_desc = make_umma_desc<...>(smem_b[0], 0, 0);
    
    uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
    while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
        // 等待 TMEM 上一个 iteration 的 epilogue 完成
        auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
        tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
        
        const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
        for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
            // 等待 TMA load 完成
            full_barriers[stage_idx]->wait(phase);
            
            // 发射 UMMA FMA：BLOCK_K / UMMA_K = 64/16 = 4 次 FMA
            for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                           k_block_idx > 0 or k > 0, runtime_instr_desc);
            }
            
            // 每个 k_block 完成后 arrive empty barrier
            // 最后一个 k_block 同时 arrive tmem_full_barrier
            empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
        }
    }
}
```

**UMMA (Unified Matrix Multiply-Accumulate)** 是 Blackwell 架构的新指令：
- 操作数从 smem 读取，结果写入 TMEM (Tensor Memory)
- 累积器使用双缓冲 (`kNumEpilogueStages = 2`)，MMA 和 Epilogue 可流水

### 2.8 Warp 2-3: Epilogue — TMEM → smem → 远端 NVLink 写入

这是 GEMM-RS 最核心的 epilogue 逻辑：

```cpp
else if (warp_idx >= kNumNonEpilogueThreads / 32 and
         warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
    
    uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
    while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
        // 等待 MMA 完成
        tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
        
        const uint32_t dst_rank = m_block_idx / num_m_blocks_per_rank;
        const bool is_self_rank = (dst_rank == rank_idx);
        
        for (uint32_t w = 0; w < kNumMWaves; ++ w) {
            for (uint32_t s = 0; s < kNumStores; ++ s) {
                
                // ═══ Step 1: TMEM → smem (de-swizzle + BF16 转换) ═══
                cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                // 从 TMEM 读取 → BF16 转换 → 写入 smem (swizzle 布局)
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr, ...);
                cutlass::arch::fence_view_async_tmem_load();
                ptx::st_shared(smem_ptr, cast_into_bf16_and_pack(...));
                
                // 释放 TMEM stage 给下一个 MMA iteration
                if (最后一个 sub-tile) {
                    tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                }
                
                // ═══ Step 2: smem → 远端 partial buffer (cp.async.bulk) ═══
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    // 计算目标地址：
                    //   self rank → 直接用本地指针
                    //   远端 rank → SymBuffer.map() 转换为 NVLink 可达地址
                    auto* dst_partial_ptr = is_self_rank ?
                        workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m, n_idx) :
                        sym_buffer.map(
                            workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m, n_idx),
                            dst_rank);
                    
                    // cp.async.bulk: 非阻塞的 bulk store
                    ptx::tma_store_1d(dst_partial_ptr, smem_cd[tma_stage_idx],
                                      STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t));
                    cute::tma_store_arrive();
                }
            }
        }
        
        // ═══ Step 3: 设置 ready flag ═══
        if (epilogue_thread_idx == 0) {
            ptx::tma_store_wait();  // 等待所有 TMA store 完成
            
            // Release store: 保证所有 prior store 对远端可见
            ptx::st_rel_sys(remote_ready_ptr, 1u);
        }
    }
}
```

**关键设计决策：为什么用 `cp.async.bulk` 而不是 TMA Store 2D？**

TMA Store 2D (`SM90_TMA_STORE_2D::copy`) 需要一个 **Tensor Map Descriptor**，这个描述符在创建时编码了固定的基地址。但我们写远端时，目标地址是 **运行时动态计算** 的（不同 rank 的 NVLink 地址不同），无法用一个静态 tensor map 覆盖。

`cp.async.bulk` (`ptx::tma_store_1d`) 接受运行时的全局指针，完美适配跨 rank NVLink 写入：

```asm
// PTX: cp.async.bulk.global.shared::cta.bulk_group [dst_gmem], [src_smem], size, cache_hint
cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;
```

**内存序保证的三层防护：**

```
1. cute::tma_store_fence()    → 确保 smem 写入在 TMA store 发射前完成
2. ptx::tma_store_wait()      → 等待 cp.async.bulk 完成（数据已到达 gmem）
3. ptx::st_rel_sys()          → release 语义的 store，保证前面的所有 store 
                                 对 system scope（包括其他 GPU）可见后才写 flag
```

### 2.9 PDL 完成信号

```cpp
// kernel 末尾
__syncthreads();
if (warp_idx == 0)
    Allocator().free(0, kNumTmemCols);  // 释放 TMEM

cudaTriggerProgrammaticLaunchCompletion();  // 通知 PDL: 本 kernel 即将完成
```

---

## 3. 阶段2: Reduce Epilogue Kernel

### 3.1 Kernel 签名

```cpp
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads = 256,
          typename cd_dtype_t = cutlass::bfloat16_t>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_bf16_reduce_epilogue_impl(
    cd_dtype_t* __restrict__ output,
    const uint32_t runtime_m_per_rank,
    const uint32_t shape_n,
    const void* __restrict__ workspace_base,
    const uint32_t shape_m_per_rank);
```

### 3.2 PDL 等待机制

```cpp
// 进入 kernel 后第一步：等待前序 GEMM kernel 完成
cudaGridDependencySynchronize();
```

**PDL (Programmatic Dependent Launch)** 的工作原理：

```
时间线:
  ┌─────────────────────────────────────────────────┐
  │ GEMM+Push Kernel                                │
  │   ... 计算中 ...                                 │
  │   cudaTriggerProgrammaticLaunchCompletion() ←───── 通知即将完成
  │   ... 最后几个 tile ...                          │
  └─────────────────────────────────────────────────┘
                    ↕ 重叠
  ┌─────────────────────────────────────────────────┐
  │ Reduce Kernel                                   │
  │   cudaGridDependencySynchronize() ←─── 等待信号  │
  │   ... 开始 reduce ...                           │
  └─────────────────────────────────────────────────┘
  
  关键: GPU 硬件在 GEMM kernel 触发 Completion 后，
       就可以开始调度 Reduce kernel 的 CTA，
       而不必等 GEMM 的所有 CTA 都退出
```

### 3.3 向量化 Reduce

```cpp
constexpr uint32_t kVecSize = 8;  // 8 个 BF16 = 16 bytes = uint4
const uint32_t total_elements = runtime_m_per_rank * shape_n;
const uint32_t total_vecs = total_elements / kVecSize;

// 主循环：每个线程处理若干个 16-byte 向量
for (uint32_t vec_idx = global_thread_idx; vec_idx < total_vecs; vec_idx += total_threads) {
    const uint32_t elem_base = vec_idx * kVecSize;
    const uint32_t row = elem_base / shape_n;
    const uint32_t col = elem_base - row * shape_n;
    
    // FP32 累加器
    float acc[kVecSize] = {0};
    
    // 遍历各 rank 的 partial results
    for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
        const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col);
        const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
        uint4 data = *vec_ptr;  // 一次读 16 bytes
        const auto* bf16_data = reinterpret_cast<const cd_dtype_t*>(&data);
        for (uint32_t i = 0; i < kVecSize; ++ i)
            acc[i] += static_cast<float>(bf16_data[i]);
    }
    
    // FP32 → BF16 并写出
    uint4 result;
    auto* result_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
    for (uint32_t i = 0; i < kVecSize; ++ i)
        result_bf16[i] = cd_dtype_t(acc[i]);
    *reinterpret_cast<uint4*>(output + row * shape_n + col) = result;
}

// 处理尾部非对齐元素
const uint32_t remaining_start = total_vecs * kVecSize;
for (uint32_t elem_idx = remaining_start + global_thread_idx;
     elem_idx < total_elements; elem_idx += total_threads) {
    float acc = 0.0f;
    for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank)
        acc += static_cast<float>(*workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col));
    output[row * shape_n + col] = cd_dtype_t(acc);
}
```

**Reduce 的数据流：**

```
partial_buffer[rank_0][row][col]  ──┐
partial_buffer[rank_1][row][col]  ──┤ FP32 累加
partial_buffer[rank_2][row][col]  ──┤ ────────→ BF16 → output[row][col]
...                                 │
partial_buffer[rank_N-1][row][col] ─┘
```

**为什么不需要查 Ready Flag？**

因为 PDL 保证 `cudaGridDependencySynchronize()` 返回时，GEMM kernel 已经完成。所有 partial buffer 数据都已写入并可见（`st_rel_sys` + PDL 的隐式 memory fence）。

---

## 4. Host 端调度与 PDL 流水线

### 4.1 配置 (GemmRSConfig)

```cpp
struct GemmRSConfig {
    int block_m = 128, block_n = 128;
    int block_k = 64;  // BF16: 128/2 = 64
    int num_stages;     // 由 smem 容量决定，通常 2-8
    int num_rs_threads = 0;    // BF16: 0 (无 RS warps)
    int num_non_epilogue_threads = 128;  // Warp 0 + Warp 1
    int num_epilogue_threads = 128;      // Warp 2 + Warp 3
    int reduce_num_threads = 256;        // Reduce kernel 线程数
};
```

### 4.2 两阶段 Launch 流程

```cpp
static void sm100_bf16_gemm_rs_nt(...) {
    const auto num_sms = device_runtime->get_num_sms();
    auto config = get_gemm_rs_config(m, n, k, num_sms, 2 /* BF16 */);
    
    // 创建 TMA 描述符
    const auto tensor_map_a = make_tma_2d_desc(a, ...);
    const auto tensor_map_b = make_tma_2d_desc(b, ...);
    
    // ── 阶段1: GEMM + Push ──
    // Grid = num_sms, Block = 256 threads
    SM100BF16GemmRSRuntime::Args gemm_args = {
        .launch_args = LaunchArgs(num_sms, 256, smem_size, 1)
    };
    SM100BF16GemmRSRuntime::launch(gemm_runtime, gemm_args);
    
    // ── 阶段2: Reduce Epilogue (PDL) ──
    // Grid = num_sms, Block = 256 threads, smem = 0
    SM100BF16ReduceEpilogueRuntime::Args reduce_args = {
        .launch_args = LaunchArgs(num_sms, 256, 0, 1)
    };
    // LaunchRuntime 自动设置 cudaLaunchAttributeProgrammaticStreamSerialization
    SM100BF16ReduceEpilogueRuntime::launch(reduce_runtime, reduce_args);
}
```

**PDL 属性设置（在 LaunchRuntime 基类中自动完成）：**

```cpp
cudaLaunchAttribute attr;
attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr.val.programmaticStreamSerializationAllowed = 1;
cudaLaunchConfigSetAttribute(config, &attr);
```

---

## 5. Persistent Kernel 调度机制

### 5.1 什么是 Persistent Kernel？

传统 GEMM kernel 的 launch 模式：
```
Grid 大小 = 总 tile 数
每个 CTA 只处理 1 个 tile，然后退出
```

Persistent kernel 的 launch 模式：
```
Grid 大小 = SM 数量（远小于总 tile 数）
每个 CTA 在循环中处理多个 tile，直到所有 tile 耗尽才退出
```

**为什么 Persistent Kernel 更好？**

1. **减少 launch overhead**: 一次 kernel launch 完成所有计算
2. **负载均衡**: tile 动态分配，快的 CTA 多做，慢的 CTA 少做
3. **缓解长尾问题**: 传统模式下，最后一个 tile 可能让少量 CTA 空转等待；persistent 模式下所有 CTA 同时结束
4. **更好的 cache 利用**: CTA 可以在 smem/L2 中重用跨 tile 的数据

### 5.2 DeepGEMM 中的 Persistent 调度

#### 5.2.1 普通 GEMM Scheduler

```cpp
// 文件: scheduler/gemm.cuh
struct Scheduler {
    int current_iter = -1;
    
    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;
        if (next_block_idx >= num_blocks)
            return false;
        get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        return true;
    }
};

// Kernel 内:
while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
    // 处理 tile (m_block_idx, n_block_idx)
}
```

**调度图示 (4 SM, 12 tiles):**
```
SM 0: tile 0 → tile 4 → tile 8 → exit
SM 1: tile 1 → tile 5 → tile 9 → exit
SM 2: tile 2 → tile 6 → tile 10 → exit
SM 3: tile 3 → tile 7 → tile 11 → exit

步进公式: next = (++iter) * 4 + blockIdx.x
  iter=0: SM0→0, SM1→1, SM2→2, SM3→3
  iter=1: SM0→4, SM1→5, SM2→6, SM3→7
  iter=2: SM0→8, SM1→9, SM2→10, SM3→11
  iter=3: next=12+SM >= 12, 全部退出
```

**L2 Cache Swizzle:**

Scheduler 还包含 swizzle 逻辑，将连续的 tile 映射到不连续的 M/N block，让相邻 SM 访问的 gmem 地址交错，提升 L2 cache 命中率：

```
无 Swizzle:          有 Swizzle:
SM0: (0,0)          SM0: (0,0)
SM1: (0,1)          SM1: (0,8)
SM2: (0,2)          SM2: (0,1)
SM3: (0,3)          SM3: (0,9)
...                  ...
(连续访问，L2 冲突)   (交错访问，L2 友好)
```

#### 5.2.2 GEMM-RS Scheduler (get_next_block)

```cpp
// GEMM-RS 的调度器嵌入在 kernel 中，不是独立类
auto get_next_block = [&](uint32_t& block_idx, ...) {
    if (block_idx >= num_m_blocks * num_n_blocks) return false;
    
    // 将一维索引分解为 (rank_wave, local_m, n)
    const uint32_t m_rank_wave = block_idx / (num_m_blocks_per_rank * num_n_blocks);
    ...
    // 远程 rank 优先，本地 rank 最后
    const uint32_t dst_rank = (m_rank_wave + 1 < kNumRanks) ?
        (rank_idx + m_rank_wave + 1) % kNumRanks : rank_idx;
    
    block_idx += kNumSMs;  // 同样的 round-robin 步进
    return true;
};

while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
    // 处理 tile，epilogue 写到 dst_rank
}
```

#### 5.2.3 MegaMoE Scheduler

MegaMoE 有更复杂的两层调度 (L1/L2 wave)，但核心思想相同：

```cpp
// 文件: scheduler/mega_moe.cuh
template <typename Func>
CUTLASS_DEVICE void for_each_block(Func&& func) {
    fetch_expert_recv_count();  // 从 workspace 读取每个 expert 的 token 数
    set_expert_idx(0);
    
    while (true) {
        auto [block_phase, expert_idx, m_block_idx, n_block_idx] = get_next_block();
        if (block_phase == BlockPhase::None) break;
        func(block_phase, expert_idx, k_blocks, m_block_idx, n_block_idx);
    }
}
```

**MegaMoE 的 L1/L2 Wave 机制：**

```
L1 Wave: 处理每个 expert 的前 L1_SHAPE_N 列的 tile
         (较窄的 N 维度，适合 expert 间负载不均的场景)

L2 Wave: 处理每个 expert 的后 L2_SHAPE_N 列的 tile
         (较宽的 N 维度，适合大 token 数的 expert)

调度顺序 (8 expert, kNumExpertsPerWave=4):
  L1 Wave 0: expert 0,1,2,3 的 L1 tiles
  L2 Wave 0: expert 0,1,2,3 的 L2 tiles
  L1 Wave 1: expert 4,5,6,7 的 L1 tiles
  L2 Wave 1: expert 4,5,6,7 的 L2 tiles

同一个 wave 内的 expert 共享 A 矩阵的 TMA load，减少重复加载
```

### 5.3 Hopper/Blackwell 架构的动态调度改进

从 Hopper (SM90) 开始，GPU 硬件对 persistent kernel 的调度更加动态：

#### 5.3.1 传统调度的问题：长尾效应

```
传统 tile 静态分配 (非 persistent):
  Grid = num_tiles, 每个线程块处理 1 个 tile
  
  时间: ──────────────────────────────────────▶
  SM0:  [tile 0][tile 4][tile 8]
  SM1:  [tile 1][tile 5]          ← 提前完成，空转
  SM2:  [tile 2][tile 6][tile 10][长尾 tile]
  SM3:  [tile 3][tile 7][tile 11]
                               ↑
                          长尾: SM2 的 tile 特别慢，
                          其他 SM 空转等待 kernel 结束
```

#### 5.3.2 Persistent 调度如何缓解长尾

```
Persistent 调度:
  Grid = num_sms, 每个 CTA 循环取 tile
  
  时间: ──────────────────────────────────────▶
  SM0:  [tile 0][tile 4][tile 8] [tile 12]
  SM1:  [tile 1][tile 5][tile 9] [tile 13] ← 快的 SM 多做
  SM2:  [tile 2][tile 6][tile 10]
  SM3:  [tile 3][tile 7][tile 11]
                               ↑
                          没有 SM 空转！快的 SM 自动
                          领取更多 tile，慢的少领
```

#### 5.3.3 Blackwell 的硬件级增强

1. **PDL (Programmatic Dependent Launch)**: 允许后续 kernel 在前序 kernel 完成前就开始调度 CTA，减少 kernel 间间隙

2. **Cluster 级调度**: 2-CTA cluster 允许同 cluster 的 CTA 共享 smem 和同步，提升数据局部性

3. **TMEM (Tensor Memory)**: Blackwell 新增的 on-SM tensor memory，作为 UMMA 的累积器，减少 smem 压力，使 persistent 循环更高效

4. **cp.async.bulk**: 非阻塞的 bulk 传输，epilogue 的远端写入不阻塞下一轮计算

5. **Dynamic SM allocation**: GPU 调度器可以根据 CTA 的实际进度动态分配 SM 资源，而不是严格按照 launch 顺序

```
Blackwell 上的完整流水线:

时间 ─────────────────────────────────────────────────▶

SM0: [GEMM tile 0][GEMM tile 4]...[GEMM tile N-4] ─┐
SM1: [GEMM tile 1][GEMM tile 5]...[GEMM tile N-3]   │ PDL
SM2: [GEMM tile 2][GEMM tile 6]...[GEMM tile N-2]   │ 重叠
SM3: [GEMM tile 3][GEMM tile 7]...[GEMM tile N-1] ─┘
                      ↕
SM0:                    [Reduce tile 0][Reduce tile 4]...
SM1:                    [Reduce tile 1][Reduce tile 5]...
SM2:                    [Reduce tile 2][Reduce tile 6]...
SM3:                    [Reduce tile 3][Reduce tile 7]...

GEMM 最后几个 tile 和 Reduce 最初几个 tile 在时间上重叠
```

### 5.4 三种 Scheduler 的对比

| 特征 | 普通 GEMM | GEMM-RS | MegaMoE |
|------|----------|---------|---------|
| Grid 大小 | num_sms | num_sms | num_sms |
| 调度步进 | `iter * kNumSMs + blockIdx.x` | `blockIdx.x + iter * kNumSMs` | `blockIdx.x + iter * kNumSMs` |
| 额外维度 | 无 | rank wave (远程优先) | expert + L1/L2 wave |
| Swizzle | 有 | 暂无 | 暂无 |
| 跨 rank | 否 | 是 (NVLink push) | 是 (NVLink dispatch/reduce) |
| 注释 | "Persistently schedule" | 内嵌 lambda | "Persistently schedule" |

---

## 6. 辅助基础设施

### 6.1 SymBuffer: 对称缓冲区地址映射

```cpp
template <uint32_t kNumRanks>
struct SymBuffer {
    int64_t base;                    // 本 rank 的缓冲区基地址
    int64_t offsets[kNumMaxRanks];   // 各 rank 的偏移量
    uint32_t rank_idx;               // 本 rank 编号
    
    // 将本地指针映射到目标 rank 的 NVLink 地址
    template <typename ptr_t>
    CUTLASS_DEVICE ptr_t map(const ptr_t& ptr, const uint32_t& dst_rank_idx) const {
        int64_t mapped_ptr = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return *reinterpret_cast<ptr_t*>(&mapped_ptr);
    }
};
```

**原理**: 所有 rank 的 SymBuffer 在 NVLink 互联的同一地址空间中，`offsets[i]` = rank i 的基地址 - 本 rank的基地址。因此 `本地指针 + offset[i]` = rank i 中对应位置的 NVLink 地址。

### 6.2 GemmRSWorkspace: Workspace 内存管理

```cpp
struct GemmRSWorkspace {
    void* base;
    uint32_t num_ranks, num_max_tokens_per_rank, hidden, elem_size;
    uint32_t block_m = 128, block_n = 128;
    
    // Partial buffer: 存放各 rank 的 GEMM 部分结果
    template <typename dtype_t>
    CUTLASS_DEVICE dtype_t* get_partial_ptr(
        const uint32_t& slot_idx,    // rank 编号
        const uint32_t& token_idx,   // M 维偏移
        const uint32_t& hidden_idx)  // N 维偏移
    const;
    
    // Ready flags: 标记某个 tile 的 partial 结果已就绪
    CUTLASS_DEVICE uint32_t* get_ready_ptr(
        const uint32_t& slot_idx,      // rank 编号
        const uint32_t& m_block_idx,   // M 方向 block 编号
        const uint32_t& n_block_idx);  // N 方向 block 编号
};
```

### 6.3 PTX 指令封装

```cpp
// Release store (system scope) — 替代 __threadfence_system + volatile write
CUTLASS_DEVICE void st_rel_sys(uint32_t* ptr, uint32_t val) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(ptr), "r"(val) : "memory");
}

// cp.async.bulk store — 非阻塞的 smem→gmem bulk 传输
CUTLASS_DEVICE void tma_store_1d(const void* dst_ptr, const void* src_ptr, 
                                  const uint32_t& num_bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr), "r"(cvta_shared(src_ptr)), "r"(num_bytes), "l"(hint) : "memory");
}

// 等待 cp.async.bulk 完成
CUTLASS_DEVICE void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
```

---

## 7. 完整源码

### 7.1 Kernel 实现 (sm100_bf16_gemm_rs.cuh)

```cpp
#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// ============================================================================================
//  sm100_bf16_gemm_rs_nt_impl —— BF16 GEMM + Reduce-Scatter (Push Only)
// ============================================================================================
//
//  【设计思想】
//
//  两阶段分离架构：
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段1: GEMM + NVLink Push (本 kernel)                       │
//  │                                                              │
//  │  调度顺序: rank i 先计算发往 rank i+1 的 chunk → push       │
//  │           再计算 rank i+2 → push                            │
//  │           ...                                                │
//  │           最后计算自己 rank i 的 → 直接写本地 partial buf     │
//  │                                                              │
//  │  N 次计算掩盖 N-1 次通信                                     │
//  │  Epilogue: TMEM → smem → TMA store 到远端 partial buffer    │
//  │  完成后设置 ready flag (st_rel_sys)                          │
//  └──────────────────────────────────────────────────────────────┘
//             │
//             │ PDL (Programmatic Dependent Launch)
//             ↓
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段2: Reduce Epilogue (独立 kernel)                         │
//  │                                                              │
//  │  cudaGridDependencySynchronize() 等待阶段1完成               │
//  │  从 partial buffer 读取各 rank 的数据                        │
//  │  element-wise 累加 → 写 output                               │
//  └──────────────────────────────────────────────────────────────┘
//
//  【优势】
//
//  相比 RS Warp 内嵌方案：
//  1. GEMM kernel 线程全部用于计算和通信，无空转
//  2. Reduce kernel 不需要自旋等待，进入时数据已就绪
//  3. TMA 异步写远端，epilogue 不阻塞下一轮 MMA 发射
//  4. 两个 kernel 间通过 PDL 重叠，reduce 可在 GEMM 即将结束时开始
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_nt_impl(const uint32_t shape_m_per_rank,
                           const uint32_t runtime_m_per_rank,
                           const uint32_t shape_n,
                           const uint32_t shape_k,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;
    using ab_dtype_t = cutlass::bfloat16_t;

    // ── 常量定义 ──
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64,
                     "The BF16 GEMM+RS version expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128,
                     "Invalid GEMM thread layout");

    constexpr uint32_t UMMA_M = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // ── 运行时变量 ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(cd_dtype_t), BLOCK_M, BLOCK_N);

    // ── Prefetch TMA descriptors ──
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }

    // ── Shared memory layout ──
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2);

    // ── Initialize barriers ──
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // ── Clean ready flags (cross-rank sync) ──
    for (uint32_t i = sm_idx * kNumThreads + thread_idx;
         i < kNumRanks * workspace.get_num_m_blocks_per_rank() * workspace.get_num_n_blocks();
         i += kNumSMs * kNumThreads) {
        auto* ready_base = workspace.get_ready_ptr();
        ready_base[i] = 0;
    }
    constexpr uint32_t kAfterReadyCleanBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kAfterReadyCleanBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // ── Pipeline state ──
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ── Block scheduling: rotate through ranks for load-balanced communication ──
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        const uint32_t m_rank_wave = block_idx / (num_m_blocks_per_rank * num_n_blocks);
        const uint32_t rem = block_idx - m_rank_wave * num_m_blocks_per_rank * num_n_blocks;
        const uint32_t local_m_block_idx = rem / num_n_blocks;
        n_block_idx = rem - local_m_block_idx * num_n_blocks;
        const uint32_t dst_rank = (m_rank_wave + 1 < kNumRanks) ?
            (rank_idx + m_rank_wave + 1) % kNumRanks : rank_idx;
        m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;
        block_idx += kNumSMs;
        ++ iter_idx;
        return true;
    };

    // ════════════════════════════════════════════════════════════════
    //  Warp 0 (TMA Load Warp): Load A + B tiles into shared memory
    // ════════════════════════════════════════════════════════════════
    if (warp_idx == 0 and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, global_m, 1);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 1 (MMA Issue Warp): Execute UMMA FMA → TMEM accumulator
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == 1 and is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto a_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();
                using mma_t = ptx::SM100_MMA_F16BF16_SS;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        a_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                   k_block_idx > 0 or k > 0, runtime_instr_desc);
                    }
                }
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 2~3 (Epilogue Warps): TMEM → smem → TMA store to remote
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx >= kNumNonEpilogueThreads / 32 and
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() { tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages; };
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const uint32_t dst_rank = m_block_idx / num_m_blocks_per_rank;
            const uint32_t local_m_block_idx = m_block_idx - dst_rank * num_m_blocks_per_rank;
            const uint32_t local_m = local_m_block_idx * BLOCK_M;
            const bool is_self_rank = (dst_rank == rank_idx);

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                    // ── Step 1: TMEM → smem (de-swizzle + BF16 conversion) ──
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    const uint32_t n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                        epilogue_warp_idx * 32 * kSwizzleCDMode +
                                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                        uint32_t values[kNumElemsPerBankGroup];
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                        } else {
                            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::st_shared(smem_ptr,
                                           cast_into_bf16_and_pack(values[0], values[1]),
                                           cast_into_bf16_and_pack(values[2], values[3]),
                                           cast_into_bf16_and_pack(values[4], values[5]),
                                           cast_into_bf16_and_pack(values[6], values[7]));
                        }
                    }
                    // Release TMEM stage for next MMA iteration
                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // ── Step 2: smem → remote partial buffer (cp.async.bulk store async) ──
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        auto* dst_partial_ptr = is_self_rank ?
                            workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m + w * WAVE_BLOCK_M, n_idx) :
                            sym_buffer.map(
                                workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m + w * WAVE_BLOCK_M, n_idx),
                                dst_rank);

                        ptx::tma_store_1d(
                            dst_partial_ptr,
                            smem_cd[tma_stage_idx],
                            STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }

            // ── Step 3: Set ready flag after all waves of this block are stored ──
            if (epilogue_thread_idx == 0) {
                ptx::tma_store_wait();
                auto* remote_ready_ptr = is_self_rank ?
                    workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx) :
                    sym_buffer.map(
                        workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx),
                        dst_rank);
                ptx::st_rel_sys(remote_ready_ptr, 1u);
            }
        }
    }

    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    // ── PDL: 通知后续 reduce kernel 本 GEMM kernel 即将完成 ──
    cudaTriggerProgrammaticLaunchCompletion();

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

// ============================================================================================
//  sm100_bf16_reduce_epilogue_impl —— Reduce Epilogue (PDL 依赖启动)
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads = 256,
          typename cd_dtype_t = cutlass::bfloat16_t>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_bf16_reduce_epilogue_impl(cd_dtype_t* __restrict__ output,
                                const uint32_t runtime_m_per_rank,
                                const uint32_t shape_n,
                                const void* __restrict__ workspace_base,
                                const uint32_t shape_m_per_rank) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)

    // ── 等待前序 GEMM kernel 完成 ──
    cudaGridDependencySynchronize();

    const auto workspace = layout::GemmRSWorkspace(
        const_cast<void*>(workspace_base), kNumRanks, shape_m_per_rank, shape_n, sizeof(cd_dtype_t), BLOCK_M, BLOCK_N);

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_thread_idx = sm_idx * kNumThreads + thread_idx;
    const uint32_t total_threads = kNumSMs * kNumThreads;

    // ── 向量化 reduce ──
    constexpr uint32_t kVecSize = 8;
    const uint32_t total_elements = runtime_m_per_rank * shape_n;
    const uint32_t total_vecs = total_elements / kVecSize;

    for (uint32_t vec_idx = global_thread_idx; vec_idx < total_vecs; vec_idx += total_threads) {
        const uint32_t elem_base = vec_idx * kVecSize;
        const uint32_t row = elem_base / shape_n;
        const uint32_t col = elem_base - row * shape_n;

        float acc[kVecSize];
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i)
            acc[i] = 0.0f;

        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col);
            const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
            uint4 data = *vec_ptr;
            const auto* bf16_data = reinterpret_cast<const cd_dtype_t*>(&data);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i) {
                acc[i] += static_cast<float>(bf16_data[i]);
            }
        }

        uint4 result;
        auto* result_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i) {
            result_bf16[i] = cd_dtype_t(acc[i]);
        }
        auto* out_vec_ptr = reinterpret_cast<uint4*>(output + row * shape_n + col);
        *out_vec_ptr = result;
    }

    // ── 处理剩余元素（非对齐的尾部） ──
    const uint32_t remaining_start = total_vecs * kVecSize;
    for (uint32_t elem_idx = remaining_start + global_thread_idx;
         elem_idx < total_elements;
         elem_idx += total_threads) {
        const uint32_t row = elem_idx / shape_n;
        const uint32_t col = elem_idx - row * shape_n;
        if (row >= runtime_m_per_rank) continue;

        float acc = 0.0f;
        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col);
            acc += static_cast<float>(*partial_ptr);
        }
        output[row * shape_n + col] = cd_dtype_t(acc);
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
```

### 7.2 Host 端 Launch 代码 (sm100_bf16_gemm_rs.hpp)

```cpp
#pragma once

#include <torch/python.h>
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include "../heuristics/gemm_rs.hpp"

namespace deep_gemm {

// 阶段1: GEMM + NVLink Push kernel runtime
class SM100BF16GemmRSRuntime final : public LaunchRuntime<SM100BF16GemmRSRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int m, n, k;
        int num_ranks;
        at::ScalarType y_dtype;
        GemmRSConfig config;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_rs_nt_impl<
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.y_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t shape_m_per_rank = static_cast<uint32_t>(args.max_m_per_rank);
        uint32_t runtime_m_per_rank = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            shape_m_per_rank, runtime_m_per_rank, shape_n, shape_k,
            args.sym_buffer_ptrs, args.tensor_map_a, args.tensor_map_b));
    }
};

// 阶段2: Reduce Epilogue kernel runtime
class SM100BF16ReduceEpilogueRuntime final : public LaunchRuntime<SM100BF16ReduceEpilogueRuntime> {
public:
    struct Args {
        int runtime_m_per_rank;
        int n;
        int num_ranks;
        int max_m_per_rank;
        int num_sms;
        at::ScalarType y_dtype;
        GemmRSConfig config;
        void* y;
        void* workspace_base;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_reduce_epilogue_impl<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", args.config.block_m, args.config.block_n,
    args.launch_args.grid_dim.first, args.num_ranks,
    args.config.reduce_num_threads,
    to_string(args.y_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        void* output_ptr = args.y;
        uint32_t runtime_m = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        void* workspace = args.workspace_base;
        uint32_t max_m = static_cast<uint32_t>(args.max_m_per_rank);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            output_ptr, runtime_m, shape_n, workspace, max_m));
    }
};

// 统一入口
static void sm100_bf16_gemm_rs_nt(const torch::Tensor& y,
                                  const torch::Tensor& a,
                                  const torch::Tensor& b,
                                  const torch::Tensor& sym_buffer,
                                  const std::vector<int64_t>& sym_buffer_ptrs,
                                  const int& rank_idx,
                                  const int& max_m_per_rank,
                                  const int& runtime_m_per_rank,
                                  const int& n,
                                  const int& k,
                                  const std::string& compiled_dims) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_config(m, n, k, num_sms, static_cast<int>(a.element_size()));
    DG_HOST_ASSERT(config.block_k == 64);

    // TMA 描述符
    const auto tensor_map_a = make_tma_2d_desc(a, k, m, config.block_k, config.load_block_m,
                                                static_cast<int>(a.stride(-2)), config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, n, config.block_k, config.load_block_n,
                                                static_cast<int>(b.stride(-2)), config.swizzle_b_mode);

    // 阶段1
    const SM100BF16GemmRSRuntime::Args gemm_args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .m = m, .n = n, .k = k,
        .num_ranks = num_ranks,
        .y_dtype = y.scalar_type(),
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .launch_args = LaunchArgs(num_sms, 256, config.smem_size, 1)
    };
    const auto gemm_code = SM100BF16GemmRSRuntime::generate(gemm_args);
    const auto gemm_runtime = compiler->build("sm100_bf16_gemm_rs_nt", gemm_code);
    SM100BF16GemmRSRuntime::launch(gemm_runtime, gemm_args);

    // 阶段2 (PDL)
    void* workspace_base = sym_buffer.data_ptr();
    const SM100BF16ReduceEpilogueRuntime::Args reduce_args = {
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n, .num_ranks = num_ranks,
        .max_m_per_rank = max_m_per_rank,
        .num_sms = num_sms,
        .y_dtype = y.scalar_type(),
        .config = config,
        .y = y.data_ptr(),
        .workspace_base = workspace_base,
        .launch_args = LaunchArgs(num_sms, 256, 0, 1)
    };
    const auto reduce_code = SM100BF16ReduceEpilogueRuntime::generate(reduce_args);
    const auto reduce_runtime = compiler->build("sm100_bf16_reduce_epilogue", reduce_code);
    SM100BF16ReduceEpilogueRuntime::launch(reduce_runtime, reduce_args);
}

} // namespace deep_gemm
```

### 7.3 Heuristics 配置 (gemm_rs.hpp)

```cpp
#pragma once
#include <iostream>
#include <unordered_set>
#include "sm100.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

struct GemmRSConfig {
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode;
    int num_stages, smem_size;
    int num_rs_threads;  // FP8: 128, BF16: 0
    int num_non_epilogue_threads, num_epilogue_threads;
    int num_multicast;
    int reduce_num_threads;  // BF16: 256
};

static GemmRSConfig get_gemm_rs_config(const int& m, const int& n, const int& k,
                                        const int& num_sms, const int& elem_size_ab = 1) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;  // BF16: 64, FP8: 128

    constexpr int load_block_m = block_m;
    constexpr int load_block_n = block_n;
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    const int num_rs_threads = (elem_size_ab == 2) ? 0 : 128;  // BF16 无 RS warps
    constexpr int num_non_epilogue_threads = 128;
    constexpr int num_epilogue_threads = 128;
    constexpr int num_multicast = 1;
    constexpr int num_tma_store_stages = 2;
    constexpr int num_epilogue_stages = 2;
    constexpr int reduce_num_threads = 256;

    const int smem_cd = 128 * swizzle_cd_mode * num_tma_store_stages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;
    const int smem_sfa_per_stage = 128 * 4;
    const int smem_sfb_per_stage = 128 * 4;
    const int smem_barriers = 32 * 8 * 3 + num_epilogue_stages * 8 * 2 + 8;
    const int smem_tmem_ptr = 4;
    const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    const int num_stages = std::min((SM100ArchSpec::smem_capacity - smem_extra) / smem_per_stage, 8);
    DG_HOST_ASSERT(num_stages >= 2);

    return GemmRSConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_extra + num_stages * smem_per_stage,
        num_rs_threads,
        num_non_epilogue_threads, num_epilogue_threads,
        num_multicast,
        reduce_num_threads
    };
}

} // namespace deep_gemm
```

---

## 附录: 术语表

| 术语 | 含义 |
|------|------|
| **UMMA** | Unified Matrix Multiply-Accumulate, Blackwell 架构的新 MMA 指令，操作数在 smem，结果在 TMEM |
| **TMEM** | Tensor Memory, Blackwell 新增的 on-SM 存储单元，作为 UMMA 的累积器 |
| **TMA** | Tensor Memory Accelerator, 异步的 tensor 维度加载/存储引擎 |
| **cp.async.bulk** | 非阻塞的 bulk 内存拷贝指令（smem↔gmem），支持 NVLink 跨 GPU 传输 |
| **PDL** | Programmatic Dependent Launch, 允许后续 kernel 在前序 kernel 完成前开始调度的硬件机制 |
| **SymBuffer** | 对称缓冲区，通过 offsets 数组实现本地指针到远端 NVLink 地址的映射 |
| **Persistent Kernel** | Grid 大小 = SM 数量，每个 CTA 循环处理多个 tile 的 kernel 执行模式 |
| **st_rel_sys** | Release 语义的 system-scope store，保证前面的所有 store 对其他 GPU 可见 |
| **Ready Flag** | 标记某个 tile 的 partial GEMM 结果已写入 workspace 的标志位 |
| **Rank Wave** | GEMM-RS 的调度策略：按目标 rank 分波，先算远程 rank 的 tile 以尽早启动通信 |
