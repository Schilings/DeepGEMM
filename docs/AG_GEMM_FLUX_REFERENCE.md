# Flux AG GEMM 设计参考（BF16 + SM90 + 单机）

> **来源**：Flux 项目 `/root/.local/codebuddy/flux`
> **分析范围**：BF16 精度、SM90 架构、单机内（nnodes=1）
> **目标**：为 DeepGEMM SM100 AG GEMM 提供设计参考

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                AllGatherGemmOp::forward()                     │
│                                                              │
│  输入: input[M/tp, K]  (每个 rank 本地分片)                    │
│        weight[N, K]   (每个 rank 相同)                        │
│  输出: output[M, N]   (每个 rank 完整输出)                     │
│                                                              │
│  Step 1: ag_op.run(input)         ← Copy Stream              │
│          └─ CE DMA: cudaMemcpyAsync 拷贝各 rank chunk         │
│             CUStreamWriteValue 写 barrier flag                │
│                                                              │
│  Step 2: gemm_op.forward(...)     ← Compute Stream           │
│          └─ GEMM kernel 逐 tile wait barrier → 计算          │
└──────────────────────────────────────────────────────────────┘
```

### 关键文件（Flux 项目路径）

| 文件 | 作用 |
|------|------|
| `src/ag_gemm/ths_op/all_gather_gemm_op.cc` | 顶层编排：`forward_default_impl` 双流调度 |
| `src/coll/ths_op/all_gather_op.cc` | `copy_all_to_all()` — CE DMA 循环 + `CUStreamWriteValue` |
| `src/ag_gemm/sm90_all_gather_gemm_tma_warpspecialized_pingpong.hpp` | SM90 GEMM kernel：warpgroup barrier wait + WGMMA |
| `src/ag_gemm/sm90_all_gather_gemm_tile_scheduler.hpp` | AG swizzle：`get_current_work()` 中 M_idx 重映射 |
| `src/ag_gemm/gemm_v3_ag_kernel.hpp` | `GemmV3AGKernel` — 绑定 AG scheduler + barrier 参数到 GEMM |
| `src/ag_gemm/ths_op/gemm_with_barrier.cc` | `GemmWithBarirer` — 构造 kernel args，含 `barrier_buffer` |
| `include/flux/args/ag_gemm.h` | `AGKernelArguments` 结构体 |
| `src/ag_gemm/all_gather_swizzle.hpp` | 跨节点 swizzle 常量（单机时 nnodes=1 不涉及） |

---

## 二、通信部分：CE DMA AllGather

### 2.1 通信模式

- BF16 默认使用 **CE DMA**（非 CUDA core kernel）
- NVSwitch/NVLink 全互联 → **All2All 模式**
- 通信在**独立的高优先级 Copy Stream** 上执行

### 2.2 数据布局

```
每个 rank 的对称 Buffer 布局:

input_buffer[M, K]: 完整 M×K 的本地接收 buffer
  ├── [0 : M/tp]           ← 本地 rank 自己的数据（先 copy 到位）
  ├── [M/tp : 2*M/tp]      ← rank 1 的数据（CE DMA 拷贝）
  ├── [2*M/tp : 3*M/tp]    ← rank 2 的数据
  └── ...

barrier_buffer[world_size]: barrier flag 数组
  ├── barrier[0]           ← rank 0 数据 chunk 已就绪?
  ├── barrier[1]           ← rank 1 数据 chunk 已就绪?
  └── ...
```

### 2.3 CE DMA 拷贝流程（All2All 模式）

```
Copy Stream 上:
  1. cudaMemsetAsync(barrier_buffer, 0)          // 清零所有 flag
  2. cudaMemcpyAsync(local_chunk, local_input)     // 本地数据先拷贝
  3. CUStreamWriteValue(barrier[self_rank], 1)    // 标记本 rank chunk 就绪

  4. for each remote rank r:
       cudaMemcpyAsync(
         dst = input_buffer[r * M/tp : (r+1) * M/tp],
         src = remote_rank_r.input_buffer[self_rank 区域],
         size = (M/tp) * K * sizeof(bf16)
       )
       CUStreamWriteValue(barrier[r], 1)           // CE 写 barrier flag
```

**核心**：`CUStreamWriteValue` 是 CE（Copy Engine）指令，MEM 拷贝完成后 CE 硬件直接写 flag = 1，**不经过 CUDA core，不占用 SM 算力**。

---

## 三、计算部分：Barrier 驱动的 GEMM

### 3.1 Kernel 架构（SM90 Pingpong）

```
每个 ThreadBlock = 3 WarpGroups:

  WarpGroup 0 (Producer, 1 warp):  TMA load A/B → smem → pipeline
  WarpGroup 1 (Consumer0, 4 warps): WGMMA tile N → epilogue store
  WarpGroup 2 (Consumer1, 4 warps): WGMMA tile N+1 → epilogue store
                                      (与 Consumer0 交替，数学并行)
```

### 3.2 Barrier 等待逻辑

GEMM kernel 启动后，每个 TB 在计算 tile 前：

```
1. 获取 tile 的 M 坐标：tile_m = get_tile_offset(tile_idx).m()

2. 计算该 tile 的数据依赖:
   data_chunk_id = tile_m * TILE_SIZE_M / (M / world_size)
   即：tile 的第一行属于哪个 rank 的 data_chunk

3. 等待数据就绪：
   if warp_group_role == Consumer0:
     Consumer0SystemBarrier::wait_eq(barrier[data_chunk_id], 1)
   elif warp_group_role == Consumer1:
     Consumer1SystemBarrier::wait_eq(barrier[data_chunk_id], 1)
   else:
     ProducerSystemBarrier::wait_eq(barrier[data_chunk_id], 1)
   __syncthreads()

4. 数据就绪后执行：
   Producer: TMA load A → smem
   Consumer: WGMMA → accumulator → epilogue store
```

**关键**：Pingpong kernel 中 Producer、Consumer0、Consumer1 使用**三个独立的 SystemBarrier slot**，避免等待时的 warpgoup 间竞争。

### 3.3 进入下一个 Tile 时的增量等待

```
while 还有 tile 需要处理:
  计算当前 tile → epilogue store
  scheduler.advance_to_next_work()
  新 tile 的 M 坐标 → 新 data_chunk_id

  if 新 data_chunk_id != 当前 data_chunk_id:
    等待新 chunk 的 barrier flag
  继续计算新 tile
```

**优化**：如果下一个 tile 与当前 tile 属于同一个 data_chunk，则**无需重新等待** barrier。

---

## 四、并行性的三层来源

### 4.1 硬件级：CE 与 SM 异构并行

```
时间 →

CE (Copy Engine):
  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
  │cpy rank0 ││cpy rank1 ││cpy rank2 ││cpy rank3 │  (4-way TP)
  │flag[0]=1 ││flag[1]=1 ││flag[2]=1 ││flag[3]=1 │
  └──────────┘└──────────┘└──────────┘└──────────┘

SM (GEMM Kernel):
       ┌──────────────────────────────────────────┐
       │TB_0:  wait flag[0] ──► WGMMA tile0       │
       │TB_1:  wait flag[0] ──► WGMMA tile1       │
       │TB_2:  wait flag[1] ───────► WGMMA tile2  │  ← CE 拷 rank1 的同时
       │TB_3:  wait flag[1] ───────► WGMMA tile3  │     SM 已开始算 rank0 区块
       │TB_4:  wait flag[2] ──────────────► ...    │
       └──────────────────────────────────────────┘
```

CE 和 SM 是不同的硬件单元，可以真正并行执行。

### 4.2 粒度级：per-chunk barrier

- 不等全矩阵 AG 完成再启动 GEMM
- 每完成一个 data_chunk 的 CE DMA 拷贝，对应的 TB 就立即开始计算
- Barrier 粒度 = `M / world_size` 行

### 4.3 调度级：Tile Swizzle（Local-First）

AG tile scheduler 在 `get_current_work()` 中重排 tile 顺序：

```cpp
// 单机 (nnodes=1):
new_M_idx = (M_idx + problem_blocks_m_offset) % problem_blocks_m;

// 其中 problem_blocks_m_offset = M_start / TILE_SIZE_M
// M_start = (M / world_size) * rank
```

**效果**：
- rank 0 的 TB 优先调度 M[0 : M/tp] 区域的 tile
- rank 1 的 TB 优先调度 M[M/tp : 2M/tp] 区域的 tile
- 这些 tile 对应的 data_chunk **最先就绪**（本地数据 + 最早从本地发出的 CE DMA）
- → 最小化 TB 的 barrier 等待时间

---

## 五、与 NCCL AllGather + GEMM 的对比

| 维度 | 分离式 AG + GEMM | Flux AG GEMM |
|------|-----------------|--------------|
| 通信方式 | NCCL AllGather（可能走 ring） | CE DMA All2All |
| 同步粒度 | 全矩阵 AG 完成后才 GEMM | per-chunk barrier |
| 硬件并行 | AG 期间 SM 空闲 | CE + SM 同时工作 |
| 内存 | AG 单独分配临时 buffer | 使用预分配的对称 buffer |
| 启动开销 | 两次 kernel launch | 一次通信编排 + 一次 GEMM launch |

---

## 六、与 DeepGEMM SM100 的差异要点

Flux SM90 的 BF16 AG GEMM 实现可作为 DeepGEMM SM100 的设计参考，注意以下差异：

| 维度 | Flux SM90 | DeepGEMM SM100 |
|------|-----------|----------------|
| 通信后端 | CUDA CE DMA (`cudaMemcpyAsync`) | 需要适配 SM100 的 PE DMA |
| Barrier 同步 | `SystemBarrier` / `WarpGroupSystemBarrier` | SM100 可能有不同 barrier ISA |
| MMA 指令 | WGMMA (SM90) | UMMA (SM100) |
| 数据搬运 | TMA load (SM90) | TMA load (SM100 兼容) |
| Cluster | SM90 cluster (可选 2 CTA/cluster) | SM100 2CTA cluster |
| PDL | 支持（CUDA core AG 路径） | SM100 可能有等价机制 |

当前 DeepGEMM 的 AG GEMM 迭代（参见 `AG_GEMM_ITERATION.md`）已按 Flux 风格完成骨架改造：
- 通信已拆到独立 comm stream（CE DMA + `CUStreamWriteValue`）
- Kernel 已具备 per-chunk barrier 等待
- 调度已改为 local-first M swizzle

后续优化方向可参考 Flux：
1. 更细粒度 chunk（自适应 M_per_rank / BLOCK_M 确定 chunk 数量）
2. 类似 Pingpong 的双 consumer warpgroup 交替执行
3. 评估 SM100 上是否也需要区分 Producer/Consumer0/Consumer1 barrier slot
