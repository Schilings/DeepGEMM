# Flux GEMM+RS 架构深度分析与 DeepGEMM 借鉴指南

> ⚠️ 历史研究文档：用于架构参考，不代表当前代码状态。
> 当前真实进度请以 `docs/PROGRESS.md` 为准。
>
> 📌 落地更新（2026-06-18）：本文档第 5 节的「双 kernel」方向已落地为主线，
> 并进一步采用**真·Flux pull 式**（GEMM epilogue 纯本地 scatter write + 独立 RS reduce
> kernel 从远端 pull），而非方案 C 的 host CE DMA。实现细节见 `GEMM_RS_DESIGN.md`，
> 迭代记录见 `GEMM_RS_ITERATION.md`(Iteration 3)。注意通信通道选型与本文 5.3 推荐不同：
> 单机 NVLink 下直接用 `sym_buffer.map` P2P 直读(pull)，以获得 tile 级细粒度 overlap。

> 研究日期：2026-06-15
> 目标：分析 ByteDance Flux (SM90 Hopper) GEMM+RS 架构原理，提取可借鉴到 DeepGEMM (SM100 Blackwell) 的设计思想

---

## 1. Flux GEMM+RS 整体架构

### 1.1 双 Kernel 方案（核心设计）

Flux 采用**双 kernel 分离**架构：

```
Kernel 1: GEMM Kernel (compute-only)
  - 标准 CUTLASS SM90 Cooperative GEMM
  - Epilogue 改为 scatter write + set per-tile flag

Kernel 2: RS Reduce Kernel (communication-only)
  - 独立 kernel，在不同 stream 上运行
  - Per-tile poll ready flag → TMA load from remote → reduce → write output
  - 与 GEMM kernel 通过 per-tile flag 天然实现 tile 级流水线 overlap
```

**与 DeepGEMM 单 kernel 的根本区别**：

| 维度 | Flux (双kernel) | DeepGEMM (单kernel) |
|------|-----------------|---------------------|
| Kernel 数量 | 2 (GEMM + RS) | 1 (融合) |
| GEMM 线程数 | 256T (标准cooperative) | 384T (含128T comm) |
| GEMM 寄存器 | ~252/thread (无spilling) | ~168/thread (严重spilling!) |
| GEMM 吞吐 | ~1100 TFLOPS | ~600 TFLOPS |
| Overlap 粒度 | Stream 级 (kernel间) | Warp 级 (warp specialization) |
| 通信线程 | 独立 kernel (可配置) | 128T in-kernel comm warps |

### 1.2 为什么双 kernel 更好？

**根因：384T launch_bounds 导致 register spilling**

```
SM100 每SM寄存器 = 65536 (SM100 规范)
384T x 168 regs = 64512 → 占用 98.5%
384T x 252 regs = 96768 → 超出！所以编译器只能给 168 regs
168 regs 不够 UMMA 所需 → register spilling → GEMM 吞吐暴跌

256T x 252 regs = 64512 → 刚好够，与标准 GEMM 一致
```

DeepGEMM 的 128T Comm Warps 在同一个 CTA 内占了 4 个 warp slots 但不贡献计算，
还挤占了 GEMM 的寄存器预算 → GEMM 吞吐量仅为标准 GEMM 的 ~55%。

---

## 2. Flux Kernel 1: GEMM Kernel 详解

### 2.1 Warp 角色 (SM90 Cooperative, 256T)

```
Producer Warp Group (128T = 4 warps):
  W0: Mainloop Load (TMA A+B)      — 标准 GEMM load
  W1: Epilogue Load (TMA C)        — 加载累加器输出
  W2: RS Fetch (TMA from remote)   — 从远端拉 partial 数据 [NEW]
  W3: RS Reduce (smem → gmem)      — reduce + 写输出 [NEW]

Consumer Warp Group 0 (128T = 4 warps):
  W4-W7: MMA (cooperative UMMA)   — 标准 cooperative MMA

Consumer Warp Group 1 (128T = 4 warps):
  W8-W11: (2SM cooperative 备用)
```

**关键洞察**：Flux 把 RS 的 Fetch 和 Reduce 放在 **Producer Warp Group** 中，
与 GEMM 的 Load warp 共享同一个 warp group。这在 SM90 的 warp-specialized 架构中是自然的——
Producer 负责「数据搬入」，RS Fetch 也是「数据搬入」。

### 2.2 Epilogue: Scatter Write + Flag

Flux 的 Epilogue 使用 `Sm90AuxStoreReduceScatter`：

```c++
// Epilogue 完成后设置 per-tile flag
CUTLASS_DEVICE void end() {
    auto [m, n, _] = tile_coord_mnl;
    int tile_idx = tile_layout(m, n);
    int flag_idx = tile_idx * 2;
    tma_store_wait<0>();
    // 使用 GenericSystemBarrier wait_eq_reset 设置 flag = 1
    Barrier::wait_eq_reset(params_ptr->barrier_ptr, thread_idx, flag_idx, 0, 1);
}
```

**Flag 结构** (`PerTileFlags`, 128B 对齐)：

```c++
struct PerTileFlags {
    int epilogue;             // GEMM epilogue 完成标志
    int padding_epilogue[8];  // 对齐填充
    int reduce;               // reduce 完成标志（ring 传递用）
    int padding_reduce[8];
    int reduce_sub_node;      // 跨节点 reduce 标志
    int padding_reduce_sub_node[8];
    int epilogue_queue;       // barrier queue 模式用
    int reduce_queue;
    int extra;                // 额外用途
};
```

### 2.3 Epilogue Scatter Write

GEMM 的 Epilogue 不是写回原始输出矩阵，而是 **scatter write**：
- 根据 tile 的 M 坐标确定 `dst_rank`
- 写到 `output_scatter_ptrs[dst_rank]` 对应的位置
- 写完后 set `epilogue` flag

---

## 3. Flux Kernel 2: Sm90ReduceScatterDma 详解

这是 Flux 最核心的创新——**单 warp (32T) 的 TMA 流水线 RS DMA 引擎**。

### 3.1 架构

```
Sm90ReduceScatterDma<Stages, TileShape, EpilogueTile, SmemLayoutAtom, Element, StrideMNL, CommKind, FuseReduction>

线程数: 32 (1 warp)
流水线: FetchPipeline (PipelineTransactionAsync<Stages>)
同步: NamedBarrier (FluxNamedBarriers::ReduceScatterFetch / ReduceScatterReduce)
```

### 3.2 Fetch 阶段 (Producer)

```c++
CUTLASS_DEVICE auto fetch(FetchPipeline fetch_pipeline, PipelineState fetch_write_state,
                          ProblemShapeMNKL problem_shape, TileCoordMNKL tile_coord) {
    // 1. 计算源 rank 和 M 偏移
    int src_rank = m / params_ptr->tile_m_perrank;
    int local_src_rank = src_rank % params_ptr->local_world_size;
    int m_fetch = m + (params_ptr->local_rank - local_src_rank) * params_ptr->tile_m_perrank;

    // 2. 等待 GEMM epilogue 完成标志 (system-scope barrier)
    Barrier::wait_eq_reset(params_ptr->local_barrier_ptr[local_src_rank],
                           thread_idx, fetch_tile_idx * 2, 1);

    // 3. TMA Load: 从远端 rank 的 scatter buffer 拉数据到 smem
    //    按 EpilogueTile 粒度循环，每个 epi_m x epi_n 块一次 TMA 操作
    for (epi_n = 0; epi_n < ...; ++epi_n) {
        for (epi_m = 0; epi_m < ...; ++epi_m) {
            // TMA copy: gmem → smem
            copy(tma_load.with(*tma_barrier), bGS_gFetch, bGS_sFetch);
            fetch_pipeline.producer_expect_transaction(fetch_write_state);
            ++fetch_write_state;
        }
    }
    return fetch_write_state;
}
```

### 3.3 Reduce 阶段 (Consumer)

```c++
CUTLASS_DEVICE auto reduce(FetchPipeline fetch_pipeline, PipelineState fetch_read_state,
                           ProblemShapeMNKL problem_shape, TileCoordMNKL tile_coord) {
    // 1. 计算目标 rank
    int dst_rank = m / params_ptr->tile_m_perrank;
    int local_dst_rank = dst_rank % params_ptr->local_world_size;

    // 2. 如果启用 FuseReduction，等待本地 rank 先 reduce
    if constexpr (FuseReduction) {
        if (not is_local_tile_reduce) {
            Barrier::wait_lt(lock_ptr, thread_idx, flag_idx, 1);
        }
    }

    // 3. 消费 fetch pipeline 中的数据: smem → register → reduce → gmem
    for (epi_n = 0; epi_n < ...; ++epi_n) {
        for (epi_m = 0; epi_m < ...; ++epi_m) {
            fetch_pipeline.consumer_wait(fetch_read_state, barrier_token);
            // smem → register
            copy(tiled_copy, tsReduce_epi, trReduce);
            // reduce to gmem
            if constexpr (FuseReduction) {
                if (is_local_tile_reduce) {
                    copy(tiled_copy, trReduce, tgReduce_epi);  // 首个 rank: 直接写入
                } else {
                    cutlass::arch::local_red<VecType>(trReduce, tgReduce_epi.data(), true);
                }
            } else {
                copy(tiled_copy, trReduce, tgReduce_epi);
            }
            fetch_pipeline.consumer_release(fetch_read_state);
            ++fetch_read_state;
        }
    }

    // 4. FuseReduction: 到达 barrier，最后一个 rank 重置 flag
    if constexpr (FuseReduction) {
        int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);
        if (reduce_count == params_ptr->local_world_size) {
            Barrier::wait_eq_reset(lock_ptr, thread_idx, flag_idx, params_ptr->local_world_size, 0);
        }
    }
}
```

### 3.4 关键设计思想

1. **TMA 流水线**：Fetch 和 Reduce 解耦为 producer-consumer，
   Fetch 拉下一块数据到 smem 的同时 Reduce 正在处理上一块
2. **Per-tile poll**：不等全部 rank 完成，某个 rank 的 tile ready 就 fetch
3. **FuseReduction 模式**：多个 local rank 的数据 reduce 到同一个 reduce_buffer，
   用 barrier 计数跟踪完成进度
4. **32T 极简线程**：只 1 个 warp，寄存器压力极低，SM 占用极小

---

## 4. Flux Ring2d Pull/Push RS Kernel 详解

### 4.1 ReduceScatterRing2dPull（Pull 模式）

```
线程组织: kNumWorkersPerGroup = 256 (8 warps) + 1 waiter warp = 288T/group
多 group: blockDim.x / 288T groups per CTA, x gridDim.x = 总 group 数

流程:
for s in 0..kNumaWorldSize:      // 遍历 NUMA 拓扑步
  rrank = topo.rank_from[s][rank]  // 当前步的远端 rank

  for sid in {NextNodeRank(segment), segment}:  // 跨节点 + 本节点段
    for bid in gid..num_tiles step num_groups:  // 并行处理 tiles
      // 1. Wait: 等 GEMM epilogue 或上一步的 reduce flag
      if is_local: wait_eq_dev(flags(rank).epilogue_ptr(tile_idx))
      if is_prev_inter: wait_eq_sys(flags(rrank).reduce_ptr(reduce_tile_idx))

      // 2. Copy/Reduce: 从远端拉数据到本地并 reduce
      ldata(i) = add<T>(&ldata(i), &rdata(i))   // BF16 __hadd2

      // 3. Set ready: waiter warp 设置 reduce flag
      set_ready_dev(flags(rank).reduce_ptr(reduce_tile_idx))
```

### 4.2 ReduceScatterRing2dPushGemmk（Push 模式）

```
线程组织: kNumWorkersPerGroup = 128 (4 warps) + 1 waiter warp = 160T/group

流程 (Ring Push):
for s in 0..kNumaWorldSize:      // 遍历拓扑步
  to_rank = topo.rank_to[s][rank]   // 推送目标
  from_rank = topo.rank_from[s][rank]  // 上游来源

  for sid in {NextNodeRank(segment), segment}:
    for bid in gid..num_tiles step num_groups:
      // 1. Wait: 等上游的 reduce flag + 本地 epilogue flag
      if !is_ring_start: wait_eq_sys(flags(from_rank).reduce_ptr(tile_idx))
      wait_eq_dev(flags(rank).epilogue_ptr(tile_idx))

      // 2. Copy/Reduce: 本地数据 + 上游 reduce buffer → 下游 reduce buffer
      if is_ring_start: rdata = ldata         // 第一步：直接 copy
      else: rdata = add(ldata, local_reduce)   // 后续步：reduce + forward

      // 3. Set ready: 推送到下游
      set_ready(flags(rank).reduce_ptr(tile_idx))
```

**Push vs Pull 的选择**：
- **Pull**: 每个 rank 主动从所有 peer 拉数据并 reduce → 简单但通信量 O(N-1) per rank
- **Push**: Ring 模式逐步 reduce + forward → 通信量 O(1) per step，但延迟高（串行依赖）

---

## 5. DeepGEMM 借鉴方案：双 Kernel 重构

### 5.1 当前瓶颈（为什么需要重构）

```
DeepGEMM GEMM+RS iter23 性能:
- 8 GPU geo_mean: 1.040x (vs GEMM+NCCL RS 分离)
- GEMM 吞吐: ~600 TFLOPS (B300 峰值 ~1400T)
- 标准 GEMM 吞吐: ~1100 TFLOPS
- 核心瓶颈: 384T launch_bounds → 168 regs/thread → register spilling
```

### 5.2 双 Kernel 方案对比

#### 方案 A: 同 AG GEMM 的 Host-side Comm + Compute-only Kernel

```
Stream 0 (compute_stream):
  GEMM kernel (标准 256T cooperative) → Epilogue scatter write + set flag

Stream 1 (comm_stream):
  Host-side cudaMemcpyAsync (CE DMA) + RS Reduce kernel (128T/256T)
```

**优势**：GEMM kernel 完全等同于标准 bf16_gemm_nt；Host-side CE DMA 不占 SM
**劣势**：CE DMA 是 bulk copy，overlap 粒度较粗

#### 方案 B: Flux 风格的 GEMM + RS DMA Kernel

```
Stream 0 (compute_stream):
  GEMM kernel (标准 256T) → Epilogue scatter write + set flag

Stream 1 (comm_stream):
  RS DMA kernel (Flux-style Sm90ReduceScatterDma)
    Per-tile poll flag → TMA fetch → reduce → write output
```

**优势**：Tile 级 overlap；TMA 硬件异步
**劣势**：SM100 TMA 行为需验证；实现复杂

#### 方案 C: AG GEMM 风格 + 轻量 Reduce Kernel

```
Stream 0 (compute_stream):
  GEMM kernel (标准 256T) → Epilogue scatter write + set flag

Stream 1 (comm_stream):
  Host-side CE DMA 搬运 partial + Reduce kernel (简单 256T vectorized reduce)
```

**优势**：最简单，复用 AG GEMM 成熟模式；CE DMA 高效
**劣势**：CE DMA chunk 级 overlap；3 次 kernel launch（大 shape 可忽略）

### 5.3 推荐路线：方案 C 先行，方案 B 升级

1. **方案 C 两天可完成**：复用 AG GEMM 的 host-side comm 框架
2. **方案 B 需要更多验证**：SM100 TMA 用于 NVLink P2P read 行为未验证
3. **方案 C 足够验证核心假设**：256T GEMM 性能是否真的恢复到 ~1100T

### 5.4 方案 C 实现计划

#### Phase 1: GEMM Compute-only Kernel

基于 `sm100_bf16_gemm.cuh` 修改 Epilogue：

```c++
int dst_rank = m_block_idx / num_m_blocks_per_rank;
if (dst_rank == rank_idx) {
    // 本地 tile: 写 output
    tma_2d_store(output + local_offset, ...);
} else {
    // 远端 tile: 写 partial buffer + set flag
    T* dst = sym_buffer.map(local_partial_ptr, dst_rank);
    tma_1d_store(dst + tile_offset, ...);
    st_rel_sys(&ready_flag[dst_rank][m_block][n_block], 1);
}
```

#### Phase 2: Host-side Comm + Reduce

```python
def bf16_gemm_rs_nt(y, a, b, sym_buffer, num_tokens_per_rank):
    # 1. Launch GEMM on compute_stream
    torch._C._bf16_gemm_rs_compute(y, a, b, sym_buffer, ...)
    # 2. Host-side CE DMA: copy partial from remote ranks
    for chunk_idx in range(num_chunks):
        for src_rank in range(num_ranks):
            if src_rank != rank_idx:
                cudaMemcpyAsync(local_buf, remote_partial[rank_idx][chunk],
                                size, cudaMemcpyDefault, comm_stream)
    # 3. Launch Reduce kernel on comm_stream
    torch._C._rs_reduce(y, sym_buffer, ...)
```

#### Phase 3: RS Reduce Kernel

```c++
__global__ void __launch_bounds__(256, 4)
rs_reduce_kernel(bf16_t* output, bf16_t** partial_ptrs, int num_ranks,
                 int rank_idx, int m_per_rank, int n_dim) {
    for (int i = tid + blockIdx.x * blockDim.x; i < m_per_rank * n_dim; i += grid.x * blockDim.x) {
        bf16x2 acc = partial_ptrs[rank_idx][i/2];
        for (int r = 0; r < num_ranks; r++) {
            if (r != rank_idx) acc = __hadd2(acc, partial_ptrs[r][i/2]);
        }
        output[i/2] = acc;
    }
}
```

### 5.5 预期性能提升

| 场景 | 当前 (单kernel) | 双kernel (预期) | 改善 |
|------|----------------|-----------------|------|
| GEMM 吞吐 | ~600 TFLOPS | ~1100 TFLOPS | +83% |
| K=4096 shapes | 1.07-1.16x | **1.3-1.5x** | GEMM 恢复 |
| K=7168 shapes | 0.98-1.04x | **1.1-1.3x** | 翻赢 |
| K=2048 shapes | 0.78-0.95x | **0.85-1.0x** | 仍受限 |
| Geo Mean (8 GPU) | 1.040x | **1.15-1.25x** | |

---

## 6. Flux 其他可借鉴的技术细节

### 6.1 Flag/Barrier 体系

Flux 的 PerTileFlags (128B 对齐) 支持 SoA + AoS 双模式：
- **SoA**: 所有 tile 的 epilogue flag 连续存储 → cache 友好的 polling
- **AoS**: 每个 tile 的所有 flag 挨着 → 方便一次操作

DeepGEMM 可升级为 PerTileFlags 结构以支持更丰富的同步语义。

### 6.2 向量化 Reduce (BF16 __hadd2)

Flux 的 `add<T>()` 以 uint4 (16B) 为单位做 BF16 __hadd2：
- 16B = 8 x BF16 = 4 x __nv_bfloat162
- 4 次 __hadd2 处理 16B 数据
- DeepGEMM iter20 已实现类似逻辑（32B wide vector）

### 6.3 Ring2d 拓扑

Flux 支持 NUMA 感知的 2D Ring 拓扑：
```
Node 0: GPU 0 → 1 → 2 → 3
         ↓              ↑
Node 1: GPU 4 ← 5 ← 6 ← 7
```
DeepGEMM 当前单节点 8 GPU 无需此逻辑。

### 6.4 nanosleep 初始等待

```c++
nanosleep(params.sleep_ns);  // first wave: ~100us
```
避免 RS kernel 在 GEMM 还没产出 tile 时 busy polling。DeepGEMM 可借鉴。

---

## 7. AG GEMM 重构经验对照

### 7.1 重构路径

```
旧: in-kernel NVLink ring-push (4 Comm Warps) → 384T, register spilling
新: host-side CE DMA + compute-only kernel → 128T compute, mc2: 0.976x → 1.135x
```

### 7.2 AG GEMM → GEMM RS 的迁移差异

| 维度 | AG GEMM (AllGather) | GEMM RS (ReduceScatter) |
|------|---------------------|------------------------|
| 通信方向 | 多→一 (gather) | 一→多 (scatter+reduce) |
| 通信内容 | 拉取输入 A 的远程 chunk | 推送输出 C 的 partial 到远端 |
| Reduce | 无 (纯 copy) | 有 (N-1 rank partial 累加) |
| Epilogue | 标准 TMA store | scatter write + flag |
| Reduce kernel | 不需要 | 需要 |

**关键差异**：GEMM RS 多了 Reduce 步骤。

---

## 8. 总结与下一步

### 核心结论

1. **双 kernel 是正确方向**：单 kernel 的 register spilling 是结构性瓶颈
2. **方案 C (host-side CE DMA + reduce kernel) 最可行**：复用 AG GEMM 成熟模式
3. **预期性能提升显著**：GEMM 吞吐 +83%，geo_mean 1.04x → 1.15-1.25x
4. **Flux 的 TMA RS DMA 可作为 P1 升级方向**：更细粒度 overlap

### 实施步骤

1. **Phase 1**: 创建 `sm100_bf16_gemm_rs_compute.cuh` (从标准 GEMM 改 epilogue)
2. **Phase 2**: 创建 `sm100_rs_reduce.cuh` (简单 256T vectorized reduce)
3. **Phase 3**: 修改 JIT runtime + host-side comm 编排
4. **Phase 4**: 正确性测试 + 性能 benchmark
5. **Phase 5**: 根据结果决定是否升级到 Flux-style TMA RS DMA
