# GEMM+RS v3: Dual-Kernel Architecture Design

## 动机

当前单kernel方案 (iter23, geo_mean 1.040x) 已到微优化天花板。核心瓶颈：

**384T `launch_bounds` → 每线程仅 168 寄存器 → register spilling → GEMM 吞吐量仅为标准 GEMM 的 ~60%**

即使降到 288T（32T comm），GEMM 仍然比标准 256T kernel 慢 30-40%，因为 Comm warps 占用了 SM 资源（warp slots、寄存器、调度带宽）但不贡献计算。

双kernel方案的核心思路：**让 GEMM kernel 回归标准配置（128T/256T, 无 register spilling），通信由独立 kernel 完成，两个 kernel 在不同 CUDA stream 上通过 per-tile flag 实现流水线 overlap。**

## Flux 双kernel参考

Flux (ByteDance, SM90) 的 GEMM+RS 就是双kernel方案：

1. **Kernel 1: GEMM** — 标准 CUTLASS GEMM，Epilogue 改为 **scatter write**（按 `dst_rank` 写到不同 rank 的 partial buffer），写完一个 tile 就 set per-tile ready flag
2. **Kernel 2: RS DMA** — 独立 kernel，per-tile poll ready flag → TMA load from remote partial buffer → smem → reduce → write output

Flux 的 `Sm90ReduceScatterDma` 就是最关键的第二 kernel：
- 32 threads（1 warp）per CTA
- TMA load 从远端 rank 的 partial buffer 拉数据到 smem
- smem → registers → reduce (FP32/BF16 __hadd2)
- reduce 结果写入 reduce_buffer 或直接写 output
- 支持 FuseReduction（多个 remote rank 的数据累加到 reduce_buffer）

## 双kernel架构设计

### 总体流程

```
Stream 0 (compute_stream):
  [GEMM Kernel] ─── Epilogue: scatter write to partial buffer + set ready flag per tile

Stream 1 (comm_stream):
  [RS Kernel]   ─── poll ready flag → TMA load from remote → reduce → write output

Overlap: GEMM 计算后续 tile 的同时，RS kernel 在处理前面已 ready 的 tile
```

### Kernel 1: GEMM-only Kernel (Compute)

**目标**：最大化 GEMM 吞吐量，接近标准 `sm100_bf16_gemm` 的性能。

**关键改动 vs 当前单kernel**：

| 维度 | 当前 (单kernel) | 双kernel (GEMM) |
|------|----------------|-----------------|
| 总线程 | 384T (12 warps) | **256T (8 warps)** 或 **128T** |
| Comm Warps | 128T (4 warps) | **0** (移除) |
| 寄存器/线程 | ~168 (spilling!) | **~252** (与标准 GEMM 一致) |
| Epilogue | scatter write + push to remote | **scatter write + set flag** (相同) |
| GEMM 吞吐 | ~600 TFLOPS | **~1100 TFLOPS** (预期) |

**Warp 布局 (256T = 标准 2SM GEMM)**：

```
W0: TMA Load A+B (elect_one)      — 32T, 40 regs
W1: MMA Issue (is_leader_cta)     — 32T, 40 regs  
W2: Reserved / TMEM Allocator     — 32T, 40 regs
W3: Reserved                      — 32T, 40 regs
W4-W7: Epilogue Warps             — 128T, 208 regs
```

**Epilogue 行为**：
- 计算 tile 属于哪个 `dst_rank`（根据 `m_block_idx / num_m_blocks_per_rank`）
- `dst_rank == rank_idx`：直接写 output（TMA 2D store / per-row TMA 1D）
- `dst_rank != rank_idx`：写远端 partial buffer（`sym_buffer.map(local_ptr, dst_rank)`）+ `st_rel_sys` set ready flag
- 与当前单kernel的 Epilogue **完全相同**，只是少了 Comm Warps

**输出**：
- `output[local_rows, :]`：本地 chunk 的直接输出
- `partial_buffer[rank_idx][global_row, global_col]`：推送到远端的 partial（通过 NVLink P2P write）
- `ready_flag[rank_idx][m_block, n_block]`：per-tile ready flag（推送到远端）

### Kernel 2: RS Reduce Kernel (Communication)

**目标**：高效地将所有 rank 的 partial 结果 reduce 到本地 output。

**设计参考**：Flux `Sm90ReduceScatterDma` + `ring_reduce.cu`

**两种实现策略**：

#### Strategy A: 简单 Reduce Kernel（先实现）

```c++
// 每个 CTA 处理一个 tile 的 reduce
// 256 threads/CTA, grid = (num_tiles, 1, 1)
__global__ void __launch_bounds__(256, 4)
rs_reduce_kernel(
    comm_dtype_t* output,           // [m_per_rank, N]
    const comm_dtype_t** partial_ptrs, // 指向每个 rank 的 partial buffer
    int* ready_flags,               // [num_ranks][num_m_blocks][num_n_blocks]
    int num_ranks, int rank_idx,
    int m_per_rank, int n_dim,
    int block_m, int block_n) {
    
    const int tile_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int my_m_block = tile_idx / num_n_blocks;
    const int my_n_block = tile_idx % num_n_blocks;
    
    // 1. Poll ALL ranks' ready flags for this tile
    for (int src = 0; src < num_ranks; src++) {
        if (tid == 0) {
            while (ld_acq_sys(&ready_flags[src][my_m_block][my_n_block]) == 0);
        }
        __syncthreads();
    }
    
    // 2. Vectorized reduce: N-1 remote + self
    //    256T 并行 load + BF16 __hadd2 / FP32 accumulate
    const int base_row = my_m_block * block_m;
    const int base_col = my_n_block * block_n;
    const int elems_per_tile = block_m * block_n;
    
    for (int elem = tid; elem < elems_per_tile; elem += 256) {
        int row = elem / block_n;
        int col = elem % block_n;
        int global_row = base_row + row;
        int global_col = base_col + col;
        
        // Self-rank's contribution (in output)
        comm_dtype_t acc = output[global_row * n_dim + global_col];
        
        // Add remote ranks
        for (int src = 0; src < num_ranks; src++) {
            if (src == rank_idx) continue;
            comm_dtype_t val = partial_ptrs[src][global_row * n_dim + global_col];
            acc = __hadd(acc, val);  // BF16 fast path
        }
        
        output[global_row * n_dim + global_col] = acc;
    }
}
```

**优点**：简单、直接、与当前 Comm Warp 逻辑等价
**缺点**：必须等 ALL ranks ready 才开始 reduce → 尾延迟高

#### Strategy B: Flux-style Pipelined Reduce（优化方向）

参考 Flux `Sm90ReduceScatterDma`：
- 每个 CTA 有 TMA fetch pipeline（2+ stages）
- Per-rank 逐个 fetch+reduce：不等所有 rank，某个 rank ready 就 fetch
- TMA 异步 load from remote partial buffer → smem → reduce → write output
- 更好的 overlap：fetch rank_k+1 的数据同时 reduce rank_k 的数据

**复杂度更高，先实现 Strategy A 验证基线性能，再升级到 Strategy B。**

### Host-side 编排

```python
# Python API (deep_gemm/gemm_rs/__init__.py)
def bf16_gemm_rs_nt(y, a, b, sym_buffer, num_tokens_per_rank, compiled_dims='nk'):
    # 1. Launch GEMM kernel on compute_stream
    torch._C._bf16_gemm_rs_compute(y, a, b, sym_buffer, ...)
    
    # 2. Launch RS reduce kernel on comm_stream (overlapping with GEMM)
    torch._C._rs_reduce(y, sym_buffer, ...)
    
    # 3. Synchronize
    torch.cuda.current_stream().wait_stream(comm_stream)
```

**关键**：GEMM 和 RS kernel 在**不同 stream** 上运行：
- GEMM kernel 启动后立即返回（不等待完成）
- RS kernel 在 comm_stream 上启动，与 GEMM 并行执行
- RS kernel 内部 per-tile poll ready flag，自然实现流水线 overlap
- 最终通过 event/stream 同步确保两个 kernel 都完成

### Symmetric Buffer 布局（不变）

与当前单kernel完全兼容，无需修改：
- `partial_buffer[num_ranks][max_tokens_per_rank][hidden]`：每个 rank 的 partial 结果
- `ready_flags[num_ranks][num_m_blocks][num_n_blocks]`：per-tile ready flag

## 性能预期

| 场景 | 当前单kernel | 双kernel (预期) | 改善 |
|------|-------------|----------------|------|
| GEMM 吞吐 | ~600 TFLOPS | **~1100 TFLOPS** | +83% |
| K=7168 shapes (0.98-1.04x) | 输/打平 | **1.1-1.3x** | GEMM 吞吐提升 |
| K=4096 shapes (1.07-1.16x) | 赢 | **1.2-1.5x** | 进一步拉开 |
| K=2048 shapes (0.78-0.95x) | 输 | **0.85-1.0x** | 仍受 comm ratio 限制 |
| Geo Mean (8 GPU) | 1.040x | **1.15-1.25x** | |

**关键假设**：
- GEMM kernel 性能恢复到标准 GEMM 的 90-95%（scatter epilogue 略慢于 TMA 2D store）
- RS kernel 能有效 overlap：GEMM 计算后续 tile 时 RS 在 reduce 前面已完成的 tile
- Stream 级 overlap 有 ~10-20μs 的 launch gap，但对大 shape（M≥4K）影响小

## 实现计划

### Phase 1: GEMM-only Kernel（P0）

1. 基于 `sm100_bf16_gemm.cuh` 修改 Epilogue：
   - Epilogue scatter write：按 `dst_rank` 写到不同 rank 的 partial buffer
   - Per-tile ready flag signaling（`st_rel_sys`）
   - `dst_rank == rank_idx`：写 output
   - `dst_rank != rank_idx`：NVLink push + set remote flag

2. 新文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs_compute.cuh`
3. 新 JIT runtime：`csrc/jit_kernels/impls/sm100_bf16_gemm_rs_compute.hpp`
4. Heuristics：与标准 GEMM 一致（256T），无需 Comm threads

### Phase 2: RS Reduce Kernel（P0）

1. Strategy A（简单版）：
   - 新文件：`deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh`
   - 256T/CTA, vectorized BF16 __hadd2 / FP32 reduce
   - Per-tile poll ready flags, reduce N ranks → write output

2. 新 JIT runtime：`csrc/jit_kernels/impls/sm100_rs_reduce.hpp`

### Phase 3: Host 编排 + Python API（P0）

1. 双 stream 编排：compute_stream + comm_stream
2. 修改 `deep_gemm/gemm_rs/__init__.py`：先 launch GEMM → 再 launch RS
3. 修改 `csrc/apis/gemm_rs.hpp`：C++ 端双 stream 管理

### Phase 4: 测试 & Benchmark（P0）

1. 复用 `tests/test_gemm_rs.py`（正确性测试不变）
2. 复用 `benchmarks/bench_gemm_rs.py`（性能对比不变）

### Phase 5: Flux-style Pipelined Reduce（P1）

1. TMA fetch pipeline in RS kernel
2. Per-rank sequential reduce with prefetch
3. 参考 `Sm90ReduceScatterDma::fetch()` + `reduce()`

## 与当前方案的对比

| 维度 | 单kernel (当前) | 双kernel (v3) |
|------|----------------|---------------|
| GEMM 吞吐 | 600 TFLOPS | 1100 TFLOPS |
| 总线程/CTA | 384T | 256T + 256T |
| Register spilling | 严重 (168 reg/thread) | **无** (~252 reg/thread) |
| Overlap 粒度 | tile 级 (warp specialization) | stream 级 (kernel overlap) |
| 同步机制 | in-kernel barrier + flag | per-tile flag (cross-kernel) |
| 复杂度 | 1 kernel, 808 行 | 2 kernels, ~500 行 + ~200 行 |
| NCCL 依赖 | 无 | 无 |
| 内存开销 | partial + flags | 相同 |

## 风险 & 缓解

| 风险 | 缓解 |
|------|------|
| Stream 级 overlap 粒度太粗 | RS kernel 内部 per-tile polling 天然实现 tile 级 overlap |
| 两次 kernel launch 开销 | 对大 shape (M≥4K) 可忽略；小 shape 本来就赢 |
| GEMM epilogue scatter 效率 | 与当前单kernel epilogue 相同（已验证 TMA 1D async store 可行） |
| RS kernel 等待 flag 的延迟 | 紧跟 GEMM stream 启动，GEMM 第一批 tile 完成时 RS 刚开始 polling |
