# GEMM + A2A + PDL Local Reduce: Alternative Architecture Design

> ⚠️ 历史备选架构文档：仅用于方案参考，不代表当前主线实现。
> 当前唯一主线进度与运行口径请以 `docs/PROGRESS.md` 为准。

## 1. 动机

### 1.1 V3 双核架构的 NVLink Barrier Bug

当前 V3 dual-kernel GEMM+RS 架构存在一个持续性 bug：NVLink barrier state 在 kernel launch 间持久化，导致重复调用时 barrier timeout。

**Bug 根因分析**（基于代码审查）：

`sm100_bf16_gemm_rs_compute.cuh` 中的 `nvlink_barrier` 使用 `GemmRSWorkspace` 中的三个持久化状态字段：

```c++
// layout/gemm_rs.cuh
CUTLASS_DEVICE uint32_t* get_grid_sync_count_ptr() const;     // +0 bytes
CUTLASS_DEVICE uint32_t* get_nvl_barrier_counter_ptr() const; // +4 bytes
CUTLASS_DEVICE int* get_nvl_barrier_signal_ptr(phase) const;  // +20/24 bytes
```

`nvlink_barrier` 实现逻辑（`comm/barrier.cuh`）：

```c++
// 1. SM 0 读取 counter_ptr 的 status（低2位: phase + sign）
const auto status = (*counter_ptr) & 3;
const auto signal_phase = status & 1, signal_sign = status >> 1;

// 2. 向所有远端 rank 发送信号
ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);

// 3. 更新 counter 并等待 signal 到达 target
ptx::red_add(counter_ptr, 1);
while (ptx::ld_acq_sys(signal_ptr) != target) { /* timeout */ }
```

**问题**：`counter_ptr` 和 `signal_ptr` 位于 `sym_buffer` 的持久化内存中。在第一次调用后，`counter` 被更新（+1），`signal` 被修改。第二次调用时：

- `counter` 的新值导致 `status` 解析错误（phase/sign 错位）
- `signal` 的残留值可能已经等于 target，导致提前通过；或不等于 target，导致永久等待
- 仅部分 rank 的 signal 状态错误，导致只有部分 rank 发送了信号

**修复尝试**：在 kernel 启动前重置 barrier 状态（`cudaMemsetAsync` 清零前 32 字节），但由于 GEMM compute kernel 和 RS reduce kernel 的异步执行，时序难以保证。

### 1.2 为什么需要替代方案

V3 架构的核心问题不只是 barrier bug，而是对 `nvlink_barrier` 的架构性依赖：

1. **Persistent kernel + nvlink_barrier** 是整个 V3 架构的基石
2. Barrier 状态持久化是设计决定的（不是 bug，而是 feature -- 避免每次调用重新初始化）
3. 在 dual-kernel 模式下，两个 kernel 间的 barrier 状态协调变得非常复杂
4. 任何 barrier 修复都是 fragile 的 -- 对时序敏感

因此，我们需要一个**从根本上不依赖 nvlink_barrier** 的替代架构。

## 2. 新架构：GEMM + A2A + PDL Local Reduce

### 2.1 核心思想

将 GEMM+ReduceScatter 分解为三个独立步骤：

```
Step 1: GEMM Compute    ->  C = A x B^T   (纯计算，无通信)
Step 2: All-to-All       ->  重分布 C 的行块  (标准通信)
Step 3: Local Reduce     ->  求和收到的块    (PDL overlap)
```

**关键差异 vs V3**：
- V3: GEMM kernel 边算边 scatter write（NVLink P2P push）-> 需要 nvlink_barrier 同步
- 新方案: GEMM 完全独立 -> 无需任何跨 rank 同步 -> 无 barrier bug

### 2.2 ReduceScatter = All-to-All + Local Reduce 的数学等价性

**ReduceScatter 语义**：
- 输入：每个 rank 有 tensor X（shape `[M_total, N]`）
- 操作：sum 所有 rank 的 X（逐元素），然后每个 rank 拿 1/N
- 输出：rank r 得到 `sum_rank(X_r)[r*M_per:(r+1)*M_per, :]`

**All-to-All + Local Reduce 等价分解**：
1. 每个 rank 将 X 切分为 N 个行块：`X_chunk[j] = X[j*M_per:(j+1)*M_per, :]`
2. All-to-All：rank r 发送 `X_chunk[j]` 给 rank j（对所有 j）
3. All-to-All 后，rank r 收到来自所有 rank 的 `X_chunk[rank_r]`（共 N 份）
4. Local Reduce：rank r 对收到的 N 份求和 -> `sum_rank(X_r[rank_r_rows, :])`

**在 GEMM+RS 语境下**：
- 每个rank有相同的 A（AllGather后的全量tokens）和相同的 B（expert weight）
- 每个 rank 独立计算 `C = A x B^T`（shape `[M_total, N]`）
- ReduceScatter(C) = A2A(C) + LocalReduce = 每个 rank 得到 `num_ranks * C[my_rows, :]`
- 这与 V3 的结果完全一致（见 `tests/test_gemm_rs_v3.py` 的 reference 实现）

### 2.3 三步详细设计

#### Step 1: GEMM Compute（标准 bf16_gemm_nt）

**使用现有 API**：
```python
import deep_gemm
C_full = torch.empty((total_m, N), dtype=torch.bfloat16, device='cuda')
deep_gemm.bf16_gemm_nt(A, B, C_full)
```

**关键参数**：
- `A`: shape `[total_m, K]`, bf16, K-major（contiguous in K）
- `B`: shape `[N, K]`, bf16, K-major（NT layout）
- `C_full`: shape `[total_m, N]`, bf16 或 fp32

**性能预期**：
- 256T 标准 GEMM，无 register spilling
- ~1100 TFLOPS on B300（vs V3 GEMM ~600 TFLOPS）
- 无 nvlink_barrier、无 scatter write、无 ready flag

**实现**：完全复用 `sm100_bf16_gemm.cuh`，零修改。

#### Step 2: All-to-All

**两种实现选项**：

**Option A: NCCL all_to_all_single（简单，推荐先实现）**

```python
import torch.distributed as dist

# 准备 send/recv buffer
send_tensor = C_full  # [total_m, N]，按 rank 切分行块
recv_tensor = torch.empty((num_ranks, M_per_rank, N), dtype=torch.bfloat16, device='cuda')

dist.all_to_all_single(recv_tensor, send_tensor, split_sizes=[M_per_rank]*num_ranks, group=group)
```

- **优点**：零自定义通信代码，NCCL 高度优化
- **缺点**：原子操作（全部完成才返回），无 per-chunk overlap
- **带宽**：NVLink ~900 GB/s，All-to-All 有效带宽 = 900 * (N-1)/N GB/s

**Option B: Custom CE DMA A2A（PDL overlap，高级）**

参考现有 `sm100_bf16_a2a_gemm.hpp` 中的 `launch_bf16_a2a_gemm_comm()`：

```c++
// Host-side 编排（在 comm_stream 上执行）：
// 1. cudaMemsetAsync: 清零 slot_state flags
// 2. 本地拷贝: C_full[rank_rows] -> slot[rank_idx], set flags
// 3. 远端拉取: 对于每个远端 rank j:
//      cudaMemcpyAsync(slot[j] <- rank_j C_full[rank_rows])
//      cudaMemsetAsync(slot_state[j] = 1)  // set ready flag
// 4. kernel 通过 per-chunk flag 轮询数据到达
```

- **优点**：per-chunk overlap（A2A 数据逐步到达，reduce 逐步处理）
- **缺点**：需要 Symmetric Buffer + 自定义 flag 机制
- **适用**：当 A2A 传输时间显著（>50us）时，overlap 收益大

#### Step 3: PDL Local Reduce

**核心思想**：一个轻量级 CUDA kernel，将 All-to-All 收到的 N 个行块求和为最终输出。

**Option A 对应的 Reduce Kernel（NCCL A2A 后）**：

```c++
// 简单向量求和 kernel
// 输入: recv_buffer[num_ranks][M_per_rank][N] (A2A 后的数据)
// 输出: output[M_per_rank][N]
__global__ void __launch_bounds__(256, 4)
pdl_local_reduce_kernel(
    bf16* output,              // [M_per_rank, N]
    const bf16* recv_buffer,   // [num_ranks, M_per_rank, N]
    const uint32_t M_per_rank,
    const uint32_t N,
    const uint32_t num_ranks) {

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * 256 + tid;
    const uint32_t total_elems = M_per_rank * N;

    // 128-bit vectorized access (8 BF16 per vector)
    constexpr uint32_t kVecSize = 8;
    const uint32_t num_vecs = total_elems / kVecSize;

    for (uint32_t vec_idx = gid; vec_idx < num_vecs; vec_idx += gridDim.x * 256) {
        const uint32_t elem_offset = vec_idx * kVecSize;

        // 从 rank 0 的数据开始累加（FP32 精度）
        float acc[kVecSize];
        auto* base_ptr = reinterpret_cast<const uint4*>(recv_buffer + elem_offset);
        uint4 data = *base_ptr;
        const auto* bf16_ptr = reinterpret_cast<const __nv_bfloat16*>(&data);
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++i)
            acc[i] = __bfloat162float(bf16_ptr[i]);

        // 累加 rank 1 到 rank N-1
        #pragma unroll 1
        for (uint32_t r = 1; r < num_ranks; ++r) {
            auto* rank_ptr = reinterpret_cast<const uint4*>(
                recv_buffer + r * M_per_rank * N + elem_offset);
            uint4 rdata = *rank_ptr;
            const auto* rbf16 = reinterpret_cast<const __nv_bfloat16*>(&rdata);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++i)
                acc[i] += __bfloat162float(rbf16[i]);
        }

        // FP32 -> BF16 写回 output
        uint4 result;
        auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&result);
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++i)
            out_bf16[i] = __float2bfloat16(acc[i]);
        *reinterpret_cast<uint4*>(output + elem_offset) = result;
    }
}
```

**Option B 对应的 Reduce Kernel（Custom A2A + PDL overlap）**：

```c++
// PDL-aware reduce kernel: per-rank 逐步 reduce
// 与 A2A 数据到达 overlap
__global__ void __launch_bounds__(256, 4)
pdl_local_reduce_kernel(
    bf16* output,
    const void* sym_buffer_base,
    const SymBuffer<num_ranks> sym_buffer,
    const uint32_t M_per_rank,
    const uint32_t N,
    const uint32_t num_ranks,
    const uint32_t* slot_state) {  // per-rank per-chunk ready flags

    const uint32_t tid = threadIdx.x;
    const uint32_t rank_idx = sym_buffer.rank_idx;

    // Ring order: 先处理本地 rank（立即可用），再处理远端 rank
    for (uint32_t rank_step = 0; rank_step < num_ranks; ++rank_step) {
        const uint32_t src_rank = (rank_idx + num_ranks - rank_step) % num_ranks;

        // 等待此 rank 的数据就绪
        if (tid < num_ready_chunks) {
            auto* flag_ptr = slot_state + src_rank * kNumReadyChunksPerSlot + tid;
            while (ld_acq_sys(flag_ptr) == 0) { /* spin */ }
        }
        __syncthreads();

        // Vectorized reduce: 将 src_rank 的数据加到 output
        // ... (同 Option A 的向量 reduce 逻辑)
    }
}
```

**PDL 启动配置**：

DeepGEMM 已有完整的 PDL 支持基础设施。在 `csrc/jit/handle.hpp` 中：

```c++
// cudaLaunchAttributeProgrammaticStreamSerialization
if (enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
}
```

对于 Reduce Kernel 的 PDL 启动：
1. Reduce Kernel 通过 `LaunchArgs(..., enable_pdl=true)` 启用 PDL
2. PDL 允许 kernel 在前序 kernel（A2A）完成前就开始执行
3. Kernel 内部通过 polling 机制等待数据就绪
4. 调用 `cudaTriggerProgrammaticLaunchCompletion()` 通知 runtime kernel 的"依赖部分"已完成

### 2.4 内存布局

#### Option A (NCCL A2A) 内存布局

```
GEMM Output (C_full):     [total_m, N]         bf16  -- GEMM 计算结果
A2A Send Buffer:          = C_full             bf16  -- 复用 GEMM 输出（按行切分）
A2A Recv Buffer:          [num_ranks, M_per, N] bf16  -- 收到各 rank 的行块
Reduce Output:            [M_per_rank, N]      bf16  -- 最终输出

总额外内存: num_ranks * M_per_rank * N * sizeof(bf16) (recv buffer)
```

**内存开销示例** (8 GPU, M_per_rank=4096, N=7168):
- C_full: 8 * 4096 * 7168 * 2 = 448 MB
- Recv buffer: 8 * 4096 * 7168 * 2 = 448 MB
- 总额外: 448 MB (recv buffer)

对比 V3:
- V3 partial buffer: 8 * 4096 * 7168 * 2 = 448 MB (sym_buffer)
- V3 ready flags: 可忽略
- 总额外: 448 MB

**结论**: 内存开销与 V3 相当。

#### Option B (Custom A2A) 内存布局

```
GEMM Output (C_full):     [total_m, N]         bf16  -- GEMM 计算结果
Symmetric Buffer:
  slot_x[0..N-1]:         [num_ranks, M_per, N] bf16  -- 接收各 rank 的行块
  slot_state:             [num_ranks, chunks]   u32   -- per-chunk ready flags
  barrier_signal:         32 bytes                    -- grid sync (不用 nvlink_barrier)
Reduce Output:            [M_per_rank, N]      bf16  -- 最终输出
```

## 3. 性能建模

### 3.1 时间分解

**V3 Dual-Kernel (有 overlap)**:
```
T_v3 = max(T_gemm_v3, T_rs_overlap) + T_tail
     = T_gemm_v3 + T_rs_tail  (GEMM 主导时)
     = T_gemm_v3 * (1 + rs_ratio)
```
- `T_gemm_v3`: ~600 TFLOPS GEMM (register spilling)
- `T_rs_overlap`: RS reduce 与 GEMM overlap 的部分
- `T_rs_tail`: GEMM 完成后 RS reduce 的尾部

**新方案 (GEMM + A2A + Reduce, 串行)**:
```
T_new = T_gemm_new + T_a2a + T_reduce
```
- `T_gemm_new`: ~1100 TFLOPS GEMM (纯计算)
- `T_a2a`: NCCL All-to-All 传输时间
- `T_reduce`: Local reduce 时间

### 3.2 各步骤时间估算 (8x B300, NVLink 900 GB/s, HBM3e 4 TB/s)

| Shape (M x N x K) | GEMM (1100T) | A2A (900GB/s) | Reduce (4TB/s) | Total | V3 (600T+overlap) |
|-------------------|-------------|---------------|----------------|-------|-------------------|
| 4096x7168x7168 | 393 us | 52 us | 15 us | **460 us** | ~540 us |
| 4096x7168x4096 | 224 us | 52 us | 15 us | **291 us** | ~310 us |
| 2048x7168x7168 | 196 us | 26 us | 7 us | **229 us** | ~270 us |
| 2048x4096x7168 | 112 us | 15 us | 4 us | **131 us** | ~155 us |
| 512x7168x4096 | 28 us | 7 us | 2 us | **37 us** | ~40 us |

**估算公式**：
- GEMM: `2 * M * N * K / (1100e12 / 2)` = `4*M*N*K / 1.1e12` 秒
- A2A (8 ranks): `(7/8) * M * N * 2 / 900e9` 秒（发送 7/8 的数据到远端）
- Reduce: `(num_ranks-1) * M_per_rank * N * 2 * 2 / 4e12` 秒（读 N-1 份 + 写 1 份）

### 3.3 性能预期

**vs V3 (如果 V3 bug 修复后)**:
- 大 shape (M>=4096): **新方案可能持平或略优**，因为 GEMM 吞吐从 600->1100 TFLOPS 的提升足以抵消无 overlap 的损失
- 小 shape (M<2048): **V3 可能更优**，因为 V3 的 overlap 优势在小 shape 上更显著（GEMM 时间短，overlap 比例高）
- K=2048 shapes: **新方案更优**，因为 GEMM 吞吐提升对这些 compute-bound shape 影响最大

**vs NCCL 分离方案 (GEMM + reduce_scatter)**:
- 新方案等价于 NCCL 分离，但 reduce_scatter 比 all_to_all + local_reduce 更高效
- NCCL 的 reduce_scatter 使用 ring/tree 协议，只需 (N-1)/N 的数据传输
- All-to-All 需要 (N-1)/N 的数据传输（相同），但额外需要 local reduce
- **预期**: 与 NCCL 分离方案性能相当（略慢，因为 local reduce 额外开销）

## 4. 实现方案

### 4.1 Phase 1: NCCL A2A + Simple Reduce (P0)

**目标**: 快速验证架构可行性，消除 nvlink_barrier bug。

**新增文件**：
1. `deep_gemm/gemm_rs_v4/__init__.py` -- Python API
2. `deep_gemm/include/deep_gemm/impls/sm100_pdl_local_reduce.cuh` -- PDL reduce kernel
3. `csrc/jit_kernels/impls/sm100_pdl_local_reduce.hpp` -- C++ JIT entry point
4. `tests/test_gemm_rs_v4.py` -- 正确性测试

**Python API**：
```python
def bf16_gemm_rs_nt_v4(
    y: torch.Tensor,           # [M_per_rank, N] output
    a: torch.Tensor,           # [total_m, K] input
    b: torch.Tensor,           # [N, K] weight (NT layout)
    group: dist.ProcessGroup,  # process group
    num_tokens_per_rank: int,  # actual M per rank
    compiled_dims: str = 'nk',
):
    # Step 1: GEMM (pure compute, no communication)
    C_full = torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)
    deep_gemm.bf16_gemm_nt(a, b, C_full)

    # Step 2: All-to-All via NCCL
    total_m, N = C_full.shape
    M_per_rank = total_m // group.size()
    recv_buf = torch.empty(group.size(), M_per_rank, N, dtype=torch.bfloat16, device=a.device)
    dist.all_to_all_single(recv_buf, C_full,
                           split_sizes=[M_per_rank] * group.size(),
                           group=group)

    # Step 3: PDL Local Reduce
    _C.pdl_local_reduce(y, recv_buf, M_per_rank, N, group.size())
```

**PDL Local Reduce Kernel**（复用现有 `sm100_rs_reduce.cuh` 的核心逻辑）：

```c++
// sm100_pdl_local_reduce.cuh
template <uint32_t kNumRanks, typename cd_dtype_t, uint32_t kNumThreads = 256>
__global__ void __launch_bounds__(kNumThreads, 4)
sm100_pdl_local_reduce_impl(cd_dtype_t* __restrict__ output,
                             const cd_dtype_t* __restrict__ recv_buffer,
                             const uint32_t M_per_rank,
                             const uint32_t N) {
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * kNumThreads + tid;

    constexpr uint32_t kVecBytes = 16;  // 128-bit vectorization
    constexpr uint32_t kVecSize = kVecBytes / sizeof(cd_dtype_t);  // 8 for BF16
    const uint32_t total_elems = M_per_rank * N;
    const uint32_t num_vecs = total_elems / kVecSize;

    for (uint32_t vec_idx = gid; vec_idx < num_vecs; vec_idx += gridDim.x * kNumThreads) {
        const uint32_t elem_offset = vec_idx * kVecSize;

        if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
            // BF16 -> FP32 accumulation
            float acc[kVecSize];
            auto* base_ptr = reinterpret_cast<const uint4*>(recv_buffer + elem_offset);
            uint4 data = *base_ptr;
            const auto* bf16 = reinterpret_cast<const __nv_bfloat16*>(&data);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++i)
                acc[i] = __bfloat162float(bf16[i]);

            #pragma unroll 1
            for (uint32_t r = 1; r < kNumRanks; ++r) {
                auto* rptr = reinterpret_cast<const uint4*>(
                    recv_buffer + r * M_per_rank * N + elem_offset);
                uint4 rdata = *rptr;
                const auto* rbf16 = reinterpret_cast<const __nv_bfloat16*>(&rdata);
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++i)
                    acc[i] += __bfloat162float(rbf16[i]);
            }

            uint4 result;
            auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&result);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++i)
                out_bf16[i] = __float2bfloat16(acc[i]);
            *reinterpret_cast<uint4*>(output + elem_offset) = result;
        }
    }
}
```

**PDL 启动**：Reduce kernel 通过 `LaunchArgs(grid_size, 256, 0, 1, true)` 启用 PDL，利用 `cudaLaunchAttributeProgrammaticStreamSerialization` 减少与 A2A 完成之间的 launch gap。

### 4.2 Phase 2: Custom A2A + PDL Overlap Reduce (P1)

**目标**: 实现 A2A 与 Local Reduce 的 per-chunk 级 overlap。

**新增文件**：
1. `deep_gemm/include/deep_gemm/impls/sm100_pdl_a2a_reduce.cuh` -- PDL-aware reduce kernel with per-rank polling
2. `deep_gemm/include/deep_gemm/layout/pdl_a2a_reduce.cuh` -- Workspace layout
3. `csrc/jit_kernels/impls/sm100_pdl_a2a_reduce.hpp` -- C++ entry point with host-side A2A

**设计要点**：
- 复用现有 `BF16A2AGemmWorkspace` 的 slot/slot_state 布局
- Host-side A2A 复用 `launch_bf16_a2a_gemm_comm()` 的 CE DMA 编排
- Reduce kernel 在 GEMM kernel 完成后启动，polling per-rank ready flags
- Ring order 逐步 reduce：先处理 self rank（立即可用），再处理远端 ranks

**Symmetric Buffer**：
```
slot_state[num_ranks][kNumReadyChunksPerSlot]: u32 -- per-rank per-chunk ready flags
slot_x[num_ranks][M_per_rank][N]: bf16           -- received row-chunks
```

**Host 编排**：
```c++
// 在 comm_stream 上执行：
// 1. cudaMemsetAsync: 清零 slot_state
// 2. C_full[rank_rows] -> slot[rank_idx], set flags  (本地拷贝)
// 3. For each remote rank j:
//      cudaMemcpyAsync(slot[j] <- rank_j C_full[rank_rows])
//      cudaMemsetAsync(slot_state[j] = 1)
// 4. reduce kernel 在 compute_stream 上启动（wait for local_ready_event）
```

**PDL Reduce Kernel**：
```c++
// Ring-order per-rank reduce with polling
for (uint32_t rank_step = 0; rank_step < kNumRanks; ++rank_step) {
    const uint32_t src_rank = (rank_idx + kNumRanks - rank_step) % kNumRanks;

    // Poll ready flags for this rank
    if (tid < num_ready_chunks) {
        while (ld_acq_sys(&slot_state[src_rank * kChunksPerSlot + tid]) == 0) { /* spin */ }
    }
    __syncthreads();

    // Vectorized reduce: add src_rank data to output
    // ... (same as Phase 1 vector reduce logic, data source is slot_x[src_rank])
}
```

### 4.3 Phase 3: GEMM + A2A + PDL Reduce 全融合 (P2, 可选)

**目标**: 将 GEMM 的 PDL completion 信号与 A2A 启动衔接，消除 GEMM -> A2A 之间的 host 同步。

**设计**：
- GEMM kernel 调用 `cudaTriggerProgrammaticLaunchCompletion()` 通知 runtime GEMM 完成
- A2A comm_stream wait on GEMM 的 PDL completion event（而非 host synchronize）
- Reduce kernel 通过 PDL 与 A2A completion 衔接

**挑战**：
- GEMM kernel 需要修改以支持 `cudaTriggerProgrammaticLaunchCompletion()`
- PDL 信号触发时序需要精确控制
- 需要验证 `cudaLaunchAttributeProgrammaticStreamSerialization` 与 NCCL/custom A2A 的兼容性

## 5. 与 V3 的对比

| 维度 | V3 Dual-Kernel | GEMM+A2A+PDL Reduce |
|------|---------------|---------------------|
| **GEMM 吞吐** | ~600 TFLOPS (384T, register spilling) | **~1100 TFLOPS** (256T, no spilling) |
| **通信方式** | NVLink P2P push (scatter write) | NCCL All-to-All / Custom CE DMA |
| **同步机制** | nvlink_barrier (buggy) | **无跨 rank kernel 同步** |
| **Overlap** | GEMM+RS tile-level | A2A+Reduce chunk-level (Phase 2) |
| **内存开销** | partial + flags (~448MB) | recv_buffer (~448MB) |
| **NCCL 依赖** | 无 | Phase 1: 有 / Phase 2: 无 |
| **代码复杂度** | 高 (custom barrier + scatter) | **低** (标准 GEMM + 标准通信) |
| **Bug 风险** | 高 (barrier state) | **极低** (无自定义同步原语) |
| **可调试性** | 难 (跨 kernel 异步 bug) | **易** (每步可独立验证) |

## 6. 风险与缓解

| 风险 | 严重程度 | 缓解措施 |
|------|---------|---------|
| GEMM 与 A2A 之间无 overlap -> 大 shape 性能退化 | 中 | GEMM 吞吐从 600->1100 TFLOPS 大幅提升应能补偿 |
| NCCL A2A 可能比 fused NVLink push 慢 | 中 | Phase 2 使用 Custom CE DMA + PDL overlap |
| Local reduce 额外内存带宽开销 | 低 | HBM3e 4TB/s 足够，reduce 时间占比小 |
| PDL 兼容性问题 | 低 | DeepGEMM 已有完整 PDL 基础设施 |
| reduce_scatter vs all_to_all 语义差异 | 低 | 数学等价已证明；但需注意 NCCL reduce_scatter 内部更高效 |

## 7. 测试计划

### 7.1 正确性测试

复用 `tests/test_gemm_rs_v3.py` 的测试框架，替换 kernel 调用：

```python
# V4 correctness test
def compute_reference(a, b, rank_idx, num_ranks, tokens_per_rank, local_rank):
    # 完全复用 V3 的 reference 计算
    total_m = tokens_per_rank * num_ranks
    n_dim = b.shape[0]
    d_full = torch.zeros((total_m, n_dim), dtype=torch.bfloat16, device=f'cuda:{local_rank}')
    deep_gemm.bf16_gemm_nt(a, b, d_full)
    # ... all_gather + sum rows
    return ref

def test_v4():
    # Step 1: GEMM
    C_full = torch.empty((total_m, n_dim), dtype=torch.bfloat16, device=...)
    deep_gemm.bf16_gemm_nt(a, b, C_full)

    # Step 2: A2A
    recv_buf = torch.empty(num_ranks, tokens_per_rank, n_dim, ...)
    dist.all_to_all_single(recv_buf, C_full, ...)

    # Step 3: Local Reduce
    y = torch.zeros(tokens_per_rank, n_dim, ...)
    deep_gemm._C.pdl_local_reduce(y, recv_buf, tokens_per_rank, n_dim, num_ranks)

    # Verify: y should equal num_ranks * C_full[rank_rows]
    assert_close(y.float(), ref.float(), atol=0.01 * num_ranks)
```

### 7.2 多次调用稳定性测试

**关键**: V3 bug 在多次调用时出现。V4 必须通过此测试。

```python
# Stress test: 1000 iterations without barrier timeout
for i in range(1000):
    y = bf16_gemm_rs_nt_v4(y, a, b, group, tokens_per_rank)
torch.cuda.synchronize()
print(f"PASS: 1000 iterations without timeout")
```

### 7.3 性能 Benchmark

复用 `benchmarks/bench_gemm_rs.py`，对比：
1. NCCL 分离方案 (GEMM + reduce_scatter)
2. V3 dual-kernel (如果 bug 修复)
3. V4 NCCL A2A + simple reduce (Phase 1)
4. V4 Custom A2A + PDL reduce (Phase 2)

## 8. 实现时间线

| Phase | 工作量 | 预期时间 |
|-------|-------|---------|
| Phase 1: NCCL A2A + Simple Reduce | 2-3 天 | Python API + reduce kernel + test |
| Phase 2: Custom A2A + PDL Reduce | 3-5 天 | Symmetric buffer + host A2A + PDL kernel |
| Phase 3: GEMM+PDL 全融合 | 2-3 天 | GEMM PDL trigger + stream 编排 |

**Phase 1 优先级最高**，因为：
1. 可以立即验证架构可行性
2. 消除 nvlink_barrier bug
3. 代码改动最少
4. 性能可接受（GEMM 吞吐大幅提升补偿无 overlap）

## 9. 关键代码引用

| 文件 | 用途 |
|------|------|
| `deep_gemm/__init__.py` | `bf16_gemm_nt` API 入口 |
| `deep_gemm/gemm_rs/__init__.py` | V3 `bf16_gemm_rs_nt_v3` API |
| `deep_gemm/a2a_gemm/__init__.py` | A2A-GEMM Python API 参考 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` | 标准 GEMM kernel (复用) |
| `deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh` | RS reduce kernel 参考 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_a2a_gemm.cuh` | A2A-GEMM kernel 参考 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs_compute.cuh` | V3 GEMM compute kernel |
| `deep_gemm/include/deep_gemm/comm/barrier.cuh` | nvlink_barrier (bug 来源) |
| `deep_gemm/include/deep_gemm/layout/gemm_rs.cuh` | GemmRSWorkspace 布局 |
| `deep_gemm/include/deep_gemm/layout/bf16_a2a_gemm.cuh` | A2A Workspace 布局参考 |
| `deep_gemm/include/deep_gemm/layout/sym_buffer.cuh` | SymBuffer NVLink P2P 映射 |
| `csrc/jit_kernels/impls/sm100_bf16_a2a_gemm.hpp` | Host-side A2A 编排参考 |
| `csrc/jit_kernels/impls/sm100_bf16_gemm_rs_compute.hpp` | V3 双核 stream 编排 |
| `csrc/jit/handle.hpp` | PDL launch attribute 实现 |
| `csrc/jit/kernel_runtime.hpp` | `LaunchArgs.enable_pdl` 配置 |
| `tests/test_gemm_rs_v3.py` | V3 正确性测试 (参考) |

## 10. 附录：NVLink Barrier State 持久化问题详解

### 10.1 Barrier 状态机

`nvlink_barrier` 使用一个 phase-based counter/signal 协议：

```
counter (uint32_t):
  bits[1:0] = status = {signal_phase, signal_sign}
  bits[31:2] = invocation count

signal (int):
  当前值 = 累积的信号值
  target = signal_sign ? 0 : kNumRanks
```

**初始状态** (buffer zeroed):
- counter = 0, status = {phase=0, sign=0}
- signal = 0
- target = kNumRanks (因为 sign=0)

**第一次调用**:
1. status = 0b00 -> phase=0, sign=0
2. signal_ptr = get_nvl_barrier_signal_ptr(0) -> +20 bytes
3. 每个rank发送 +1 (因为 sign=0): signal 从 0 -> kNumRanks
4. counter += 1 -> counter = 1, status = 0b01 (phase=1, sign=0)
5. signal 到达 kNumRanks = target -> barrier 通过

**第二次调用** (如果未重置):
1. status = 0b01 -> phase=1, sign=0
2. signal_ptr = get_nvl_barrier_signal_ptr(1) -> +24 bytes (不同的 signal 地址!)
3. 但 +24 处的初始值不确定（可能非零）
4. 如果上次 write 到了 +24 处（某些 kernel 路径），则 signal 可能已经有值
5. target = kNumRanks (因为 sign=0)
6. 如果 signal 残留值 = kNumRanks -> barrier 提前通过（不等所有 rank）
7. 如果 signal 残留值 != kNumRanks -> 永久等待（timeout）

**关键**: `get_nvl_barrier_signal_ptr(phase)` 根据 phase 选择不同的地址：
- phase 0 -> +20 bytes
- phase 1 -> +24 bytes

但 32 字节的 barrier signal 区域只分配了 2 个 phase 的空间。如果 counter 的低 2 位因为累积调用而跨越了 phase 边界多次，signal 地址可能指向已被其他数据覆盖的区域。

### 10.2 Dual-Kernel 模式下的加剧因素

在 V3 dual-kernel 模式下，GEMM compute kernel 调用 `nvlink_barrier` 两次（init + final），每次调用修改 counter。RS reduce kernel 不调用 barrier，但在下次 GEMM compute kernel 启动时，counter 的状态已经不正确。

此外，`grid_sync` 使用的 `get_grid_sync_count_ptr<kGridSyncIndex>()` 也有类似问题 -- 它的 atomic counter 在 kernel 间也会持久化。

### 10.3 为什么新方案从根本上解决此问题

新方案的 PDL Local Reduce kernel **完全不使用 nvlink_barrier**：
- 无 `grid_sync`（reduce kernel 不需要跨 SM 全局同步 -- 各 CTA 独立处理不同元素）
- 无 `nvlink_barrier`（reduce 是纯本地操作，不需要跨 rank 同步）
- 无 persistent state（kernel 完成后不需要保留任何状态）

这是**架构性**的改善，而非临时修复。
