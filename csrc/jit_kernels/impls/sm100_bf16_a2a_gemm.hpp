#pragma once

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/bf16_a2a_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/ag_gemm.hpp"  // Reuse AG+GEMM heuristics (same tile/thread config)

namespace deep_gemm {

namespace {

inline cudaStream_t get_bf16_a2a_gemm_comm_stream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(true, at::cuda::current_device());
    return stream.stream();
}

inline cudaEvent_t get_bf16_a2a_gemm_input_ready_event() {
    static thread_local cudaEvent_t event = []() {
        cudaEvent_t input_ready_event;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&input_ready_event, cudaEventDisableTiming));
        return input_ready_event;
    }();
    return event;
}

inline cudaEvent_t get_bf16_a2a_gemm_local_ready_event() {
    static thread_local cudaEvent_t event = []() {
        cudaEvent_t local_event;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&local_event, cudaEventDisableTiming));
        return local_event;
    }();
    return event;
}

inline cudaEvent_t get_bf16_a2a_gemm_comm_done_event() {
    static thread_local cudaEvent_t event = []() {
        cudaEvent_t done_event;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming));
        return done_event;
    }();
    return event;
}

// =========================================================================
//  Host-side A2A communication via CE DMA (Flux-style).
// =========================================================================
//
// A2A semantics: each rank i has local_x[num_ranks, M_per_rank, K],
// where local_x[j] is the chunk to send to rank j.
//
// Communication (PULL pattern):
//   1. Clear local slot_state flags
//   2. Copy local_x[rank_idx] -> slot[rank_idx] (local copy, self chunk)
//   3. For each remote rank j: pull j's local_x[rank_idx] into my slot[j]
//   4. Each copy sets per-chunk ready flags
//
// =========================================================================
//  Event 机制：两条 Stream + 三个 Event 实现通信计算 overlap
// =========================================================================
//
//  Stream 分工：
//   ┌──────────────────┐     ┌──────────────────────────┐
//   │  current_stream  │     │       comm_stream         │
//   │   (计算流)        │     │    (通信流，独立 stream)    │
//   │                  │     │                          │
//   │  GEMM kernel ←───┼─────┤  清除 slot_state 标志      │
//   │   (TMA + MMA +   │     │  本地拷贝 (self→self)      │
//   │    epilogue)     │     │  设本地就绪标志             │
//   │                  │     │  record local_ready ──────┤
//   │                  │  ┌──┤                          │
//   │  轮询 slot_state │  │  │  环形 PULL 远程数据        │
//   │  等待远程数据就绪  │  │  │  record comm_done ────────┤
//   └──────────────────┘  │  └──────────────────────────┘
//                         │
//                         └─ 两条流通过 Event 同步，kernel 不等 remote 全完成就开始
//
//  三个 Event 的时间线：
//
//   时间 ──────────────────────────────────────────────────────────────►
//
//   current_stream: ──► [用户写入 sym_buffer.x ...] ──┐
//                                                      │ record
//                                                      ▼ input_ready_event
//   comm_stream:            (等待 input_ready) ────────┼──► 清标志 ──►
//                                                      │
//                                                      │  本地拷贝 ──►
//                                                      │  设标志    ──►
//                                                      │  record
//                                                      ▼ local_ready_event
//   current_stream:  (等待 local_ready) ───────────────┼──► Kernel 启动！
//                                                      │   ┌ 轮询远程 flag
//                                                      │   │
//   comm_stream:           远程PULL(i-1) ──► 远程PULL(i-2) ──► ...
//                                                      │   │
//                                                      │   │  PULL(i+1) ──►
//                                                      │   │  record
//                                                      │   │  comm_done_event
//                                                      │   │
//                                                      │   └──► kernel 算完
//                                                      │         current_stream
//                                                      │         等待 comm_done
//                                                      │         (确保 buffer 不提前释放)
//
//  Key insight:
//   - local_ready_event 触发 kernel 启动，此时远程数据可能还在传输
//   - kernel 内部通过 ld_acq_sys 轮询 slot_state，等到 chunk 就绪才加载
//   - comm_done_event 保证函数返回前所有 DMA 完成，sym_buffer 不会被提前释放
inline void launch_bf16_a2a_gemm_comm(const torch::Tensor& sym_buffer,
                                      const std::vector<int64_t>& sym_buffer_ptrs,
                                      const int& rank_idx,
                                      const int& max_m_per_rank,
                                      const int& runtime_m_per_rank,
                                      const int& num_slots,
                                      const int& k,
                                      const int& block_m,
                                      uint32_t& ready_chunk_rows,
                                      uint32_t& num_ready_chunks,
                                      cudaEvent_t& comm_done_event) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto workspace = layout::BF16A2AGemmWorkspace(nullptr, num_ranks, max_m_per_rank, k, num_slots);
    // kNumReadyChunksPerSlot = 4（每个 slot 最多 4 个 chunk）
    // 目的是将每个 rank 的数据拆分成 1~4 个 chunk，让 kernel 可以逐 chunk 轮询就绪标志，实现通信和计算的重叠。
    constexpr uint32_t kNumReadyChunksPerSlot = layout::BF16A2AGemmWorkspace::kNumReadyChunksPerSlot;
    DG_HOST_ASSERT(runtime_m_per_rank > 0 and block_m > 0);

    auto ceil_div = [](const int a, const int b) { return (a + b - 1) / b; };
    auto align_up = [&](const int x, const int alignment) { return ceil_div(x, alignment) * alignment; };

    
    // block_m = 128（kernel tile 的 M 维度）
    // Step 1: ceil_div(runtime_m, 4) — 如果均匀分成 4 个 chunk，每 chunk 多少行 
    // Step 2: align_up(..., 128) — 对齐到 tile 边界 
    // Step 3: max(128, ...) — 保证每 chunk 至少一个 tile 
    ready_chunk_rows = static_cast<uint32_t>(std::max(block_m, align_up(ceil_div(runtime_m_per_rank, static_cast<int>(kNumReadyChunksPerSlot)), block_m)));
    // Step 4: ceil_div(runtime_m, ready_chunk_rows) — 反算实际多少个 chunk
    num_ready_chunks = static_cast<uint32_t>(ceil_div(runtime_m_per_rank, static_cast<int>(ready_chunk_rows)));
    DG_HOST_ASSERT(1 <= num_ready_chunks and num_ready_chunks <= kNumReadyChunksPerSlot);

    const auto current_stream = at::cuda::getCurrentCUDAStream();       // 主计算流
    const auto comm_stream = get_bf16_a2a_gemm_comm_stream();           // 通信专用流（独立，与计算并发）
    const auto input_ready_event = get_bf16_a2a_gemm_input_ready_event(); // 输入数据就绪
    const auto local_ready_event = get_bf16_a2a_gemm_local_ready_event(); // 本地拷贝完成
    comm_done_event = get_bf16_a2a_gemm_comm_done_event();              // 所有远程拷贝完成

    // 在 current_stream 上打点，标记 sym_buffer.x 的写入已完成
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(input_ready_event, current_stream.stream()));
    // comm_stream 等待 input_ready_event：确保用户已写入 sym_buffer.x 才开始 DMA
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(comm_stream, input_ready_event, 0));

    // 清空所有就绪标志，避免 residual 导致 kernel 提前加载未到达的数据

    // Clear all slot_state flags
    auto* local_state_base = reinterpret_cast<uint32_t*>(math::advance_ptr(
        sym_buffer.data_ptr(), reinterpret_cast<uintptr_t>(workspace.get_slot_state_ptr())));
    DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        local_state_base, 0, sizeof(uint32_t) * num_slots * kNumReadyChunksPerSlot, comm_stream));

    // [Iter 8] Merge all chunks of a rank into a single memcpy for fewer host API calls
    //
    // PULL 模式：当前 rank 从 src_rank 的对称内存中拉取数据到自己的接收槽。
    //
    // 数据流（以 rank 0 拉取 rank 2 的数据为例）：
    //   源：rank 2 的 local_x[0]（rank 2 准备发给 rank 0 的那份数据）
    //   目标：rank 0 的 slot[2]（rank 0 为 rank 2 预留的接收槽）
    //
    // 关键：sym_buffer_ptrs[src_rank] 是 src_rank 对称内存的 GPU 虚拟地址，
    //       通过 PyTorch 的 symm_mem.rendezvous() 交换获得。
    //       有了这个指针，当前 rank 可以直接用 cudaMemcpyAsync 读取
    //       远程 rank 的 GPU 内存（通过 NVLink/PCIe CE DMA）。
    //
    // 优化：之前每 chunk 一次 cudaMemcpyAsync（N ranks × 4 chunks = 32 次），
    //       现在整个 rank 一次（N ranks = 8 次），大幅减少 host API 开销。
    auto launch_rank_copy = [&](const int src_rank) {
        const size_t total_bytes = static_cast<size_t>(runtime_m_per_rank) * k * sizeof(cutlass::bfloat16_t);

        // 目标地址：当前 rank 的 slot[src_rank]，即这块数据在本地 buffer 内的偏移
        auto* dst = math::advance_ptr(
            sym_buffer.data_ptr(), reinterpret_cast<uintptr_t>(workspace.get_slot_x_ptr(src_rank, 0)));

        // 源地址：src_rank 对称内存中 local_x[rank_idx] 的位置
        //        sym_buffer_ptrs[src_rank] — src_rank 的 buffer 基址（GPU 虚拟地址，跨节点有效）
        //        get_local_x_ptr(rank_idx)  — 跳到 local_x 中「发给当前 rank」的那个 chunk
        auto* src = math::advance_ptr(
            reinterpret_cast<void*>(sym_buffer_ptrs[src_rank]),
            reinterpret_cast<uintptr_t>(workspace.get_local_x_ptr(rank_idx, 0)));

        // cudaMemcpyAsync(dst, src, count, kind, stream):
        //   dst   → 当前 rank 的 slot[src_rank]（本地 buffer 内偏移）
        //   src   → src_rank 的 local_x[rank_idx]（远程 rank 的 GPU 地址）
        //   count → total_bytes = runtime_m_per_rank × K × sizeof(bf16)，实际数据量
        //   kind  → cudaMemcpyDefault（自动选择最优路径：NVLink/PCIe CE DMA）
        //   stream→ comm_stream（通信专用流，与计算流并发，实现 overlap）
        // 效果：通过 GPU 间直接 DMA（不经 CPU），从远程 rank 拉取数据到本地接收槽。
        DG_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyDefault, comm_stream));
    };
 
    // 批量设置某个 rank 的就绪标志，告知 kernel 该 rank 的数据已全部到达。
    //
    // slot_state 布局：slot_state[slot_idx][chunk_idx]，每个 slot 最多 kNumReadyChunksPerSlot(4) 个 chunk。
    // 设置所有 num_ready_chunks 个标志为 1，表示该 rank 的所有 chunk 就绪。
    //
    // kernel 端在 TMA 加载 A 矩阵前，通过 ld_acq_sys 轮询这些标志，
    // 确保对应 chunk 的数据已传输完毕才开始计算，实现通信与计算 overlap。
    auto set_rank_flags = [&](const int src_rank) {
        auto* rank_state_ptr = local_state_base + src_rank * kNumReadyChunksPerSlot;

        // cudaMemsetAsync(devPtr, value, count, stream):
        //   devPtr = rank_state_ptr           → slot_state[src_rank][0] 地址
        //   value  = 1                        → 每字节填充 0x01
        //   count  = sizeof(uint32_t) * num_ready_chunks  → 填充字节数
        //   stream = comm_stream              → 通信流，异步执行
        // 效果：每个 uint32 的 4 字节都填 0x01，结果 = 0x01010101（≠0 即就绪）
        // kernel 端用 ld_acq_sys 读取，只要 ≠0 就表示该 chunk 数据已到达。
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(rank_state_ptr, 1, sizeof(uint32_t) * num_ready_chunks, comm_stream));
    };

    // =========================================================================
    //  调度核心：PULL 环形顺序，与 kernel 的计算顺序严格对齐
    // =========================================================================
    //
    // Kernel 的计算顺序（A2A ring order，见 sm100_bf16_a2a_gemm.cuh）：
    //   rank_idx, (rank_idx-1+n)%n, (rank_idx-2+n)%n, ..., (rank_idx+1)%n
    //   即：先算自己，再逆时针环形逐个算远程 rank。
    //
    // Host 端 PULL 顺序必须与 kernel 计算顺序一致：
    //   提前拉取的数据 → 提前被 kernel 消费 → 最大化 overlap
    //
    // Step 1: 先拉取自己发给自己的数据（本地拷贝）
    //         → 设就绪标志 → 记录 local_ready_event
    //         → kernel 可以在 remote 数据还在传输中就开始算本地 tile！
    launch_rank_copy(rank_idx);
    set_rank_flags(rank_idx);
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(local_ready_event, comm_stream));

    // Step 2: 逆时针环形 PULL 远程数据
    //         顺序：rank i-1, i-2, ... i+1（与 kernel 计算顺序一致）
    //
    // [Iter 8] 每个 rank 一次 memcpy（vs 之前每 chunk 一次）
    //  cudaMemcpyAsync 调用：N×chunks → N（如 32→8）
    //  cudaMemsetAsync 调用：N×chunks → N（如 32→8）
    for (int step = 1; step < num_ranks; ++ step) {
        const int src_rank = (rank_idx + num_ranks - step) % num_ranks;
        launch_rank_copy(src_rank);
        set_rank_flags(src_rank);
    }

    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(comm_done_event, comm_stream));
    // Kernel launches after local_ready_event (wired below in the caller)
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(current_stream.stream(), local_ready_event, 0));
}

} // namespace

class SM100BF16A2AGemmRuntime final : public LaunchRuntime<SM100BF16A2AGemmRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int n, k;
        int num_slots;
        int ready_chunk_rows;
        int num_ready_chunks;
        int num_ranks;
        at::ScalarType d_dtype;
        AGGemmConfig config;

        void* d;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_a2a_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_a2a_gemm_nt_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.num_ag_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_multicast,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.d_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.d,
            args.max_m_per_rank,
            args.runtime_m_per_rank,
            args.n,
            args.k,
            args.num_slots,
            args.ready_chunk_rows,
            args.num_ready_chunks,
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_b,
            args.tensor_map_d));
    }
};

static void sm100_bf16_a2a_gemm_nt(const torch::Tensor& d,
                                   const torch::Tensor& slots_x,
                                   const torch::Tensor& b,
                                   const torch::Tensor& sym_buffer,
                                   const std::vector<int64_t>& sym_buffer_ptrs,
                                   const int& rank_idx,
                                   const int& max_m_per_rank,
                                   const int& runtime_m_per_rank,
                                   const int& num_slots,
                                   const int& n,
                                   const int& k,
                                   const std::string& compiled_dims) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    auto config = get_ag_gemm_config(runtime_m_per_rank * num_ranks, n, k, num_sms, static_cast<int>(b.element_size()));
    DG_HOST_ASSERT(config.block_k == 64);

    uint32_t ready_chunk_rows = 0, num_ready_chunks = 0;
    cudaEvent_t comm_done_event;
    const auto current_stream = at::cuda::getCurrentCUDAStream();

    // Launch host-side A2A communication on comm_stream
    launch_bf16_a2a_gemm_comm(sym_buffer, sym_buffer_ptrs, rank_idx, max_m_per_rank,
                              runtime_m_per_rank, num_slots, k, config.block_m,
                              ready_chunk_rows, num_ready_chunks, comm_done_event);

    // OVERLAP: kernel launches after local_ready_event (wired inside launch_bf16_a2a_gemm_comm).
    // Remote chunks still copying on comm_stream; kernel polls per-chunk barrier flags.

    const auto tensor_map_a = make_tma_2d_desc(slots_x,
                                               k, max_m_per_rank * num_slots,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(slots_x.stride(-2)),
                                               config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_2d_desc(d,
                                               static_cast<int>(d.size(-1)), static_cast<int>(d.size(-2)),
                                               config.swizzle_cd_mode / static_cast<int>(d.element_size()), config.block_m,
                                               static_cast<int>(d.stride(-2)),
                                               config.swizzle_cd_mode);

    const SM100BF16A2AGemmRuntime::Args args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n, .k = k,
        .num_slots = num_slots,
        .ready_chunk_rows = static_cast<int>(ready_chunk_rows),
        .num_ready_chunks = static_cast<int>(num_ready_chunks),
        .num_ranks = num_ranks,
        .d_dtype = d.scalar_type(),
        .config = config,
        .d = d.data_ptr(),
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_ag_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size,
                                  config.num_multicast)
    };

    const auto code = SM100BF16A2AGemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_a2a_gemm_nt", code);
    SM100BF16A2AGemmRuntime::launch(runtime, args);
    // Wait for comm to complete before returning (ensures sym_buffer is not freed early)
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(current_stream.stream(), comm_done_event, 0));
}

} // namespace deep_gemm
