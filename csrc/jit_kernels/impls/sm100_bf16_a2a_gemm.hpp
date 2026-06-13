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

// Host-side A2A communication via CE DMA (Flux-style).
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
// Overlap: kernel launches after local_ready_event; remote copies continue
// on comm_stream, kernel polls per-chunk flags.
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
    constexpr uint32_t kNumReadyChunksPerSlot = layout::BF16A2AGemmWorkspace::kNumReadyChunksPerSlot;
    DG_HOST_ASSERT(runtime_m_per_rank > 0 and block_m > 0);

    auto ceil_div = [](const int a, const int b) { return (a + b - 1) / b; };
    auto align_up = [&](const int x, const int alignment) { return ceil_div(x, alignment) * alignment; };

    ready_chunk_rows = static_cast<uint32_t>(std::max(block_m, align_up(ceil_div(runtime_m_per_rank, static_cast<int>(kNumReadyChunksPerSlot)), block_m)));
    num_ready_chunks = static_cast<uint32_t>(ceil_div(runtime_m_per_rank, static_cast<int>(ready_chunk_rows)));
    DG_HOST_ASSERT(1 <= num_ready_chunks and num_ready_chunks <= kNumReadyChunksPerSlot);

    const auto current_stream = at::cuda::getCurrentCUDAStream();
    const auto comm_stream = get_bf16_a2a_gemm_comm_stream();
    const auto input_ready_event = get_bf16_a2a_gemm_input_ready_event();
    const auto local_ready_event = get_bf16_a2a_gemm_local_ready_event();
    comm_done_event = get_bf16_a2a_gemm_comm_done_event();

    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(input_ready_event, current_stream.stream()));
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(comm_stream, input_ready_event, 0));

    // Clear all slot_state flags
    auto* local_state_base = reinterpret_cast<uint32_t*>(math::advance_ptr(
        sym_buffer.data_ptr(), reinterpret_cast<uintptr_t>(workspace.get_slot_state_ptr())));
    DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(
        local_state_base, 0, sizeof(uint32_t) * num_slots * kNumReadyChunksPerSlot, comm_stream));

    // [Iter 8] Merge all chunks of a rank into a single memcpy for fewer host API calls
    // Helper: copy entire rank data (all chunks) in one memcpy
    auto launch_rank_copy = [&](const int src_rank) {
        const size_t total_bytes = static_cast<size_t>(runtime_m_per_rank) * k * sizeof(cutlass::bfloat16_t);

        // Destination: my slot[src_rank], start of data
        auto* dst = math::advance_ptr(
            sym_buffer.data_ptr(), reinterpret_cast<uintptr_t>(workspace.get_slot_x_ptr(src_rank, 0)));

        // Source: src_rank's local_x[rank_idx, 0] (all data this rank sent me)
        auto* src = math::advance_ptr(
            reinterpret_cast<void*>(sym_buffer_ptrs[src_rank]),
            reinterpret_cast<uintptr_t>(workspace.get_local_x_ptr(rank_idx, 0)));

        DG_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyDefault, comm_stream));
    };

    // Helper: batch-set all chunk flags for a rank at once
    auto set_rank_flags = [&](const int src_rank) {
        auto* rank_state_ptr = local_state_base + src_rank * kNumReadyChunksPerSlot;
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(rank_state_ptr, 1, sizeof(uint32_t) * num_ready_chunks, comm_stream));
    };

    // Step 1: Copy local data (self -> self) first for early kernel launch
    launch_rank_copy(rank_idx);
    set_rank_flags(rank_idx);
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(local_ready_event, comm_stream));

    // Step 2: Pull remote data — rank-order with merged memcpy + batched flag setting
    // [Iter 8] Single memcpy per rank (vs N chunks), batched flags.
    // Reduces cudaMemcpyAsync calls from N*chunks to N (e.g. 32→8) and
    // cudaMemsetAsync calls from N*chunks to N (32→8).
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
