#pragma once

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/gemm_rs_compute.hpp"
#include "sm100_rs_reduce.hpp"  // SM100RSReduceRuntime (reuse existing definition)

namespace deep_gemm {

// ====================================================================
//  Stream / Event management for dual-kernel GEMM+RS overlap
// ====================================================================
namespace {

inline cudaStream_t get_gemm_rs_comm_stream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(true, at::cuda::current_device());
    return stream.stream();
}

inline cudaEvent_t get_gemm_rs_launched_event() {
    static thread_local cudaEvent_t event = []() -> cudaEvent_t {
        cudaEvent_t evt;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        return evt;
    }();
    return event;
}

inline cudaEvent_t get_gemm_rs_comm_done_event() {
    static thread_local cudaEvent_t event = []() -> cudaEvent_t {
        cudaEvent_t evt;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        return evt;
    }();
    return event;
}

} // anonymous namespace

// ====================================================================
//  Dual-kernel GEMM Compute Kernel (Part 1 of GEMM+RS v3)
//  256T, no Comm Warps — full throughput for GEMM computation
// ====================================================================
class SM100BF16GemmRSComputeRuntime final : public LaunchRuntime<SM100BF16GemmRSComputeRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int m, n, k;
        int num_ranks;
        at::ScalarType y_dtype;
        at::ScalarType comm_dtype;
        GemmRSComputeConfig config;

        layout::SymBuffer<> sym_buffer_ptrs;
        void* output;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs_compute.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_rs_compute_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.swizzle_a_mode, args.config.swizzle_b_mode, args.config.swizzle_cd_mode,
    args.config.num_multicast, args.config.is_multicast_on_a ? "true" : "false",
    args.config.swap_ab ? "true" : "false", args.config.with_accumulation ? "true" : "false",
    args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.y_dtype),
    to_string(args.comm_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t shape_m_per_rank = static_cast<uint32_t>(args.max_m_per_rank);
        uint32_t runtime_m_per_rank = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        void* output = args.output;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            shape_m_per_rank,
            runtime_m_per_rank,
            shape_n,
            shape_k,
            output,
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_b));
    }
};

// ====================================================================
//  Legacy entry point: launch GEMM compute kernel only (serial mode)
// ====================================================================
static void sm100_bf16_gemm_rs_compute_nt(const torch::Tensor& y,
                                           const torch::Tensor& a,
                                           const torch::Tensor& b,
                                           const torch::Tensor& sym_buffer,
                                           const std::vector<int64_t>& sym_buffer_ptrs,
                                           const int& rank_idx,
                                           const int& max_m_per_rank,
                                           const int& runtime_m_per_rank,
                                           const int& n,
                                           const int& k,
                                           const std::string& compiled_dims,
                                           const at::ScalarType& comm_dtype = torch::kBFloat16) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_compute_config(m, n, k, num_sms, static_cast<int>(a.element_size()), num_ranks);

    DG_HOST_ASSERT(config.block_k == 64);

    const auto tensor_map_a = make_tma_2d_desc(a, k, m,
        config.block_k, config.load_block_m, static_cast<int>(a.stride(-2)), config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, n,
        config.block_k, config.load_block_n, static_cast<int>(b.stride(-2)), config.swizzle_b_mode);

    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads;

    const SM100BF16GemmRSComputeRuntime::Args args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .m = m, .n = n, .k = k,
        .num_ranks = num_ranks,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .output = y.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .launch_args = LaunchArgs(num_sms, total_threads, config.smem_size, config.num_multicast)
    };

    const auto code = SM100BF16GemmRSComputeRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_rs_compute_nt", code);
    SM100BF16GemmRSComputeRuntime::launch(runtime, args);
}

// ====================================================================
//  Entry point: launch dual-kernel GEMM+RS with stream-level overlap
//
//  Overlap design (Flux-inspired):
//    compute_stream:  GEMM compute kernel -> scatter write + set per-tile flag
//    comm_stream:     RS reduce kernel -> poll per-tile flag -> reduce -> write output
//
//  The RS reduce kernel polls per-tile ready flags set by GEMM epilogue.
//  As GEMM completes tiles, RS reduce naturally picks them up — tile-level
//  pipeline overlap without explicit chunk management.
//
//  Event synchronization:
//    1. compute_stream records gemm_launched_event after GEMM kernel launch
//    2. comm_stream waits on gemm_launched_event before launching RS reduce
//    3. comm_stream records comm_done_event after RS reduce completes
//    4. compute_stream waits on comm_done_event at the end
// ====================================================================
static void sm100_bf16_gemm_rs_v3_nt(const torch::Tensor& y,
                                       const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& sym_buffer,
                                       const std::vector<int64_t>& sym_buffer_ptrs,
                                       const int& rank_idx,
                                       const int& max_m_per_rank,
                                       const int& runtime_m_per_rank,
                                       const int& n,
                                       const int& k,
                                       const std::string& compiled_dims,
                                       const at::ScalarType& comm_dtype = torch::kBFloat16) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_compute_config(m, n, k, num_sms, static_cast<int>(a.element_size()), num_ranks);

    DG_HOST_ASSERT(config.block_k == 64);
    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // Create TMA descriptors
    const auto tensor_map_a = make_tma_2d_desc(a,
                                               k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(a.stride(-2)),
                                               config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);

    // -- Step 1: Launch GEMM compute kernel on compute_stream --
    const auto compute_stream = at::cuda::getCurrentCUDAStream();
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads;

    const SM100BF16GemmRSComputeRuntime::Args compute_args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .m = m, .n = n, .k = k,
        .num_ranks = num_ranks,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .output = y.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .launch_args = LaunchArgs(num_sms,
                                  total_threads,
                                  config.smem_size,
                                  config.num_multicast)
    };

    const auto compute_code = SM100BF16GemmRSComputeRuntime::generate(compute_args);
    const auto compute_runtime = compiler->build("sm100_bf16_gemm_rs_compute_nt", compute_code);
    SM100BF16GemmRSComputeRuntime::launch(compute_runtime, compute_args);

    // Record event after GEMM kernel launch (compute_stream)
    auto gemm_launched_event = get_gemm_rs_launched_event();
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(gemm_launched_event, compute_stream.stream()));

    // -- Step 2: Launch RS reduce kernel on comm_stream --
    const auto comm_stream = get_gemm_rs_comm_stream();

    // comm_stream waits for GEMM kernel to be at least launched
    // (not completed -- overlap happens naturally via per-tile polling)
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(comm_stream, gemm_launched_event, 0));

    // Grid size = total tiles for self-rank output
    const int total_tiles = (runtime_m_per_rank + config.block_m - 1) / config.block_m *
                            (n + config.block_n - 1) / config.block_n;
    // Cap grid size to avoid too many CTAs on small shapes
    const int grid_size = std::min(total_tiles, num_sms);

    const SM100RSReduceRuntime::Args reduce_args = {
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n,
        .max_m_per_rank = max_m_per_rank,
        .num_ranks = num_ranks,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .output = y.data_ptr(),
        .launch_args = LaunchArgs(grid_size,
                                  256,
                                  0,      // no smem needed for reduce kernel
                                  1)      // no multicast
    };

    const auto reduce_code = SM100RSReduceRuntime::generate(reduce_args);
    const auto reduce_runtime = compiler->build("sm100_rs_reduce", reduce_code);

    // Launch RS reduce on comm_stream using direct kernel launch
    // (LaunchRuntime::launch always uses getCurrentCUDAStream, so we bypass it)
    {
        const auto reduce_kernel = reduce_runtime->kernel;
        const dim3 grid_dim = {static_cast<unsigned>(grid_size), 1, 1};
        const dim3 block_dim = {256, 1, 1};
        auto launch_config = construct_launch_config(
            reduce_kernel, comm_stream, 0, grid_dim, block_dim, 1, false);
        uint32_t ru_m = static_cast<uint32_t>(runtime_m_per_rank);
        uint32_t ru_n = static_cast<uint32_t>(n);
        uint32_t ru_sm = static_cast<uint32_t>(max_m_per_rank);
        auto* ru_output = reduce_args.output;
        auto ru_sym_buffer = reduce_args.sym_buffer_ptrs;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(reduce_kernel, launch_config,
            ru_output, ru_sym_buffer, ru_m, ru_n, ru_sm));
    }

    // Record comm_done_event on comm_stream
    auto comm_done_event = get_gemm_rs_comm_done_event();
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(comm_done_event, comm_stream));

    // -- Step 3: compute_stream waits for RS reduce to complete --
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(compute_stream.stream(), comm_done_event, 0));
}

} // namespace deep_gemm
