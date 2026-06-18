#pragma once

#include <cstdlib>
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

// ════════════════════════════════════════════════════════════════
//  TRUE Flux-style PULL-based GEMM + Reduce-Scatter (dual-kernel)
//
//    Kernel 1 (GEMM compute, 256T, no comm warps):
//        sm100_bf16_gemm_rs_impl — epilogue scatter-writes each tile to the LOCAL
//        scatter buffer slot[dst_rank] + sets a LOCAL per-tile ready flag. No NVLink
//        traffic in the GEMM epilogue (this is the core Flux win over push-based v3).
//
//    Kernel 2 (RS reduce, 256T/CTA, pull):
//        sm100_rs_reduce_impl<..., kPullBased=true> — for each tile of this rank's chunk,
//        polls every src rank's REMOTE flag, accumulates every src rank's REMOTE slot[R]
//        (via sym_buffer.map P2P), writes the final output, then resets the remote flags.
//
//    Stream-level overlap: GEMM compute on compute_stream, RS reduce on comm_stream,
//    coordinated by CUDA events. Per-tile ready flags give natural tile-level overlap.
// ════════════════════════════════════════════════════════════════

namespace {

// Dedicated comm stream + events for the PULL GEMM+RS path.
// NOTE: distinct names from the push-based v3 helpers to avoid ODR clashes within the
// same translation unit (both headers may be included together by python_api.cpp).
inline cudaStream_t get_gemm_rs_pull_comm_stream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(true, at::cuda::current_device());
    return stream.stream();
}

inline cudaEvent_t get_gemm_rs_pull_launched_event() {
    static thread_local cudaEvent_t event = []() -> cudaEvent_t {
        cudaEvent_t evt;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        return evt;
    }();
    return event;
}

inline cudaEvent_t get_gemm_rs_pull_comm_done_event() {
    static thread_local cudaEvent_t event = []() -> cudaEvent_t {
        cudaEvent_t evt;
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&evt, cudaEventDisableTiming));
        return evt;
    }();
    return event;
}

} // anonymous namespace

// ════════════════════════════════════════════════════════════════
//  GEMM compute kernel runtime (Part 1) — 256T, no comm warps
// ════════════════════════════════════════════════════════════════
class SM100BF16GemmRSRuntime final : public LaunchRuntime<SM100BF16GemmRSRuntime> {
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
        // Kernel template parameters:
        //   BLOCK_M, BLOCK_N, BLOCK_K, kNumStages,
        //   kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
        //   kNumMulticast, kIsMulticastOnA,
        //   kSwapAB, kWithAccumulation,
        //   kNumNonEpilogueThreads, kNumEpilogueThreads,
        //   kNumSMs, kNumRanks,
        //   cd_dtype_t, comm_dtype_t
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_rs_impl<
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

// ════════════════════════════════════════════════════════════════
//  统一入口: TRUE Flux PULL-based GEMM + Reduce-Scatter (dual-kernel, overlapped)
// ════════════════════════════════════════════════════════════════
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
                                  const std::string& compiled_dims,
                                  const at::ScalarType& comm_dtype = torch::kBFloat16) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_compute_config(m, n, k, num_sms, static_cast<int>(a.element_size()), num_ranks);

    DG_HOST_ASSERT(config.block_k == 64);
    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // ── Step 0: stream setup ──
    // NOTE 1: we deliberately DO NOT memset the NVLink barrier region per call.
    // `comm::nvlink_barrier` is a self-resetting phase/sign protocol; the barrier region
    // is zeroed once at sym_buffer creation (`buffer.zero_()`) and the +1/-1 sign alternation
    // keeps the signal slots balanced across invocations. A per-call host memset on
    // compute_stream races with the PEER rank's in-flight barrier signal writes (the peer's
    // GEMM may signal our slot before our memset lands, wiping it → barrier deadlock).
    // NOTE 2: no CPU-side cudaStreamSynchronize here. Step 3 makes compute_stream wait on the
    // RS-reduce's comm_done_event, and comm_stream is FIFO-ordered, so the next call's GEMM
    // already orders after this call's reduce on the GPU — a CPU stall would only serialize
    // launches and hurt small shapes.
    const auto comm_stream = get_gemm_rs_pull_comm_stream();
    const auto compute_stream = at::cuda::getCurrentCUDAStream();

    // ── SM carveout for compute/comm overlap ──
    // The GEMM kernel is launched with `num_sms` blocks at 1 block/SM (__launch_bounds__(.,1)),
    // so it saturates every SM. Without a carveout the pull RS-reduce kernel (on comm_stream)
    // cannot become co-resident and only gets scheduled as GEMM blocks retire → ≈ zero overlap
    // (fused ≈ gemm + reduce, serial). Reserve `reduce_sms` SMs for the reduce so both kernels
    // run concurrently: GEMM keeps the tensor cores busy on (num_sms - reduce_sms) SMs while the
    // memory-bound reduce streams remote partials over P2P on the reserved SMs.
    //   DG_RS_REDUCE_SMS=0 → no carveout (gemm=num_sms, reduce=num_sms).
    // NOTE: carveout only pays off once the reduce kernel saturates P2P bandwidth with few SMs
    // (high MLP). With the latency-bound scalar reduce its throughput scales ~linearly with SM
    // count, so a naive carveout regressed; default off until the reduce is bandwidth-bound.
    int reduce_sms = 0;
    if (const char* env = std::getenv("DG_RS_REDUCE_SMS"))
        reduce_sms = std::atoi(env);
    reduce_sms = std::max(0, std::min(reduce_sms, num_sms - static_cast<int>(config.num_multicast)));
    int gemm_sms = num_sms - reduce_sms;
    gemm_sms -= gemm_sms % static_cast<int>(config.num_multicast);  // keep multiple of cluster size
    reduce_sms = num_sms - gemm_sms;

    // ── Create TMA descriptors ──
    // A: token activations [M, K] → TMA 2D load, multicast to 2 CTAs when mc=2
    const auto tensor_map_a = make_tma_2d_desc(a,
                                               k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(a.stride(-2)),
                                               config.swizzle_a_mode);
    // B: weights [N, K] → TMA 2D load
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);

    // ── Step 1: launch GEMM compute kernel on compute_stream ──
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads;

    const SM100BF16GemmRSRuntime::Args gemm_args = {
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
        .launch_args = LaunchArgs(gemm_sms,
                                  total_threads,
                                  config.smem_size,
                                  config.num_multicast)
    };

    const auto gemm_code = SM100BF16GemmRSRuntime::generate(gemm_args);
    const auto gemm_runtime = compiler->build("sm100_bf16_gemm_rs_nt", gemm_code);
    SM100BF16GemmRSRuntime::launch(gemm_runtime, gemm_args);

    // Record event after GEMM kernel launch (compute_stream)
    auto gemm_launched_event = get_gemm_rs_pull_launched_event();
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(gemm_launched_event, compute_stream.stream()));

    // ── Step 2: launch RS reduce (pull) on comm_stream ──
    // comm_stream waits only for GEMM kernel launch (not completion) — overlap happens
    // naturally via per-tile flag polling.
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(comm_stream, gemm_launched_event, 0));

    const int total_tiles = (runtime_m_per_rank + config.block_m - 1) / config.block_m *
                            ((n + config.block_n - 1) / config.block_n);
    // The reduce runs after the GEMM (SMs are free), so oversubscribing SMs with multiple
    // resident reduce blocks adds warps → more outstanding P2P loads per SM → higher effective
    // NVLink read bandwidth (the scalar reduce is latency/concurrency-bound, not BW-bound).
    int reduce_grid_mult = 2;  // 2 resident reduce blocks/SM ≈ best across the shape set
    if (const char* env = std::getenv("DG_RS_REDUCE_MULT"))
        reduce_grid_mult = std::max(1, std::atoi(env));
    const int reduce_base = reduce_sms > 0 ? reduce_sms : num_sms * reduce_grid_mult;
    const int grid_size = std::min(total_tiles, reduce_base);

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
        .launch_args = LaunchArgs(grid_size, 256, 0, 1)
    };

    const auto reduce_code = SM100RSReduceRuntime::generate(reduce_args);
    const auto reduce_runtime = compiler->build("sm100_rs_reduce", reduce_code);

    // Launch RS reduce on comm_stream via direct kernel launch
    // (LaunchRuntime::launch always targets getCurrentCUDAStream, so we bypass it).
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
    auto comm_done_event = get_gemm_rs_pull_comm_done_event();
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(comm_done_event, comm_stream));

    // ── Step 3: compute_stream waits for RS reduce to complete ──
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(compute_stream.stream(), comm_done_event, 0));
}

} // namespace deep_gemm
