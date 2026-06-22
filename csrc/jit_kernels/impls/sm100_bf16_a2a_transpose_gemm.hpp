#pragma once

#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/bf16_a2a_transpose_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/ag_gemm.hpp"
#include "sm100_a2a_transpose_comm.hpp"

namespace deep_gemm {

namespace {

inline cudaStream_t get_a2a_transpose_gemm_comm_stream() {
    static thread_local auto stream = at::cuda::getStreamFromPool(true, at::cuda::current_device());
    return stream.stream();
}
inline cudaEvent_t get_a2a_transpose_gemm_event(int which) {
    static thread_local cudaEvent_t ev[2] = {nullptr, nullptr};
    if (ev[which] == nullptr)
        DG_CUDA_RUNTIME_CHECK(cudaEventCreateWithFlags(&ev[which], cudaEventDisableTiming));
    return ev[which];
}

} // namespace

// GEMM consumer runtime (A = gathered [M, K], per-M-tile barrier wait in Load-A warp).
class SM100BF16A2ATransposeGemmRuntime final : public LaunchRuntime<SM100BF16A2ATransposeGemmRuntime> {
public:
    struct Args {
        int m, n, k;
        int num_sms;          // GEMM SM count (carveout); used as kNumSMs template param
        at::ScalarType d_dtype;
        AGGemmConfig config;
        void* d;
        const int32_t* barrier_ptr;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_a2a_transpose_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_a2a_transpose_gemm_impl<
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.config.num_multicast,
    args.launch_args.grid_dim.first,
    to_string(args.d_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.d,
            static_cast<uint32_t>(args.m),
            static_cast<uint32_t>(args.n),
            static_cast<uint32_t>(args.k),
            args.barrier_ptr,
            args.tensor_map_a,
            args.tensor_map_b,
            args.tensor_map_d));
    }
};

// Fused: transpose-scatter comm (comm_stream, per-tile barrier) overlapped with the Wo GEMM
// consumer (compute_stream, per-tile barrier wait). SM carveout reserves comm SMs so the
// persistent GEMM cannot starve the comm (which would deadlock the per-tile waits).
static void sm100_bf16_a2a_transpose_gemm(const torch::Tensor& d,
                                          const torch::Tensor& gathered,   // A: [M, hidden]
                                          const torch::Tensor& b,          // Wo: [N, hidden]
                                          const torch::Tensor& sym_buffer,
                                          const std::vector<int64_t>& sym_buffer_ptrs,
                                          const int& rank_idx,
                                          const int& bs,
                                          const int& nheads,
                                          const int& seq,
                                          const int& head_dim,
                                          const int& n,
                                          const std::string& compiled_dims) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int total_sms = device_runtime->get_num_sms();
    const int local_seq = seq / num_ranks;
    const int hidden = nheads * head_dim;
    const int m = bs * local_seq;
    const int k = hidden;
    DG_HOST_ASSERT(local_seq % 128 == 0);  // BLOCK_M == comm kTileM == 128

    // SM carveout: reserve comm SMs (default 16, tunable via DG_A2AT_COMM_SMS).
    int comm_sms = 16;
    if (const char* env = std::getenv("DG_A2AT_COMM_SMS")) comm_sms = std::max(1, std::atoi(env));
    comm_sms = std::min(comm_sms, total_sms / 2);

    auto config = get_ag_gemm_config(m, n, k, total_sms, static_cast<int>(b.element_size()));
    DG_HOST_ASSERT(config.block_k == 64 and config.block_m == 128);
    int gemm_sms = total_sms - comm_sms;
    gemm_sms -= gemm_sms % static_cast<int>(config.num_multicast);
    DG_HOST_ASSERT(gemm_sms > 0);

    const layout::BF16A2ATransposeGemmWorkspace ws(nullptr, num_ranks, bs, nheads, seq, head_dim);
    auto* barrier_ptr = reinterpret_cast<int32_t*>(math::advance_ptr(
        sym_buffer.data_ptr(), reinterpret_cast<uintptr_t>(ws.get_barrier_ptr())));

    const auto current_stream = at::cuda::getCurrentCUDAStream();
    const auto comm_stream = get_a2a_transpose_gemm_comm_stream();
    const auto ready_event = get_a2a_transpose_gemm_event(0);
    const auto comm_done_event = get_a2a_transpose_gemm_event(1);

    // comm_stream waits until the input is written on the compute stream.
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(ready_event, current_stream.stream()));
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(comm_stream, ready_event, 0));

    // ── launch transpose-scatter comm on comm_stream (set_barrier=true) ──
    {
        const SM100A2ATransposeCommRuntime::Args comm_args = {
            .num_ranks = num_ranks,
            .bs = bs, .nheads = nheads, .seq = seq, .head_dim = head_dim,
            .tile_m = static_cast<int>(config.block_m),
            .set_barrier = true,
            .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
            .launch_args = LaunchArgs(comm_sms, 256)
        };
        const auto comm_code = SM100A2ATransposeCommRuntime::generate(comm_args);
        const auto comm_runtime = compiler->build("sm100_a2a_transpose_comm", comm_code);
        const auto comm_kernel = comm_runtime->kernel;
        const dim3 grid_dim = {static_cast<unsigned>(comm_sms), 1, 1};
        const dim3 block_dim = {256, 1, 1};
        auto cfg = construct_launch_config(comm_kernel, comm_stream, 0, grid_dim, block_dim, 1, false);
        auto sb = comm_args.sym_buffer_ptrs;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(comm_kernel, cfg, sb,
            static_cast<uint32_t>(bs), static_cast<uint32_t>(nheads),
            static_cast<uint32_t>(seq), static_cast<uint32_t>(head_dim)));
    }
    DG_CUDA_RUNTIME_CHECK(cudaEventRecord(comm_done_event, comm_stream));

    // ── GEMM consumer on compute (current) stream, gemm_sms blocks ──
    const auto tensor_map_a = make_tma_2d_desc(gathered, k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(gathered.stride(-2)), config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b, k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)), config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_2d_desc(d,
                                               static_cast<int>(d.size(-1)), static_cast<int>(d.size(-2)),
                                               config.swizzle_cd_mode / static_cast<int>(d.element_size()), config.block_m,
                                               static_cast<int>(d.stride(-2)), config.swizzle_cd_mode);

    const SM100BF16A2ATransposeGemmRuntime::Args gemm_args = {
        .m = m, .n = n, .k = k,
        .num_sms = gemm_sms,
        .d_dtype = d.scalar_type(),
        .config = config,
        .d = d.data_ptr(),
        .barrier_ptr = barrier_ptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .launch_args = LaunchArgs(gemm_sms,
                                  config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, config.num_multicast)
    };
    const auto gemm_code = SM100BF16A2ATransposeGemmRuntime::generate(gemm_args);
    const auto gemm_runtime = compiler->build("sm100_bf16_a2a_transpose_gemm", gemm_code);
    SM100BF16A2ATransposeGemmRuntime::launch(gemm_runtime, gemm_args);

    // ensure comm finished (buffer not freed early) before returning.
    DG_CUDA_RUNTIME_CHECK(cudaStreamWaitEvent(current_stream.stream(), comm_done_event, 0));
}

} // namespace deep_gemm
