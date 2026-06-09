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

// ════════════════════════════════════════════════════════════════
//  Single-kernel Pull-based GEMM + Reduce-Scatter (MegaMoE warp layout + Flux RS scheduling)
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
        GemmRSConfig config;

        layout::SymBuffer<> sym_buffer_ptrs;
        void* output;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Kernel template parameters (new order matching MegaMoE style):
        //   BLOCK_M, BLOCK_N, BLOCK_K, kNumStages,
        //   kSwizzleAMode, kSwizzleBMode, kSwizzleCDMode,
        //   kNumMulticast, kIsMulticastOnA,
        //   kSwapAB, kWithAccumulation,
        //   kNumCommThreads, kNumNonEpilogueThreads, kNumEpilogueThreads,
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
        {}, {}, {},
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
    args.config.num_rs_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
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
//  统一入口: 启动单 kernel GEMM + Pull RS (Blackwell optimized)
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
    auto config = get_gemm_rs_config(m, n, k, num_sms, static_cast<int>(a.element_size()), num_ranks);

    DG_HOST_ASSERT(config.block_k == 64);

    // ── 创建 TMA 描述符 ──
    // A: token activations [M, K] → TMA 2D load, multicast to 2 CTAs
    const auto tensor_map_a = make_tma_2d_desc(a,
                                               k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(a.stride(-2)),
                                               config.swizzle_a_mode);
    // B: weights [N, K] → TMA 2D load, multicast to 2 CTAs
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);

    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // Total threads = comm + non-epilogue + epilogue
    const int total_threads = config.num_rs_threads + config.num_non_epilogue_threads + config.num_epilogue_threads;

    const SM100BF16GemmRSRuntime::Args args = {
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

    const auto code = SM100BF16GemmRSRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_rs_nt", code);
    SM100BF16GemmRSRuntime::launch(runtime, args);
}

} // namespace deep_gemm
