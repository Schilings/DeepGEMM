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
//  全融合 GEMM + Push + Reduce 单 kernel runtime
// ════════════════════════════════════════════════════════════════
class SM100BF16GemmRSFusedRuntime final : public LaunchRuntime<SM100BF16GemmRSFusedRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int m, n, k;
        int num_ranks;
        int num_reduce_threads;
        at::ScalarType y_dtype;
        at::ScalarType comm_dtype;
        GemmRSConfig config;

        void* y;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs_fused.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_gemm_rs_fused_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
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
    args.num_reduce_threads,
    args.launch_args.grid_dim.first, args.num_ranks,
    to_string(args.y_dtype),
    to_string(args.comm_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        void* output_ptr = args.y;
        uint32_t shape_m_per_rank = static_cast<uint32_t>(args.max_m_per_rank);
        uint32_t runtime_m_per_rank = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            output_ptr,
            shape_m_per_rank,
            runtime_m_per_rank,
            shape_n,
            shape_k,
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_b));
    }
};

// ════════════════════════════════════════════════════════════════
//  统一入口: 单 kernel 完成 GEMM + Push + Reduce
// ════════════════════════════════════════════════════════════════
static void sm100_bf16_gemm_rs_fused(const torch::Tensor& y,
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
                                     const at::ScalarType& comm_dtype = torch::kBFloat16,
                                     const bool& reduce_in_fp32 = true) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_config(m, n, k, num_sms, static_cast<int>(a.element_size()), num_ranks);

    DG_HOST_ASSERT(config.block_k == 64);

    // ── Reduce threads: 4 warps = 128 threads dedicated to reduce ──
    constexpr int kNumReduceThreads = 128;
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads + kNumReduceThreads;

    // ── 计算 smem size ──
    // 融合版与分离版共用同一套 smem（reduce warps 不需要额外 smem）
    // smem 需求与原始 GEMM kernel 相同
    const int smem_size = config.smem_size;

    // ── 创建 TMA 描述符 ──
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

    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    const SM100BF16GemmRSFusedRuntime::Args args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .m = m, .n = n, .k = k,
        .num_ranks = num_ranks,
        .num_reduce_threads = kNumReduceThreads,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .y = y.data_ptr(),
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .launch_args = LaunchArgs(num_sms,
                                  total_threads,
                                  smem_size,
                                  config.num_multicast,
                                  /*enable_pdl=*/false)
    };

    const auto code = SM100BF16GemmRSFusedRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_gemm_rs_fused", code);
    SM100BF16GemmRSFusedRuntime::launch(runtime, args);
}

} // namespace deep_gemm
