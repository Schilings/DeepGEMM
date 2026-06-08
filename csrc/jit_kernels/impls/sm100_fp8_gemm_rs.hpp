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
//  阶段1: FP8 GEMM + NVLink Push kernel runtime
// ════════════════════════════════════════════════════════════════
class SM100FP8GemmRSRuntime final : public LaunchRuntime<SM100FP8GemmRSRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int m, n, k;
        int num_ranks;
        int gran_k;
        at::ScalarType y_dtype;
        at::ScalarType comm_dtype;  // Communication data type for partial buffer
        GemmRSConfig config;

        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfb;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_gemm_rs.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_rs_nt_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
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
    args.gran_k,
    to_string(args.y_dtype),
    to_string(args.comm_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t shape_m_per_rank = static_cast<uint32_t>(args.max_m_per_rank);
        uint32_t runtime_m_per_rank = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            shape_m_per_rank,
            runtime_m_per_rank,
            shape_n,
            shape_k,
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_sfa,
            args.tensor_map_b,
            args.tensor_map_sfb));
    }
};

// ════════════════════════════════════════════════════════════════
//  阶段2: Reduce Epilogue kernel runtime (复用 BF16 版本的 reduce kernel)
// ════════════════════════════════════════════════════════════════
class SM100FP8ReduceEpilogueRuntime final : public LaunchRuntime<SM100FP8ReduceEpilogueRuntime> {
public:
    struct Args {
        int runtime_m_per_rank;
        int n;
        int num_ranks;
        int max_m_per_rank;
        int num_sms;
        at::ScalarType y_dtype;
        at::ScalarType comm_dtype;  // Communication data type (what's in the partial buffer)
        bool reduce_in_fp32;        // Whether to accumulate in FP32 during reduce
        GemmRSConfig config;

        void* y;
        void* workspace_base;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // NOTE: We reuse the reduce epilogue kernel defined in sm100_bf16_gemm_rs.cuh
        // It's fully templated and works for any comm_dtype_t / cd_dtype_t combination
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_gemm_rs.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_reduce_epilogue_impl<
        {}, {},
        {}, {},
        {},
        {},
        {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n,
    args.launch_args.grid_dim.first, args.num_ranks,
    args.config.reduce_num_threads,
    to_string(args.y_dtype),
    to_string(args.comm_dtype),
    args.reduce_in_fp32 ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        void* output_ptr = args.y;
        uint32_t runtime_m = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        void* workspace = args.workspace_base;
        uint32_t max_m = static_cast<uint32_t>(args.max_m_per_rank);
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            output_ptr, runtime_m, shape_n, workspace, max_m));
    }
};

// ════════════════════════════════════════════════════════════════
//  统一入口: 启动 FP8 GEMM + Push kernel，然后 PDL 启动 Reduce kernel
// ════════════════════════════════════════════════════════════════
static void sm100_fp8_gemm_rs_nt(const torch::Tensor& y,
                                 const torch::Tensor& a,
                                 const torch::Tensor& a_sf,
                                 const torch::Tensor& b,
                                 const torch::Tensor& b_sf,
                                 const torch::Tensor& sym_buffer,
                                 const std::vector<int64_t>& sym_buffer_ptrs,
                                 const int& rank_idx,
                                 const int& max_m_per_rank,
                                 const int& runtime_m_per_rank,
                                 const int& gran_k,
                                 const int& n,
                                 const int& k,
                                 const std::string& compiled_dims,
                                 const at::ScalarType& comm_dtype = torch::kBFloat16,
                                 const bool& reduce_in_fp32 = true) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_config(m, n, k, num_sms, static_cast<int>(a.element_size()));

    DG_HOST_ASSERT(config.block_k == 128);

    // ── 创建 TMA 描述符 (A, SFA, B, SFB) ──
    const auto tensor_map_a = make_tma_2d_desc(a,
                                               k, m,
                                               config.block_k, config.load_block_m,
                                               static_cast<int>(a.stride(-2)),
                                               config.swizzle_a_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, a_sf,
                                                 m, k,
                                                 config.block_m, gran_k,
                                                 1, 0);
    const auto tensor_map_b = make_tma_2d_desc(b,
                                               k, n,
                                               config.block_k, config.load_block_n,
                                               static_cast<int>(b.stride(-2)),
                                               config.swizzle_b_mode);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, b_sf,
                                                 n, k,
                                                 config.block_n, gran_k,
                                                 1, 0);

    // Validate comm_dtype
    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // ── 阶段1: FP8 GEMM + Push kernel ──
    const SM100FP8GemmRSRuntime::Args gemm_args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .m = m, .n = n, .k = k,
        .num_ranks = num_ranks,
        .gran_k = gran_k,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfb = tensor_map_sfb,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size,
                                  config.num_multicast)
    };

    const auto gemm_code = SM100FP8GemmRSRuntime::generate(gemm_args);
    const auto gemm_runtime = compiler->build("sm100_fp8_gemm_rs_nt", gemm_code);
    SM100FP8GemmRSRuntime::launch(gemm_runtime, gemm_args);

    // ── 阶段2: Reduce Epilogue kernel (PDL 依赖启动) ──
    void* workspace_base = sym_buffer.data_ptr();

    const SM100FP8ReduceEpilogueRuntime::Args reduce_args = {
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n,
        .num_ranks = num_ranks,
        .max_m_per_rank = max_m_per_rank,
        .num_sms = num_sms,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .reduce_in_fp32 = reduce_in_fp32,
        .config = config,
        .y = y.data_ptr(),
        .workspace_base = workspace_base,
        .launch_args = LaunchArgs(num_sms,
                                  config.reduce_num_threads,
                                  0,   // reduce kernel 不需要 shared memory
                                  1)   // cluster_dim = 1
    };

    const auto reduce_code = SM100FP8ReduceEpilogueRuntime::generate(reduce_args);
    const auto reduce_runtime = compiler->build("sm100_fp8_reduce_epilogue", reduce_code);
    SM100FP8ReduceEpilogueRuntime::launch(reduce_runtime, reduce_args);
}

} // namespace deep_gemm
