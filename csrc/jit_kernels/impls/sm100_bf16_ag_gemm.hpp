#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/bf16_ag_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/ag_gemm.hpp"

namespace deep_gemm {

class SM100BF16AGGemmRuntime final : public LaunchRuntime<SM100BF16AGGemmRuntime> {
public:
    struct Args {
        int max_m_per_rank;
        int runtime_m_per_rank;
        int n, k;
        int num_slots;
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
#include <deep_gemm/impls/sm100_bf16_ag_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_ag_gemm_nt_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}
    >);
}};
)", args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.num_stages,
    args.config.num_ag_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
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
            args.sym_buffer_ptrs,
            args.tensor_map_a,
            args.tensor_map_b,
            args.tensor_map_d));
    }
};

static void sm100_bf16_ag_gemm_nt(const torch::Tensor& d,
                                  const torch::Tensor& slots_x,
                                  const torch::Tensor& b,
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

    const SM100BF16AGGemmRuntime::Args args = {
        .max_m_per_rank = max_m_per_rank,
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n, .k = k,
        .num_slots = num_slots,
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

    const auto code = SM100BF16AGGemmRuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_ag_gemm_nt", code);
    SM100BF16AGGemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
