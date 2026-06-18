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

#include "../heuristics/gemm_rs_compute.hpp"

namespace deep_gemm {

// ====================================================================
//  Dual-kernel RS Reduce Kernel (Part 2 of GEMM+RS v3)
//  256T/CTA, per-tile polling of ready flags + vectorized reduce
// ====================================================================
class SM100RSReduceRuntime final : public LaunchRuntime<SM100RSReduceRuntime> {
public:
    struct Args {
        int runtime_m_per_rank;
        int n;
        int max_m_per_rank;
        int num_ranks;
        at::ScalarType y_dtype;
        at::ScalarType comm_dtype;
        GemmRSComputeConfig config;
        bool pull_based = false;

        layout::SymBuffer<> sym_buffer_ptrs;
        void* output;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Kernel template parameters:
        //   BLOCK_M, BLOCK_N,
        //   kNumRanks,
        //   cd_dtype_t, comm_dtype_t,
        //   kNumThreads=256,
        //   kPullBased
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_rs_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_rs_reduce_impl<
        {}, {},
        {},
        {},
        {},
        256,
        {}
    >);
}};
)", args.config.block_m, args.config.block_n,
    args.num_ranks,
    to_string(args.y_dtype),
    to_string(args.comm_dtype),
    args.pull_based ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t runtime_m_per_rank = static_cast<uint32_t>(args.runtime_m_per_rank);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_m_per_rank = static_cast<uint32_t>(args.max_m_per_rank);
        void* output = args.output;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            output,
            args.sym_buffer_ptrs,
            runtime_m_per_rank,
            shape_n,
            shape_m_per_rank));
    }
};

// ====================================================================
//  Entry point: launch RS reduce kernel for dual-kernel GEMM+RS
// ====================================================================
static void sm100_rs_reduce(const torch::Tensor& y,
                            const torch::Tensor& sym_buffer,
                            const std::vector<int64_t>& sym_buffer_ptrs,
                            const int& rank_idx,
                            const int& max_m_per_rank,
                            const int& runtime_m_per_rank,
                            const int& n,
                            const at::ScalarType& comm_dtype = torch::kBFloat16) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();
    const auto m = runtime_m_per_rank * num_ranks;
    auto config = get_gemm_rs_compute_config(m, n, 128, num_sms, 2, num_ranks);

    DG_HOST_ASSERT(comm_dtype == torch::kBFloat16 or comm_dtype == torch::kFloat);

    // Grid size = total_tiles = ceil_div(m_per_rank, BLOCK_M) * ceil_div(n, BLOCK_N)
    const int total_tiles = (runtime_m_per_rank + config.block_m - 1) / config.block_m *
                            (n + config.block_n - 1) / config.block_n;

    constexpr int total_threads = 256;

    const SM100RSReduceRuntime::Args args = {
        .runtime_m_per_rank = runtime_m_per_rank,
        .n = n,
        .max_m_per_rank = max_m_per_rank,
        .num_ranks = num_ranks,
        .y_dtype = y.scalar_type(),
        .comm_dtype = comm_dtype,
        .config = config,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .output = y.data_ptr(),
        .launch_args = LaunchArgs(total_tiles,
                                  total_threads,
                                  0,      // no smem needed for reduce kernel
                                  1)      // no multicast
    };

    const auto code = SM100RSReduceRuntime::generate(args);
    const auto runtime = compiler->build("sm100_rs_reduce", code);
    SM100RSReduceRuntime::launch(runtime, args);
}

} // namespace deep_gemm
