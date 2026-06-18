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
//  TRUE Flux PULL-based RS Reduce kernel (Part 2 of GEMM+RS)
//  256T/CTA, per-tile polling of REMOTE flags + vectorized P2P pull reduce
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

        layout::SymBuffer<> sym_buffer_ptrs;
        void* output;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // Kernel template parameters:
        //   BLOCK_M, BLOCK_N, kNumRanks, cd_dtype_t, comm_dtype_t, kNumThreads=256
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_rs_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_rs_reduce_impl<
        {}, {},
        {},
        {},
        {},
        256
    >);
}};
)", args.config.block_m, args.config.block_n,
    args.num_ranks,
    to_string(args.y_dtype),
    to_string(args.comm_dtype));
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

} // namespace deep_gemm
