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

namespace deep_gemm {

// JIT runtime for the Ulysses SP post-attn A2A-transpose scatter comm kernel.
class SM100A2ATransposeCommRuntime final : public LaunchRuntime<SM100A2ATransposeCommRuntime> {
public:
    struct Args {
        int num_ranks;
        int bs, nheads, seq, head_dim;
        int tile_m;
        bool set_barrier;
        bool seq_major;
        layout::SymBuffer<> sym_buffer_ptrs;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_a2a_transpose_comm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_a2a_transpose_comm_impl<{}, {}, {}, {}>);
}};
)", args.num_ranks, args.tile_m, args.set_barrier ? "true" : "false", args.seq_major ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sym_buffer_ptrs,
            static_cast<uint32_t>(args.bs),
            static_cast<uint32_t>(args.nheads),
            static_cast<uint32_t>(args.seq),
            static_cast<uint32_t>(args.head_dim)));
    }
};

// Launch the transpose-scatter comm on the current stream. Each rank pushes its attention output
// into every peer's gathered region (hidden-column offset rank*local_hidden) with the seq<->head
// transpose. Caller must barrier across the SP group before reading the gathered buffer (M0).
static void sm100_a2a_transpose_comm(const torch::Tensor& sym_buffer,
                                     const std::vector<int64_t>& sym_buffer_ptrs,
                                     const int& rank_idx,
                                     const int& bs,
                                     const int& nheads,
                                     const int& seq,
                                     const int& head_dim,
                                     const int& tile_m = 128,
                                     const bool& set_barrier = false,
                                     const bool& seq_major_in = false) {
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    // Standalone comm always uses ALL SMs (this is the realistic non-overlap deployment, and the
    // fair "separate" baseline in the bench). The fused path does its own SM carveout instead.
    int num_sms = device_runtime->get_num_sms();
    if (const char* env = std::getenv("DG_A2AT_COMM_ONLY_SMS"))   // sweep-only override
        num_sms = std::min(num_sms, std::max(1, std::atoi(env)));
    // 1024 threads/CTA best saturates per-SM NVLink bandwidth (hides P2P store latency).
    int threads = 1024;
    if (const char* env = std::getenv("DG_A2AT_COMM_THREADS"))
        threads = std::max(32, std::atoi(env));
    // Consume seq-major (BSHD, FlashAttention-native) input directly, so the caller need not
    // .permute(BSHD->BHSD).contiguous() FA's output (that permute is a full HBM pass worth ~1.3x on
    // the post-attn op). Off by default (BHSD contract unchanged); arg wins, env is a sweep override.
    bool seq_major = seq_major_in;
    if (const char* env = std::getenv("DG_A2AT_SEQ_MAJOR"))
        seq_major = std::atoi(env) != 0;

    const SM100A2ATransposeCommRuntime::Args args = {
        .num_ranks = num_ranks,
        .bs = bs, .nheads = nheads, .seq = seq, .head_dim = head_dim,
        .tile_m = tile_m,
        .set_barrier = set_barrier,
        .seq_major = seq_major,
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .launch_args = LaunchArgs(num_sms, threads)
    };

    const auto code = SM100A2ATransposeCommRuntime::generate(args);
    const auto runtime = compiler->build("sm100_a2a_transpose_comm", code);
    SM100A2ATransposeCommRuntime::launch(runtime, args);
}

} // namespace deep_gemm
