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

#include <deep_gemm/layout/fused_qkv_norm_a2a.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/gemm_rs_compute.hpp"

namespace deep_gemm::fused_qkv_norm_a2a {

struct FusedQKVNormA2AScatterMaps {
    CUtensorMap maps[8];
};

static CUtensorMap make_scatter_tma_2d_desc(
        void* gmem_ptr, const at::ScalarType& dtype, const int& elem_size,
        int gmem_inner_dim, int gmem_outer_dim,
        int smem_inner_dim, int smem_outer_dim,
        const int& gmem_outer_stride,
        const int& swizzle_mode, const int& swizzle_base = 0) {
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;
    CUtensorMap tensor_map;
    const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                      static_cast<cuuint64_t>(gmem_outer_dim)};
    const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                     static_cast<cuuint32_t>(smem_outer_dim)};
    const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size)};
    const cuuint32_t elem_strides[2] = {1, 1};
    DG_CUDA_DRIVER_CHECK(lazy_cuTensorMapEncodeTiled(
        &tensor_map, aten_dtype_to_tensor_map_dtype(dtype, false, true),
        2, gmem_ptr, gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

// Runtime class for JIT compilation (same pattern as SM100BF16GemmA2ATransposeRuntime)
class SM100BF16FusedQKVNormA2ARuntime final : public LaunchRuntime<SM100BF16FusedQKVNormA2ARuntime> {
public:
    struct Args {
        int m, n, k;
        int bs, seq, local_seq;
        int q_dim, kv_dim, local_q_n, local_kv_n;
        int num_ranks;
        float eps;
        bool do_norm_q, do_norm_k;
        at::ScalarType cd_dtype;
        at::ScalarType comm_dtype;
        GemmRSComputeConfig config;

        float* sum_buffer;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        FusedQKVNormA2AScatterMaps scatter_maps;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_fused_qkv_norm_a2a.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_fused_qkv_norm_a2a_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {}, {},
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
    args.do_norm_q ? "true" : "false",
    args.do_norm_k ? "true" : "false",
    to_string(args.cd_dtype),
    to_string(args.comm_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        uint32_t shape_m = static_cast<uint32_t>(args.m);
        uint32_t shape_n = static_cast<uint32_t>(args.n);
        uint32_t shape_k = static_cast<uint32_t>(args.k);
        uint32_t bs = static_cast<uint32_t>(args.bs);
        uint32_t seq = static_cast<uint32_t>(args.seq);
        uint32_t local_seq = static_cast<uint32_t>(args.local_seq);
        uint32_t q_dim = static_cast<uint32_t>(args.q_dim);
        uint32_t kv_dim = static_cast<uint32_t>(args.kv_dim);
        uint32_t local_q_n = static_cast<uint32_t>(args.local_q_n);
        uint32_t local_kv_n = static_cast<uint32_t>(args.local_kv_n);
        float eps = args.eps;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            shape_m, shape_n, shape_k,
            bs, seq, local_seq,
            q_dim, kv_dim, local_q_n, local_kv_n,
            eps,
            args.sum_buffer,
            args.tensor_map_a,
            args.tensor_map_b,
            args.sym_buffer_ptrs,
            args.scatter_maps));
    }
};

// ════════════════════════════════════════════════════════════════
//  Entry point: single-kernel GEMM + x²sum + scatter + rms scatter
//
//  a: [bs*local_seq, K] bf16 (local seq shard, full hidden)
//  b: [N, K] bf16, NT layout, N = q_dim + 2*kv_dim
//  sym_buffer: symmetric buffer (holds rms region + output region)
//  sum_buffer: [bs*local_seq, 2] fp32 (local, for x² partial sum)
// ════════════════════════════════════════════════════════════════
static void sm100_bf16_fused_qkv_norm_a2a_nt(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const torch::Tensor& sum_buffer,
    const int& rank_idx,
    const int& bs,
    const int& local_seq,
    const int& q_nheads,
    const int& kv_nheads,
    const int& head_dim,
    const float& eps,
    const bool& do_norm_q,
    const bool& do_norm_k) {

#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() > 1);

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int local_m = bs * local_seq;
    const int q_dim = q_nheads * head_dim;
    const int kv_dim = kv_nheads * head_dim;
    const int n_total = q_dim + 2 * kv_dim;
    const int seq = local_seq * num_ranks;
    const int local_q_n = (q_nheads / num_ranks) * head_dim;
    const int local_kv_n = (kv_nheads / num_ranks) * head_dim;
    const int local_n_total = local_q_n + 2 * local_kv_n;

    // Alignment checks
    DG_HOST_ASSERT(local_seq % 128 == 0);
    DG_HOST_ASSERT(q_dim % 128 == 0);
    DG_HOST_ASSERT(kv_dim % 128 == 0);
    DG_HOST_ASSERT(local_q_n % 128 == 0);
    DG_HOST_ASSERT(local_kv_n % 128 == 0);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(a.size(0) == local_m and a.size(1) == b.size(1));
    DG_HOST_ASSERT(b.size(0) == n_total);

    // Compute config (reuse GEMM-RS heuristic with num_ranks=1)
    const auto num_sms = device_runtime->get_num_sms();
    auto config = get_gemm_rs_compute_config(local_m, n_total, static_cast<int>(a.size(1)),
                                              num_sms, 2, 1);

    // Build TMA descriptors for A and B
    const auto tensor_map_a = make_tma_2d_desc(a,
        static_cast<int>(a.size(1)), local_m,
        config.block_k, config.load_block_m,
        static_cast<int>(a.stride(-2)), config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_2d_desc(b,
        static_cast<int>(b.size(1)), n_total,
        config.block_k, config.load_block_n,
        static_cast<int>(b.stride(-2)), config.swizzle_b_mode);

    // Build scatter_maps: one per dst_rank, targeting dst's OUTPUT region
    // (after rms region in sym buffer)
    const int elem_size = 2;  // bf16
    const auto workspace = layout::FusedQKVNormA2AWorkspace(
        nullptr, num_ranks, bs, seq, local_n_total, elem_size);
    const int64_t out_offset = static_cast<int64_t>(
        layout::FusedQKVNormA2AWorkspace::kNumBarrierSignalBytes + workspace.get_rms_region_bytes());

    FusedQKVNormA2AScatterMaps scatter_maps{};
    for (int d = 0; d < num_ranks; ++d) {
        auto* out_ptr = reinterpret_cast<char*>(static_cast<uintptr_t>(sym_buffer_ptrs[d])) + out_offset;
        scatter_maps.maps[d] = make_scatter_tma_2d_desc(
            reinterpret_cast<void*>(out_ptr), torch::kBFloat16, elem_size,
            local_n_total, bs * seq,
            config.swizzle_cd_mode / elem_size, config.block_m,
            local_n_total,
            config.swizzle_cd_mode);
    }

    // Launch args
    const int total_threads = config.num_non_epilogue_threads + config.num_epilogue_threads;
    const SM100BF16FusedQKVNormA2ARuntime::Args args = {
        .m = local_m, .n = n_total, .k = static_cast<int>(a.size(1)),
        .bs = bs, .seq = seq, .local_seq = local_seq,
        .q_dim = q_dim, .kv_dim = kv_dim, .local_q_n = local_q_n, .local_kv_n = local_kv_n,
        .num_ranks = num_ranks,
        .eps = eps,
        .do_norm_q = do_norm_q, .do_norm_k = do_norm_k,
        .cd_dtype = torch::kBFloat16,
        .comm_dtype = torch::kBFloat16,
        .config = config,
        .sum_buffer = reinterpret_cast<float*>(sum_buffer.data_ptr()),
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .scatter_maps = scatter_maps,
        .launch_args = LaunchArgs(num_sms, total_threads, config.smem_size, config.num_multicast)
    };

    // Zero sum_buffer before kernel launch
    // sum_buffer is [bs*seq, 2] (indexed by output M row, which can be up to bs*seq-1)
    AT_CUDA_CHECK(cudaMemsetAsync(sum_buffer.data_ptr(), 0, sum_buffer.nbytes(),
                                   at::cuda::getCurrentCUDAStream()));

    const auto code = SM100BF16FusedQKVNormA2ARuntime::generate(args);
    const auto runtime = compiler->build("sm100_bf16_fused_qkv_norm_a2a_nt", code);
    SM100BF16FusedQKVNormA2ARuntime::launch(runtime, args);
#else
    DG_HOST_UNREACHABLE("Fused QKV+Norm+A2A requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("sm100_bf16_fused_qkv_norm_a2a_nt", &sm100_bf16_fused_qkv_norm_a2a_nt,
          pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("sym_buffer"), pybind11::arg("sym_buffer_ptrs"),
          pybind11::arg("sum_buffer"),
          pybind11::arg("rank_idx"),
          pybind11::arg("bs"), pybind11::arg("local_seq"),
          pybind11::arg("q_nheads"), pybind11::arg("kv_nheads"), pybind11::arg("head_dim"),
          pybind11::arg("eps"), pybind11::arg("do_norm_q"), pybind11::arg("do_norm_k"));
#endif
}

} // namespace deep_gemm::fused_qkv_norm_a2a
