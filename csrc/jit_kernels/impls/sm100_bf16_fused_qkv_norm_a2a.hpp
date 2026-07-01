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

namespace deep_gemm::fused_qkv_norm_a2a {

// Scatter maps for QKV (GQA-aware): one descriptor per dst_rank, targeting
// dst's output buffer as 2D [bs*seq, local_n_total] with NO swizzle (simpler, Phase 2 v1).
struct FusedQKVNormA2AScatterMaps {
    CUtensorMap maps[8];
};

// Build a non-swizzled 2D TMA descriptor for the scatter target (dst's output buffer).
static CUtensorMap make_scatter_tma_2d_desc_noswizzle(
        void* gmem_ptr, const at::ScalarType& dtype, const int& elem_size,
        int gmem_inner_dim, int gmem_outer_dim,
        int smem_inner_dim, int smem_outer_dim,
        const int& gmem_outer_stride) {
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
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensor_map;
}

// Build a non-swizzled 2D TMA descriptor for loading from local buffer.
static CUtensorMap make_local_load_tma_2d_desc(
        void* gmem_ptr, const at::ScalarType& dtype, const int& elem_size,
        int gmem_inner_dim, int gmem_outer_dim,
        int smem_inner_dim, int smem_outer_dim,
        const int& gmem_outer_stride) {
    return make_scatter_tma_2d_desc_noswizzle(gmem_ptr, dtype, elem_size,
                                               gmem_inner_dim, gmem_outer_dim,
                                               smem_inner_dim, smem_outer_dim,
                                               gmem_outer_stride);
}

// ════════════════════════════════════════════════════════════════
//  Kernel 2: RMSNorm + A2A-transpose-scatter
//  Reads local GEMM output, applies RMSNorm (optional on Q/K), scatters to peers.
// ════════════════════════════════════════════════════════════════

static void sm100_bf16_rmsnorm_a2a_scatter(
    const torch::Tensor& local_buffer,    // [bs*local_seq, N_total] bf16 (Kernel 1 output)
    const torch::Tensor& sym_buffer,      // symmetric output buffer
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& bs,
    const int& local_seq,
    const int& q_dim,
    const int& kv_dim,
    const int& q_nheads,
    const int& kv_nheads,
    const int& head_dim,
    const float& eps,
    const c10::optional<torch::Tensor>& norm_q_weight,  // [q_dim] fp32, or None
    const c10::optional<torch::Tensor>& norm_k_weight,  // [kv_dim] fp32, or None
    const torch::Tensor& sum_buffer) {                  // [bs*local_seq, 2] fp32

    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_sms = device_runtime->get_num_sms();

    const int local_m = bs * local_seq;
    const int n_total = q_dim + 2 * kv_dim;  // [Q | K | V]
    const int seq = local_seq * num_ranks;

    const int local_q_nheads = q_nheads / num_ranks;
    const int local_kv_nheads = kv_nheads / num_ranks;
    const int local_q_n = local_q_nheads * head_dim;
    const int local_kv_n = local_kv_nheads * head_dim;
    const int local_n_total = local_q_n + 2 * local_kv_n;

    // Assert alignment
    DG_HOST_ASSERT(local_seq % 128 == 0);  // BLOCK_M alignment
    DG_HOST_ASSERT(q_dim % 128 == 0);      // Q segment divisible by BLOCK_N
    DG_HOST_ASSERT(kv_dim % 128 == 0);     // K/V segment divisible by BLOCK_N
    DG_HOST_ASSERT(local_q_n % 128 == 0 || local_q_n == 0);
    DG_HOST_ASSERT(local_kv_n % 128 == 0 || local_kv_n == 0);

    // Determine norm flags
    const bool do_norm_q = norm_q_weight.has_value();
    const bool do_norm_k = norm_k_weight.has_value();
    if (do_norm_q) {
        DG_HOST_ASSERT(norm_q_weight.value().scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(norm_q_weight.value().numel() == q_dim);
    }
    if (do_norm_k) {
        DG_HOST_ASSERT(norm_k_weight.value().scalar_type() == torch::kFloat);
        DG_HOST_ASSERT(norm_k_weight.value().numel() == kv_dim);
    }

    // ── Build scatter_maps: one 2D TMA descriptor per dst_rank ──
    // Target: dst's output buffer [bs*seq, local_n_total], no swizzle, row-major
    const int elem_size = 2;  // bf16
    const int64_t out_offset = static_cast<int64_t>(layout::FusedQKVNormA2AWorkspace::kNumBarrierSignalBytes);

    FusedQKVNormA2AScatterMaps scatter_maps{};
    for (int d = 0; d < num_ranks; ++d) {
        auto* out_ptr = reinterpret_cast<char*>(static_cast<uintptr_t>(sym_buffer_ptrs[d])) + out_offset;
        // 2D descriptor: inner=local_n_total, outer=bs*seq, stride=local_n_total, no swizzle
        // smem dims match tile: BLOCK_N=128, BLOCK_M=128
        scatter_maps.maps[d] = make_scatter_tma_2d_desc_noswizzle(
            reinterpret_cast<void*>(out_ptr), torch::kBFloat16, elem_size,
            local_n_total, bs * seq,
            128, 128,  // smem_inner, smem_outer (tile size)
            local_n_total);
    }

    // ── Build local load TMA descriptor ──
    // local_buffer: [local_m, n_total], row-major, stride=n_total
    CUtensorMap tensor_map_local = make_local_load_tma_2d_desc(
        local_buffer.data_ptr(), torch::kBFloat16, elem_size,
        n_total, local_m,
        128, 128,  // smem tile dims
        n_total);

    // ── Launch Kernel 2 ──
    // For Phase 2 v1, use NVRTC JIT (not pre-compiled)
    // The kernel is launched with 128 threads, num_sms blocks (persistent)
    const int block_m = 128, block_n = 128;
    const int num_m_blocks = (local_m + block_m - 1) / block_m;
    const int num_n_blocks = (n_total + block_n - 1) / block_n;
    const int num_tiles = num_m_blocks * num_n_blocks;
    const int grid_size = std::min(num_tiles, num_sms);

    // Build kernel name based on template params (norm flags)
    std::string norm_q_str = do_norm_q ? "true" : "false";
    std::string norm_k_str = do_norm_k ? "true" : "false";

    // JIT compile and launch
    auto code = fmt::format(R"(
#include <deep_gemm/impls/sm100_bf16_rmsnorm_a2a_scatter.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_rmsnorm_a2a_scatter_impl<
        128, 128, 128, 2, {}, {}, {}, cutlass::bfloat16_t>);
}};
)", num_ranks, norm_q_str, norm_k_str);

    // For now, use direct CUDA launch via the compiled kernel
    // Note: In the actual implementation, this goes through the JIT compiler
    // For Phase 2 v1, we'll use a simpler approach via torch extension

    // Actually, let's use the JIT compiler infrastructure
    auto runtime = compiler->build("sm100_bf16_rmsnorm_a2a_scatter", code);

    // Launch parameters
    const int num_threads = 128;
    const int smem_size = 128 * 128 * 2 * 2 + 256;  // 2 stages × 128×128 bf16 + barriers

    // Launch via the runtime
    // (This is simplified — the actual LaunchRuntime handles grid/thread config)
    // For now, we'll call the kernel directly via CUDA
    // ... (actual launch code omitted — needs LaunchRuntime integration)

    // Placeholder: the actual kernel launch will be done via the JIT runtime
    // similar to SM100BF16GemmA2ATransposeRuntime
}

// ════════════════════════════════════════════════════════════════
//  Full operator: Kernel 1 (GEMM + local write + x² sum) + Kernel 2 (norm + scatter)
// ════════════════════════════════════════════════════════════════

static void bf16_fused_qkv_norm_a2a_transpose_nt(
    const torch::Tensor& a,                    // [bs*local_seq, K] bf16
    const torch::Tensor& b,                    // [N_total, K] bf16, NT layout
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx,
    const int& bs,
    const int& local_seq,
    const int& q_nheads,
    const int& kv_nheads,
    const int& head_dim,
    const float& eps,
    const c10::optional<torch::Tensor>& norm_q_weight,
    const c10::optional<torch::Tensor>& norm_k_weight,
    const c10::optional<torch::Tensor>& bias) {

#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const int local_m = bs * local_seq;
    const int q_dim = q_nheads * head_dim;
    const int kv_dim = kv_nheads * head_dim;
    const int n_total = q_dim + 2 * kv_dim;

    // ── Step 1: GEMM + bias → local_buffer + x² sum ──
    // Allocate local buffer and sum buffer
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(a.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat).device(a.device());

    torch::Tensor local_buffer = torch::empty({local_m, n_total}, opts_bf16);
    torch::Tensor sum_buffer = torch::zeros({local_m, 2}, opts_fp32);

    // GEMM: d = a @ b^T
    // For Phase 2 v1, use deep_gemm.bf16_gemm_nt (called from Python)
    // The actual fused kernel will replace this
    // (This host function is the C++ entry point; Python wrapper handles the GEMM call)

    // ... (Kernel 1 launch — GEMM with local write + x² sum)
    // ... (Kernel 2 launch — norm + scatter)

    // For Phase 2 v1, the actual CUDA kernel launches are deferred to the Python wrapper
    // which orchestrates: bf16_gemm_nt + custom norm_sum kernel + rmsnorm_a2a_scatter kernel

    sm100_bf16_rmsnorm_a2a_scatter(
        local_buffer, sym_buffer, sym_buffer_ptrs, rank_idx,
        bs, local_seq, q_dim, kv_dim, q_nheads, kv_nheads, head_dim,
        eps, norm_q_weight, norm_k_weight, sum_buffer);
#else
    DG_HOST_UNREACHABLE("Fused QKV+Norm+A2A requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    // APIs will be registered as we implement them
    // m.def("bf16_fused_qkv_norm_a2a_transpose_nt", &bf16_fused_qkv_norm_a2a_transpose_nt, ...);
#endif
}

} // namespace deep_gemm::fused_qkv_norm_a2a
