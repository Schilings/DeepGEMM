#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "layout.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_gemm_rs_v2.hpp"
#endif

#include <deep_gemm/layout/gemm_rs.cuh>


namespace deep_gemm::gemm_rs {

static int get_token_alignment_for_gemm_rs() {
    return 128;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_gemm_rs(const int& num_ranks,
                                 const int& num_max_tokens_per_rank,
                                 const int& hidden,
                                 const bool& use_fp32_output) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank % get_token_alignment_for_gemm_rs() == 0);
    DG_HOST_ASSERT(hidden % 128 == 0);
    const int elem_size = use_fp32_output ? 4 : 2;
    const auto workspace = layout::GemmRSWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, elem_size);

    auto slice_buffers = [=](const torch::Tensor& buffer) {
        const auto workspace = layout::GemmRSWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, elem_size);
        auto partial = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_partial_ptr())),
            {num_ranks, num_max_tokens_per_rank, hidden},
            use_fp32_output ? torch::TensorOptions().dtype(torch::kFloat).device(buffer.device())
                            : torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto ready = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_ready_ptr())),
            {num_ranks, workspace.get_num_m_blocks_per_rank(), workspace.get_num_n_blocks()},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        return std::make_tuple(partial, ready);
    };
    return {static_cast<int64_t>(workspace.get_num_bytes()), slice_buffers};
}

// ════════════════════════════════════════════════════════════════
//  Pull-based Single-kernel GEMM + Reduce-Scatter
// ════════════════════════════════════════════════════════════════
static void bf16_gemm_rs_v2_nt(const torch::Tensor& y,
                               const torch::Tensor& a,
                               const torch::Tensor& b,
                               const torch::Tensor& sym_buffer,
                               const std::vector<int64_t>& sym_buffer_ptrs,
                               const int& rank_idx,
                               const int& num_max_tokens_per_rank,
                               const int& num_tokens_per_rank,
                               const std::string& compiled_dims,
                               const std::string& comm_dtype_str = "bf16") {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < static_cast<int>(sym_buffer_ptrs.size()));
    DG_HOST_ASSERT(num_tokens_per_rank > 0 and num_tokens_per_rank <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_tokens_per_rank % get_token_alignment_for_gemm_rs() == 0);

    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    const auto [m, k] = get_shape<2>(a);
    const auto [n, k_] = get_shape<2>(b);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(m == num_tokens_per_rank * num_ranks and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    const auto [ym, yn] = get_shape<2>(y);
    DG_HOST_ASSERT(ym == num_tokens_per_rank and yn == n);
    DG_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 or y.scalar_type() == torch::kFloat);
    check_major_type_cd(y);
    DG_HOST_ASSERT(n % 128 == 0 and k % 64 == 0);

    // Parse communication dtype
    at::ScalarType comm_dtype;
    if (comm_dtype_str == "bf16" or comm_dtype_str == "bfloat16") {
        comm_dtype = torch::kBFloat16;
    } else if (comm_dtype_str == "fp32" or comm_dtype_str == "float32") {
        comm_dtype = torch::kFloat;
    } else {
        DG_HOST_UNREACHABLE("Unsupported comm_dtype: must be 'bf16' or 'fp32'");
    }

    const bool use_fp32_comm = (comm_dtype == torch::kFloat);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_gemm_rs(
        num_ranks, num_max_tokens_per_rank, n, use_fp32_comm);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));

    sm100_bf16_gemm_rs_v2_nt(y, a, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                             num_max_tokens_per_rank, num_tokens_per_rank, n, k, compiled_dims,
                             comm_dtype);
#else
    DG_HOST_UNREACHABLE("BF16 GEMM+RS requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_gemm_rs", &get_token_alignment_for_gemm_rs);
    m.def("get_symm_buffer_size_for_gemm_rs", &get_symm_buffer_size_for_gemm_rs);
    m.def("bf16_gemm_rs_v2_nt", &bf16_gemm_rs_v2_nt,
          pybind11::arg("y"), pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sym_buffer"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_tokens_per_rank"),
          pybind11::arg("compiled_dims") = "nk",
          pybind11::arg("comm_dtype") = "bf16");
#endif
}


} // namespace deep_gemm::gemm_rs
