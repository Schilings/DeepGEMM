#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "layout.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_gemm_a2a_transpose.hpp"
#endif

#include <deep_gemm/layout/gemm_a2a_transpose.cuh>


namespace deep_gemm::gemm_a2a_transpose {

// Ulysses-SP pre-attn: seq is sharded (each rank holds local_seq), so M must tile cleanly.
static int get_seq_alignment_for_gemm_a2a_transpose() {
    return 128;
}

// Returns (symm buffer byte size, slice lambda). The symmetric buffer holds ONLY the output
// region [bs*seq, local_n] (plus the 32B barrier header). Unlike GEMM-RS there are no partial
// slots and no ready flags — the A2A is a pure permutation. The slice lambda exposes the output
// region as a [bs, seq, local_n] view for the caller.
static std::tuple<int64_t, std::function<torch::Tensor(const torch::Tensor&)>>
get_symm_buffer_size_for_gemm_a2a_transpose(const int& num_ranks,
                                            const int& bs,
                                            const int& max_seq,
                                            const int& n,
                                            const bool& use_fp32_output) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(bs > 0 and max_seq > 0 and n > 0);
    DG_HOST_ASSERT(max_seq % get_seq_alignment_for_gemm_a2a_transpose() == 0);
    DG_HOST_ASSERT(n % num_ranks == 0);
    const int local_n = n / num_ranks;
    DG_HOST_ASSERT(local_n % 128 == 0);
    const int elem_size = use_fp32_output ? 4 : 2;
    const auto workspace = layout::GemmA2ATransposeWorkspace(nullptr, num_ranks, bs, max_seq, local_n, elem_size);

    auto slice_buffer = [=](const torch::Tensor& buffer) {
        const auto workspace = layout::GemmA2ATransposeWorkspace(nullptr, num_ranks, bs, max_seq, local_n, elem_size);
        auto out = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_output_ptr())),
            {bs, max_seq, local_n},
            use_fp32_output ? torch::TensorOptions().dtype(torch::kFloat).device(buffer.device())
                            : torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return out;
    };
    return {static_cast<int64_t>(workspace.get_num_bytes()), slice_buffer};
}

// ════════════════════════════════════════════════════════════════
//  Single-kernel pre-attn GEMM + All2All-transpose
// ════════════════════════════════════════════════════════════════
static void bf16_gemm_a2a_transpose_nt(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& sym_buffer,
                                       const std::vector<int64_t>& sym_buffer_ptrs,
                                       const int& rank_idx,
                                       const int& bs,
                                       const int& max_seq,
                                       const int& local_seq,
                                       const std::string& compiled_dims,
                                       const std::string& comm_dtype_str = "bf16") {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < static_cast<int>(sym_buffer_ptrs.size()));

    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(local_seq > 0 and local_seq * num_ranks <= max_seq * num_ranks);
    DG_HOST_ASSERT(local_seq % get_seq_alignment_for_gemm_a2a_transpose() == 0);

    const auto major_a = get_major_type_ab(a);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);
    const auto [am, ak] = get_shape<2>(a);
    const auto [n, bk] = get_shape<2>(b);
    DG_HOST_ASSERT(am == bs * local_seq and ak == bk);
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16 and b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(n % num_ranks == 0);
    DG_HOST_ASSERT(n % 128 == 0 and ak % 64 == 0);
    const int local_n = n / num_ranks;
    DG_HOST_ASSERT(local_n % 128 == 0);

    // Parse communication / output dtype
    at::ScalarType comm_dtype;
    if (comm_dtype_str == "bf16" or comm_dtype_str == "bfloat16") {
        comm_dtype = torch::kBFloat16;
    } else if (comm_dtype_str == "fp32" or comm_dtype_str == "float32") {
        comm_dtype = torch::kFloat;
    } else {
        DG_HOST_UNREACHABLE("Unsupported comm_dtype: must be 'bf16' or 'fp32'");
    }

    const bool use_fp32_comm = (comm_dtype == torch::kFloat);
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_gemm_a2a_transpose(
        num_ranks, bs, max_seq, n, use_fp32_comm);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));

    sm100_bf16_gemm_a2a_transpose_nt(a, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                                     bs, local_seq, n, static_cast<int>(ak), compiled_dims,
                                     comm_dtype);
#else
    DG_HOST_UNREACHABLE("BF16 GEMM+A2A-transpose requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_seq_alignment_for_gemm_a2a_transpose", &get_seq_alignment_for_gemm_a2a_transpose);
    m.def("get_symm_buffer_size_for_gemm_a2a_transpose", &get_symm_buffer_size_for_gemm_a2a_transpose);
    m.def("bf16_gemm_a2a_transpose_nt", &bf16_gemm_a2a_transpose_nt,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sym_buffer"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("bs"), pybind11::arg("max_seq"), pybind11::arg("local_seq"),
          pybind11::arg("compiled_dims") = "nk",
          pybind11::arg("comm_dtype") = "bf16");
#endif
}


} // namespace deep_gemm::gemm_a2a_transpose
