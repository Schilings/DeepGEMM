#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "../utils/layout.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_a2a_transpose_comm.hpp"
#include "../jit_kernels/impls/sm100_bf16_a2a_transpose_gemm.hpp"
#endif

#include <deep_gemm/layout/bf16_a2a_transpose_gemm.cuh>

namespace deep_gemm::a2a_transpose_gemm {

// Returns (num_bytes, slice_fn). slice_fn(buffer) -> (x_input, gathered):
//   x_input : [bs, local_nheads, seq, head_dim]   — write this rank's attention output here
//   gathered: [bs*local_seq, hidden]              — A matrix for Wo GEMM (filled by the comm)
static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_bf16_a2a_transpose_gemm(const int& num_ranks,
                                                 const int& bs,
                                                 const int& nheads,
                                                 const int& seq,
                                                 const int& head_dim) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(bs > 0 and nheads > 0 and seq > 0 and head_dim > 0);
    DG_HOST_ASSERT(nheads % num_ranks == 0 and seq % num_ranks == 0);
    DG_HOST_ASSERT(head_dim % 8 == 0);  // uint4-vectorized comm
    const auto ws = layout::BF16A2ATransposeGemmWorkspace(nullptr, num_ranks, bs, nheads, seq, head_dim);
    const int local_nheads = static_cast<int>(nheads / num_ranks);
    const int local_seq = static_cast<int>(seq / num_ranks);
    const int hidden = nheads * head_dim;
    auto slice_buffers = [=](const torch::Tensor& buffer) {
        const auto ws = layout::BF16A2ATransposeGemmWorkspace(nullptr, num_ranks, bs, nheads, seq, head_dim);
        auto x_input = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(ws.get_input_ptr())),
            {bs, local_nheads, seq, head_dim},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto gathered = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(ws.get_gathered_ptr())),
            {bs * local_seq, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return std::make_tuple(x_input, gathered);
    };
    return {static_cast<int64_t>(ws.get_num_bytes()), slice_buffers};
}

// M0: transpose-scatter comm only (caller barriers + runs the Wo GEMM separately).
static void bf16_a2a_transpose_comm(const torch::Tensor& sym_buffer,
                                    const std::vector<int64_t>& sym_buffer_ptrs,
                                    const int& rank_idx,
                                    const int& bs,
                                    const int& nheads,
                                    const int& seq,
                                    const int& head_dim,
                                    const bool& seq_major = false) {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(nheads % num_ranks == 0 and seq % num_ranks == 0 and head_dim % 8 == 0);
    // seq_major=true consumes FlashAttention-native BSHD input [bs, seq, local_nheads, head_dim]
    // directly (no .permute to BHSD needed); false keeps the BHSD [bs, local_nheads, seq, head_dim].
    sm100_a2a_transpose_comm(sym_buffer, sym_buffer_ptrs, rank_idx, bs, nheads, seq, head_dim,
                             /*tile_m=*/128, /*set_barrier=*/false, /*seq_major_in=*/seq_major);
#else
    DG_HOST_UNREACHABLE("BF16 A2A-transpose requires TensorMap support");
#endif
}

// Fused (M1): transpose-scatter comm overlapped with the Wo GEMM (per-M-tile barrier).
static void bf16_a2a_transpose_gemm_nt(const torch::Tensor& d,
                                       const torch::Tensor& gathered,
                                       const torch::Tensor& b,
                                       const torch::Tensor& sym_buffer,
                                       const std::vector<int64_t>& sym_buffer_ptrs,
                                       const int& rank_idx,
                                       const int& bs,
                                       const int& nheads,
                                       const int& seq,
                                       const int& head_dim,
                                       const std::string& compiled_dims) {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 1 and rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(nheads % num_ranks == 0 and seq % num_ranks == 0 and head_dim % 8 == 0);
    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto [n, k] = get_shape<2>(b);
    DG_HOST_ASSERT(k == nheads * head_dim);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(n % 128 == 0 and k % 64 == 0);
    sm100_bf16_a2a_transpose_gemm(d, gathered, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                                  bs, nheads, seq, head_dim, n, compiled_dims);
#else
    DG_HOST_UNREACHABLE("BF16 A2A-transpose GEMM requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_symm_buffer_size_for_bf16_a2a_transpose_gemm",
          &get_symm_buffer_size_for_bf16_a2a_transpose_gemm);
    m.def("bf16_a2a_transpose_comm", &bf16_a2a_transpose_comm,
          pybind11::arg("sym_buffer"), pybind11::arg("sym_buffer_ptrs"),
          pybind11::arg("rank_idx"), pybind11::arg("bs"), pybind11::arg("nheads"),
          pybind11::arg("seq"), pybind11::arg("head_dim"),
          pybind11::arg("seq_major") = false);
    m.def("bf16_a2a_transpose_gemm_nt", &bf16_a2a_transpose_gemm_nt,
          pybind11::arg("d"), pybind11::arg("gathered"), pybind11::arg("b"),
          pybind11::arg("sym_buffer"), pybind11::arg("sym_buffer_ptrs"),
          pybind11::arg("rank_idx"), pybind11::arg("bs"), pybind11::arg("nheads"),
          pybind11::arg("seq"), pybind11::arg("head_dim"), pybind11::arg("compiled_dims") = "nk");
#endif
}

} // namespace deep_gemm::a2a_transpose_gemm
