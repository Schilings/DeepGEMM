#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "../utils/layout.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_a2a_transpose_comm.hpp"
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
                                    const int& head_dim) {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    DG_HOST_ASSERT(nheads % num_ranks == 0 and seq % num_ranks == 0 and head_dim % 8 == 0);
    sm100_a2a_transpose_comm(sym_buffer, sym_buffer_ptrs, rank_idx, bs, nheads, seq, head_dim);
#else
    DG_HOST_UNREACHABLE("BF16 A2A-transpose requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_symm_buffer_size_for_bf16_a2a_transpose_gemm",
          &get_symm_buffer_size_for_bf16_a2a_transpose_gemm);
    m.def("bf16_a2a_transpose_comm", &bf16_a2a_transpose_comm,
          pybind11::arg("sym_buffer"), pybind11::arg("sym_buffer_ptrs"),
          pybind11::arg("rank_idx"), pybind11::arg("bs"), pybind11::arg("nheads"),
          pybind11::arg("seq"), pybind11::arg("head_dim"));
#endif
}

} // namespace deep_gemm::a2a_transpose_gemm
