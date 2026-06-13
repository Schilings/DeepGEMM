#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "../utils/layout.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_a2a_gemm.hpp"
#endif

#include <deep_gemm/layout/bf16_a2a_gemm.cuh>

namespace deep_gemm::a2a_gemm {

static int get_token_alignment_for_a2a_gemm() {
    return 128;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_bf16_a2a_gemm(const int& num_ranks,
                                       const int& num_max_tokens_per_rank,
                                       const int& hidden,
                                       const int& num_slots) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);
    DG_HOST_ASSERT(num_slots >= 2);
    const auto workspace = layout::BF16A2AGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, num_slots);
    auto slice_buffers = [=](const torch::Tensor& buffer) {
        const auto workspace = layout::BF16A2AGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, num_slots);
        // local_x: [num_ranks, M_per_rank, K] — data to scatter
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_local_x_ptr())),
            {num_ranks, num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        // slots_x: [num_slots, M_per_rank, K] — receive buffers
        auto slots_x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_slot_x_ptr())),
            {num_slots, num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return std::make_tuple(x, slots_x);
    };
    return {static_cast<int64_t>(workspace.get_num_bytes()), slice_buffers};
}

static void bf16_a2a_gemm_nt(const torch::Tensor& d,
                             const torch::Tensor& sym_buffer,
                             const torch::Tensor& b,
                             const std::vector<int64_t>& sym_buffer_ptrs,
                             const int& rank_idx,
                             const int& num_max_tokens_per_rank,
                             const int& num_tokens,
                             const int& num_slots,
                             const std::string& compiled_dims) {
#if DG_TENSORMAP_COMPATIBLE
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < static_cast<int>(sym_buffer_ptrs.size()));
    DG_HOST_ASSERT(num_tokens > 0 and num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_tokens % get_token_alignment_for_a2a_gemm() == 0);
    DG_HOST_ASSERT(num_slots >= static_cast<int>(sym_buffer_ptrs.size()));

    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto [n, k] = get_shape<2>(b);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto [m_total, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m_total == num_tokens * num_ranks and n == n_);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    check_major_type_cd(d);
    DG_HOST_ASSERT(n % 128 == 0 and k % 64 == 0);

    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_bf16_a2a_gemm(
        num_ranks, num_max_tokens_per_rank, k, num_slots);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, slots_x] = slice(sym_buffer);

    sm100_bf16_a2a_gemm_nt(d, slots_x, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                           num_max_tokens_per_rank, num_tokens, num_slots, n, k, compiled_dims);
#else
    DG_HOST_UNREACHABLE("BF16 A2A+GEMM requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_token_alignment_for_a2a_gemm", &get_token_alignment_for_a2a_gemm);
    m.def("get_symm_buffer_size_for_bf16_a2a_gemm", &get_symm_buffer_size_for_bf16_a2a_gemm);
    m.def("bf16_a2a_gemm_nt", &bf16_a2a_gemm_nt,
          pybind11::arg("d"), pybind11::arg("sym_buffer"), pybind11::arg("b"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_tokens"),
          pybind11::arg("num_slots"), pybind11::arg("compiled_dims") = "nk");
#endif
}

} // namespace deep_gemm::a2a_gemm
