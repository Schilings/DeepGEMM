#pragma once

#include <functional>
#include <pybind11/functional.h>

#include "../jit/device_runtime.hpp"
#include "../utils/layout.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_fp8_ag_gemm.hpp"
#endif
#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm100_bf16_ag_gemm.hpp"
#endif

#include <deep_gemm/layout/ag_gemm.cuh>
#include <deep_gemm/layout/bf16_ag_gemm.cuh>


namespace deep_gemm::ag_gemm {

static int get_token_alignment_for_ag_gemm() {
    return 128;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_ag_gemm(const int& num_ranks,
                                 const int& num_max_tokens_per_rank,
                                 const int& hidden,
                                 const int& gran_k,
                                 const int& num_slots) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(hidden % (gran_k * 4) == 0);
    DG_HOST_ASSERT(num_slots >= 2);

    const auto workspace = layout::AGGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, gran_k, num_slots);
    const int sf_cols = workspace.get_num_sf_cols();

    auto slice_buffers = [=](const torch::Tensor& buffer) {
        const auto workspace = layout::AGGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, gran_k, num_slots);
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_local_x_ptr())),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_local_x_sf_ptr())),
            {num_max_tokens_per_rank, sf_cols},
            {1, num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));
        auto slots_x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_slot_x_ptr())),
            {num_slots, num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto slots_x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_slot_x_sf_ptr())),
            {num_slots, num_max_tokens_per_rank, sf_cols},
            {num_max_tokens_per_rank * sf_cols, 1, num_max_tokens_per_rank},
            torch::TensorOptions().dtype(torch::kInt).device(buffer.device()));

        return std::make_tuple(x, x_sf, slots_x, slots_x_sf);
    };
    return {static_cast<int64_t>(workspace.get_num_bytes()), slice_buffers};
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_bf16_ag_gemm(const int& num_ranks,
                                      const int& num_max_tokens_per_rank,
                                      const int& hidden,
                                      const int& num_slots) {
    DG_HOST_ASSERT(num_ranks > 1);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);
    DG_HOST_ASSERT(num_slots >= 2);
    const auto workspace = layout::BF16AGGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, num_slots);
    auto slice_buffers = [=](const torch::Tensor& buffer) {
        const auto workspace = layout::BF16AGGemmWorkspace(nullptr, num_ranks, num_max_tokens_per_rank, hidden, num_slots);
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_local_x_ptr())),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        auto slots_x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(workspace.get_slot_x_ptr())),
            {num_slots, num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kBFloat16).device(buffer.device()));
        return std::make_tuple(x, slots_x);
    };
    return {static_cast<int64_t>(workspace.get_num_bytes()), slice_buffers};
}

static void fp8_ag_gemm_nt(const torch::Tensor& d,

                           const std::tuple<torch::Tensor, torch::Tensor>& b_tuple,
                           const torch::Tensor& sym_buffer,
                           const std::vector<int64_t>& sym_buffer_ptrs,
                           const int& rank_idx,
                           const int& num_max_tokens_per_rank,
                           const int& num_tokens,
                           const int& gran_k,
                           const int& num_slots,
                           const std::tuple<int, int, int>& recipe,
                           const std::string& compiled_dims) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    const auto [b, b_sf] = b_tuple;
    const auto arch_major = device_runtime->get_arch_major();
    DG_HOST_ASSERT(arch_major == 10);
    DG_HOST_ASSERT(sym_buffer_ptrs.size() > 1);
    DG_HOST_ASSERT(rank_idx >= 0 and rank_idx < static_cast<int>(sym_buffer_ptrs.size()));
    DG_HOST_ASSERT(num_tokens > 0 and num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(gran_k == 32 or gran_k == 128);
    DG_HOST_ASSERT(std::get<0>(recipe) == 1 and std::get<1>(recipe) == 1 and std::get<2>(recipe) == gran_k);

    const auto major_b = get_major_type_ab(b);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    const auto [n, k] = check_ab_fp8_fp4(b, major_b, arch_major);
    const int num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto [m_total, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m_total == num_tokens * num_ranks and n == n_);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    check_major_type_cd(d);
    check_sf_layout(b_sf, n, k, 1, gran_k, std::nullopt, true, false, torch::kInt);

    DG_HOST_ASSERT(num_slots >= num_ranks);
    DG_HOST_ASSERT(num_tokens % get_token_alignment_for_ag_gemm() == 0);
    DG_HOST_ASSERT(n % 128 == 0 and k % 128 == 0);

    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_ag_gemm(
        num_ranks, num_max_tokens_per_rank, k, gran_k, num_slots);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, x_sf, slots_x, slots_x_sf] = slice(sym_buffer);

    sm100_fp8_ag_gemm_nt(d, slots_x, slots_x_sf, b, b_sf, sym_buffer_ptrs, rank_idx,
                         num_max_tokens_per_rank, num_tokens, gran_k, num_slots, n, k, compiled_dims);

#else
    DG_HOST_UNREACHABLE("FP8 AG+GEMM requires FP8 and TensorMap support");
#endif
}

static void bf16_ag_gemm_nt(const torch::Tensor& d,
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
    DG_HOST_ASSERT(num_tokens % get_token_alignment_for_ag_gemm() == 0);
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
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_bf16_ag_gemm(
        num_ranks, num_max_tokens_per_rank, k, num_slots);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    const auto [x, slots_x] = slice(sym_buffer);
    sm100_bf16_ag_gemm_nt(d, slots_x, b, sym_buffer, sym_buffer_ptrs, rank_idx,
                          num_max_tokens_per_rank, num_tokens, num_slots, n, k, compiled_dims);
#else
    DG_HOST_UNREACHABLE("BF16 AG+GEMM requires TensorMap support");
#endif
}

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE

    m.def("get_token_alignment_for_ag_gemm", &get_token_alignment_for_ag_gemm);
    m.def("get_symm_buffer_size_for_ag_gemm", &get_symm_buffer_size_for_ag_gemm);
    m.def("fp8_ag_gemm_nt", &fp8_ag_gemm_nt,
          pybind11::arg("d"), pybind11::arg("b"), pybind11::arg("sym_buffer"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_tokens"),
          pybind11::arg("gran_k"), pybind11::arg("num_slots"),
          pybind11::arg("recipe") = std::make_tuple(1, 1, 32),
          pybind11::arg("compiled_dims") = "nk");
#endif
#if DG_TENSORMAP_COMPATIBLE
    m.def("get_symm_buffer_size_for_bf16_ag_gemm", &get_symm_buffer_size_for_bf16_ag_gemm);
    m.def("bf16_ag_gemm_nt", &bf16_ag_gemm_nt,
          pybind11::arg("d"), pybind11::arg("sym_buffer"), pybind11::arg("b"),
          pybind11::arg("sym_buffer_ptrs"), pybind11::arg("rank_idx"),
          pybind11::arg("num_max_tokens_per_rank"), pybind11::arg("num_tokens"),
          pybind11::arg("num_slots"), pybind11::arg("compiled_dims") = "nk");
#endif
}


} // namespace deep_gemm::ag_gemm
