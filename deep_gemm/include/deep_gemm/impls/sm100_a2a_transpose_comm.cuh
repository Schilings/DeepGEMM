#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_bf16.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/bf16_a2a_transpose_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

// ============================================================================================
//  sm100_a2a_transpose_comm — Ulysses SP post-attention All2All-transpose scatter (cuda core)
// ============================================================================================
//
//  Each rank r pushes its attention output x[bs, local_nheads, seq, head_dim] to every dst_rank,
//  writing into dst_rank's *gathered* region at hidden-column offset r*local_hidden, with the
//  seq<->head transpose:
//      gathered_dst[b, dst_seq, (r*local_nheads + nh), hd] = x[b, nh, gs, hd]
//      where dst_rank = gs / local_seq,  dst_seq = gs % local_seq.
//  After all ranks finish (host barrier), each rank's gathered region holds the full-hidden
//  [bs, local_seq, hidden] = A matrix for the Wo GEMM.
//
//  Vectorized over head_dim with uint4 (8 bf16 = 16B); requires head_dim % 8 == 0.
//
template <uint32_t kNumRanks>
__global__ void sm100_a2a_transpose_comm_impl(
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const uint32_t bs,
        const uint32_t nheads,
        const uint32_t seq,
        const uint32_t head_dim) {
    const uint32_t rank = sym_buffer.rank_idx;
    const uint32_t local_nheads = nheads / kNumRanks;
    const uint32_t local_seq = seq / kNumRanks;
    const uint32_t hidden = nheads * head_dim;

    constexpr uint32_t kPack = 8;                         // bf16 per uint4
    const uint32_t vec_hd = head_dim / kPack;             // uint4 per head row
    const uint32_t vec_hidden = hidden / kPack;

    const layout::BF16A2ATransposeGemmWorkspace ws(
        sym_buffer.template get_base_ptr<void*>(), kNumRanks, bs, nheads, seq, head_dim);

    const uint4* in_vec = reinterpret_cast<const uint4*>(ws.template get_input_ptr<void>());
    // local gathered base, as if writing locally; mapped to dst_rank below.
    uint4* gathered_local = reinterpret_cast<uint4*>(ws.template get_gathered_ptr<void>());

    // iterate over all input vectors: layout [bs, local_nheads, seq, vec_hd] (vec_hd fastest)
    const uint64_t total_vec = static_cast<uint64_t>(bs) * local_nheads * seq * vec_hd;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total_vec; i += stride) {
        uint32_t hd_v = i % vec_hd;
        uint64_t rem = i / vec_hd;
        uint32_t gs = rem % seq;
        uint64_t rem2 = rem / seq;
        uint32_t nh = rem2 % local_nheads;
        uint32_t b = static_cast<uint32_t>(rem2 / local_nheads);

        const uint32_t dst_rank = gs / local_seq;
        const uint32_t dst_seq = gs % local_seq;

        // out offset (in uint4) within the gathered region [bs, local_seq, vec_hidden]
        const uint64_t out_vec = (static_cast<uint64_t>(b) * local_seq + dst_seq) * vec_hidden +
                                 static_cast<uint64_t>(rank * local_nheads + nh) * vec_hd + hd_v;

        uint4* dst_ptr = sym_buffer.map(gathered_local + out_vec, dst_rank);
        *dst_ptr = in_vec[i];
    }
}

} // namespace deep_gemm

#pragma clang diagnostic pop
