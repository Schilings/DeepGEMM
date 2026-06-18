#pragma once

#include <cuda_bf16.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <cutlass/bfloat16.h>
#include <cute/util/type_traits.hpp>
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

using namespace deep_gemm::math;

// ============================================================================================
//  sm100_rs_reduce_impl — local reduction of the fused PUSH partials (Part 2)
// ============================================================================================
//
//  In the fused-push model the GEMM epilogue of every rank already TMA-pushed (overlapped with
//  MMA) its partial for THIS rank's chunk into our LOCAL buffer slot[s], and the GEMM kernel's
//  final system-scope `nvlink_barrier` guarantees all those pushes are globally visible before
//  the GEMM kernel exits. The host orders this reduce strictly AFTER GEMM completion, so the
//  data is always ready — NO per-tile flags / polling / __syncthreads are needed.
//
//  Layout fact exploited: each rank's chunk = [runtime_m_per_rank x shape_n] occupies a fully
//  CONTIGUOUS region at the start of each slot (slot is [num_max_tokens x hidden], hidden==N).
//  So the reduce is a flat, perfectly-coalesced 1D streaming accumulation over `total_elems`:
//      output[i] = sum_{s=0..R-1} slot[s][i]
//  High MLP via kUnroll batched 128-bit loads → saturates local HBM bandwidth.
//
//  Precision: always FP32 accumulation regardless of comm_dtype.
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t,
          uint32_t kNumThreads = 256>
__global__ void __launch_bounds__(kNumThreads, 2)
sm100_rs_reduce_impl(cd_dtype_t* __restrict__ output,
                      const layout::SymBuffer<kNumRanks> sym_buffer,
                      const uint32_t runtime_m_per_rank,
                      const uint32_t shape_n,
                      const uint32_t shape_m_per_rank) {
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    // Vectorization: 128-bit = uint4
    constexpr uint32_t kVecBytes = 16;
    constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // 8 for BF16, 4 for FP32
    constexpr uint32_t kUnroll = 8;                                  // vectors-in-flight per thread

    // This rank's chunk is a contiguous [runtime_m_per_rank x shape_n] region per slot.
    const uint32_t total_elems = runtime_m_per_rank * shape_n;
    const uint32_t total_vecs = total_elems / kVecSize;

    // ── Per-src CONSTANT LOCAL base pointers to slot[s] element (0,0) (hoisted) ──
    const comm_dtype_t* slot_base[kNumRanks];
    #pragma unroll
    for (uint32_t s = 0; s < kNumRanks; ++ s)
        slot_base[s] = workspace.get_partial_ptr<comm_dtype_t>(s, 0, 0);

    const uint32_t global_tid = blockIdx.x * kNumThreads + threadIdx.x;
    const uint32_t grid_threads = gridDim.x * kNumThreads;

    // ── Flat, coalesced 1D streaming reduce, kUnroll vectors in flight ──
    for (uint32_t base_vec = global_tid; base_vec < total_vecs; base_vec += grid_threads * kUnroll) {
        uint4 reg[kUnroll][kNumRanks];
        uint32_t vidx[kUnroll];
        bool valid[kUnroll];

        // Load phase: issue all (kUnroll × kNumRanks) 128-bit loads up front.
        #pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++ u) {
            vidx[u] = base_vec + u * grid_threads;
            valid[u] = vidx[u] < total_vecs;
            if (valid[u]) {
                const uint32_t elem = vidx[u] * kVecSize;
                #pragma unroll
                for (uint32_t s = 0; s < kNumRanks; ++ s)
                    reg[u][s] = *reinterpret_cast<const uint4*>(slot_base[s] + elem);
            }
        }

        // Reduce + store phase.
        #pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++ u) {
            if (not valid[u])
                continue;

            float acc[kVecSize];
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i)
                acc[i] = 0.0f;

            #pragma unroll
            for (uint32_t s = 0; s < kNumRanks; ++ s) {
                if constexpr (cute::is_same_v<comm_dtype_t, cutlass::bfloat16_t>) {
                    const auto* bf16 = reinterpret_cast<const __nv_bfloat16*>(&reg[u][s]);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += __bfloat162float(bf16[i]);
                } else {
                    const auto* f32 = reinterpret_cast<const float*>(&reg[u][s]);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += f32[i];
                }
            }

            auto* out_ptr = output + vidx[u] * kVecSize;
            uint4 result;
            if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                auto* out_bf16 = reinterpret_cast<__nv_bfloat16*>(&result);
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    out_bf16[i] = __float2bfloat16(acc[i]);
            } else {
                auto* out_f32 = reinterpret_cast<float*>(&result);
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    out_f32[i] = acc[i];
            }
            *reinterpret_cast<uint4*>(out_ptr) = result;
        }
    }
}

} // namespace deep_gemm
