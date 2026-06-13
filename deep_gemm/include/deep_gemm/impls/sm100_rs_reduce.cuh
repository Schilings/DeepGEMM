#pragma once

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
//  sm100_rs_reduce_impl — RS Reduce kernel for dual-kernel GEMM+RS (Part 2)
// ============================================================================================
//
//  Independent kernel that reduces partial results from all ranks into final output.
//  Runs on a separate CUDA stream, overlapping with the GEMM compute kernel.
//
//  Design:
//    - Each CTA handles one tile of the output
//    - Per-tile polling of ready flags (set by GEMM kernel epilogue)
//    - Vectorized BF16 __hadd2 / FP32 accumulate for maximum memory bandwidth
//    - 256T/CTA for full memory bandwidth utilization
//
//  Overlap mechanism:
//    - GEMM kernel computes tiles in round-robin interleaved order
//    - This kernel polls per-tile ready flags, processing tiles as they complete
//    - Natural pipeline: while GEMM computes later tiles, we reduce earlier ones
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t,
          uint32_t kNumThreads = 256>
__global__ void __launch_bounds__(kNumThreads, 4)
sm100_rs_reduce_impl(cd_dtype_t* __restrict__ output,
                      const layout::SymBuffer<kNumRanks> sym_buffer,
                      const uint32_t runtime_m_per_rank,
                      const uint32_t shape_n,
                      const uint32_t shape_m_per_rank) {
    using namespace deep_gemm::ptx;

    const uint32_t tid = threadIdx.x;
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    const uint32_t num_m_blocks = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t total_tiles = num_m_blocks * num_n_blocks;

    // Elements per tile and vectorization
    constexpr uint32_t kVecBytes = 16;  // 128-bit = uint4
    constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // 8 for BF16, 4 for FP32

    // BF16 wide vector: 32B = 2x uint4, process 16 BF16 per iteration
    constexpr uint32_t kWideVecBytes = 32;
    constexpr uint32_t kWideVecSize = kWideVecBytes / sizeof(comm_dtype_t);  // 16 for BF16
    const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;

    // Process tiles assigned to this CTA
    for (uint32_t tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const uint32_t my_m_block = tile_idx / num_n_blocks;
        const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;
        const uint32_t base_row = my_m_block * BLOCK_M;
        const uint32_t base_col = my_n_block * BLOCK_N;

        // Phase 1: Poll ALL ranks ready flags for this tile
        if (tid < kNumRanks) {
            const uint32_t src_rank = tid;
            auto* poll_ptr = workspace.get_ready_ptr(src_rank, my_m_block, my_n_block);
            constexpr int64_t kTimeoutCycles = 30ll * 2000000000ll;
            const auto start_clock = clock64();
            while (ld_acq_sys(poll_ptr) == 0u) {
                if (clock64() - start_clock >= kTimeoutCycles) {
                    printf("RS reduce timeout: rank=%d, src=%d, tile=(%d,%d)\n",
                           rank_idx, src_rank, my_m_block, my_n_block);
                    break;
                }
            }
        }
        __syncthreads();

        // Phase 2: Vectorized reduce
        if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
            // BF16 fast path with wide vectors (32B = 16 BF16 per iteration)
            const uint32_t wide_vecs_per_tile = elems_per_tile / kWideVecSize;

            for (uint32_t wide_vec_offset = tid; wide_vec_offset < wide_vecs_per_tile; wide_vec_offset += kNumThreads) {
                const uint32_t elem_offset = wide_vec_offset * kWideVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;
                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;

                if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                    continue;

                // Load self-rank contribution from output
                auto* out_ptr = output + global_row * shape_n + global_col;
                uint4 self_data0 = *reinterpret_cast<const uint4*>(out_ptr);
                uint4 self_data1 = *reinterpret_cast<const uint4*>(out_ptr + kVecSize);

                // Accumulate N-1 remote ranks using __hadd2
                #pragma unroll 1
                for (uint32_t rank_iter = 0; rank_iter < kNumRanks - 1; ++ rank_iter) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;
                    const comm_dtype_t* partial_ptr =
                        workspace.get_partial_ptr<comm_dtype_t>(src_rank, global_row, global_col);

                    uint4 remote_data0 = *reinterpret_cast<const uint4*>(partial_ptr);
                    uint4 remote_data1 = *reinterpret_cast<const uint4*>(partial_ptr + kVecSize);

                    auto* s0 = reinterpret_cast<__nv_bfloat162*>(&self_data0);
                    auto* s1 = reinterpret_cast<__nv_bfloat162*>(&self_data1);
                    const auto* r0 = reinterpret_cast<const __nv_bfloat162*>(&remote_data0);
                    const auto* r1 = reinterpret_cast<const __nv_bfloat162*>(&remote_data1);
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i)
                        s0[i] = __hadd2(s0[i], r0[i]);
                    #pragma unroll
                    for (uint32_t i = 0; i < 4; ++ i)
                        s1[i] = __hadd2(s1[i], r1[i]);
                }

                // Write final reduced result
                *reinterpret_cast<uint4*>(out_ptr) = self_data0;
                *reinterpret_cast<uint4*>(out_ptr + kVecSize) = self_data1;
            }
        } else {
            // FP32 fallback
            const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

            for (uint32_t vec_offset = tid; vec_offset < vecs_per_tile; vec_offset += kNumThreads) {
                const uint32_t elem_offset = vec_offset * kVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;
                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;

                if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                    continue;

                auto* out_ptr = output + global_row * shape_n + global_col;
                uint4 self_data = *reinterpret_cast<const uint4*>(out_ptr);

                float acc[kVecSize];
                const auto* self_f32 = reinterpret_cast<const float*>(&self_data);
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    acc[i] = self_f32[i];

                #pragma unroll 1
                for (uint32_t rank_iter = 0; rank_iter < kNumRanks - 1; ++ rank_iter) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;
                    const comm_dtype_t* partial_ptr =
                        workspace.get_partial_ptr<comm_dtype_t>(src_rank, global_row, global_col);

                    uint4 data = *reinterpret_cast<const uint4*>(partial_ptr);
                    const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += static_cast<float>(comm_data[i]);
                }

                uint4 result;
                auto* out_f32 = reinterpret_cast<float*>(&result);
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    out_f32[i] = acc[i];
                *reinterpret_cast<uint4*>(out_ptr) = result;
            }
        }
    }
}

} // namespace deep_gemm
