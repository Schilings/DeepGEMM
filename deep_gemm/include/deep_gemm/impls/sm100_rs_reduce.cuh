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
//  sm100_rs_reduce_impl — TRUE Flux PULL-based RS Reduce kernel (Part 2)
// ============================================================================================
//
//  Independent kernel that PULLS partial results from all ranks and reduces into the final
//  output. Runs on a separate CUDA stream, overlapping with the GEMM compute kernel.
//
//  Pull model (aligned with Flux Sm90ReduceScatterDma):
//    The GEMM kernel only wrote LOCAL scatter buffer slot[dst_rank] + set LOCAL per-tile flag.
//    Here rank R, for each tile of its own chunk:
//      1. polls every src rank's REMOTE flag[R][m][n] (mapped via sym_buffer.map),
//      2. FP32-accumulates every src rank's REMOTE slot[R] (self maps back to local),
//      3. writes the final output,
//      4. resets the remote flags it consumed (Flux wait_eq_reset semantics — each
//         flag[R][*][*] is consumed by exactly one rank, so no cross-consumer race).
//
//  Precision: always FP32 accumulation regardless of comm_dtype (matches NCCL behaviour,
//  eliminates multi-rank BF16 accumulation error).
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

    // Vectorization: 128-bit = uint4
    constexpr uint32_t kVecBytes = 16;
    constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // 8 for BF16, 4 for FP32
    const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;
    const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

    constexpr int64_t kTimeoutCycles = 30ll * 2000000000ll;

    // Process tiles assigned to this CTA
    for (uint32_t tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const uint32_t my_m_block = tile_idx / num_n_blocks;
        const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;
        const uint32_t base_row = my_m_block * BLOCK_M;
        const uint32_t base_col = my_n_block * BLOCK_N;

        // ── Phase 1: Poll every src rank's REMOTE flag for our chunk (slot = rank_idx) ──
        if (tid < kNumRanks) {
            const uint32_t src_rank = tid;
            auto* local_flag = workspace.get_ready_ptr(rank_idx, my_m_block, my_n_block);
            auto* poll_ptr = reinterpret_cast<uint32_t*>(sym_buffer.map(local_flag, src_rank));
            const auto start_clock = clock64();
            while (ld_acq_sys(poll_ptr) == 0u) {
                if (clock64() - start_clock >= kTimeoutCycles) {
                    /* printf("RS reduce timeout: rank=%d, src=%d, tile=(%d,%d)\n",
                           rank_idx, src_rank, my_m_block, my_n_block); */
                    break;
                }
            }
        }
        __syncthreads();

        // ── Phase 2: Vectorized PULL + FP32 accumulation ──
        for (uint32_t vec_offset = tid; vec_offset < vecs_per_tile; vec_offset += kNumThreads) {
            const uint32_t elem_offset = vec_offset * kVecSize;
            const uint32_t tile_row = elem_offset / BLOCK_N;
            const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;
            const uint32_t global_row = base_row + tile_row;
            const uint32_t global_col = base_col + tile_col;

            if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                continue;

            float acc[kVecSize];
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i)
                acc[i] = 0.0f;

            // Sum every src rank's REMOTE slot[rank_idx] (self maps to local).
            #pragma unroll 1
            for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
                auto* local_partial = workspace.get_partial_ptr<comm_dtype_t>(rank_idx, global_row, global_col);
                const comm_dtype_t* partial_ptr = sym_buffer.map(local_partial, src_rank);
                uint4 data = *reinterpret_cast<const uint4*>(partial_ptr);
                if constexpr (cute::is_same_v<comm_dtype_t, cutlass::bfloat16_t>) {
                    const auto* bf16 = reinterpret_cast<const __nv_bfloat16*>(&data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += __bfloat162float(bf16[i]);
                } else {
                    const auto* f32 = reinterpret_cast<const float*>(&data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += f32[i];
                }
            }

            // Write final reduced result
            auto* out_ptr = output + global_row * shape_n + global_col;
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

        // ── Phase 3: Reset the remote flags consumed for this tile (wait_eq_reset) ──
        // Each flag[rank_idx][m][n] is consumed only by this rank → no cross-consumer race.
        // Must happen after all data for this tile is read.
        __syncthreads();
        if (tid < kNumRanks) {
            const uint32_t src_rank = tid;
            auto* local_flag = workspace.get_ready_ptr(rank_idx, my_m_block, my_n_block);
            auto* remote_flag = reinterpret_cast<uint32_t*>(sym_buffer.map(local_flag, src_rank));
            st_rel_sys(remote_flag, 0u);
        }
    }
}

} // namespace deep_gemm
