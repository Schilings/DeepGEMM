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
//  Independent kernel that reduces the LOCAL partials gathered by the fused PUSH epilogue.
//  Runs on a separate CUDA stream after the GEMM kernel.
//
//  Fused-push model:
//    Each src rank s already TMA-pushed (in its GEMM epilogue, overlapped with MMA) its
//    partial for THIS rank's chunk into our LOCAL buffer slot[s], and set our LOCAL flag[s].
//    Here rank R, for each tile of its own chunk:
//      1. polls every src rank's LOCAL flag[s][m][n] (set via release.sys by the peer push),
//      2. FP32-accumulates every src rank's LOCAL slot[s] (pure HBM, no NVLink),
//      3. writes the final output,
//      4. resets the local flags it consumed (single producer = peer s's GEMM, single
//         consumer = this reduce → no race).
//
//  Performance design (high memory-level parallelism):
//    * The per-src mapped base pointers to slot[R] (elem 0) and flag[R] are CONSTANT for the
//      whole kernel → precomputed once, hoisted out of the hot loop (no per-element 64-bit
//      `get_partial_ptr` mul / `map` add inside the inner loop).
//    * Phase 2 batches `kUnroll` vectors per thread, issuing all (kUnroll × kNumRanks) 128-bit
//      P2P loads BEFORE consuming any of them → MLP = kUnroll × kNumRanks outstanding loads,
//      so a handful of SMs can approach P2P bandwidth (the latency-bound scalar version did not).
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
    constexpr uint32_t kUnroll = 8;                                  // vectors-in-flight per thread
    const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;
    const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

    constexpr int64_t kTimeoutCycles = 30ll * 2000000000ll;

    // ── Precompute per-src CONSTANT base pointers (hoisted out of the hot loop) ──
    // In the PUSH/fused model every src rank s has already TMA-pushed its partial for THIS
    // rank's chunk into our LOCAL buffer slot[s], and set our LOCAL flag[s]. So the reduce is
    // a pure LOCAL accumulation (HBM, no NVLink) — the cross-card cost was already paid in the
    // GEMM epilogue, overlapped with MMA.
    //   slot_base[s] : LOCAL pointer to slot[s] element (0,0).
    //   flag_base[s] : LOCAL pointer to the ready-flag array for slot[s] (block 0,0).
    const comm_dtype_t* slot_base[kNumRanks];
    uint32_t* flag_base[kNumRanks];
    #pragma unroll
    for (uint32_t s = 0; s < kNumRanks; ++ s) {
        slot_base[s] = workspace.get_partial_ptr<comm_dtype_t>(s, 0, 0);
        flag_base[s] = workspace.get_ready_ptr(s, 0, 0);
    }

    // Process tiles assigned to this CTA
    for (uint32_t tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        const uint32_t my_m_block = tile_idx / num_n_blocks;
        const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;
        const uint32_t base_row = my_m_block * BLOCK_M;
        const uint32_t base_col = my_n_block * BLOCK_N;
        const uint32_t flag_off = my_m_block * num_n_blocks + my_n_block;

        // ── Phase 1: Poll every src rank's REMOTE flag for our chunk (slot = rank_idx) ──
        if (tid < kNumRanks) {
            const uint32_t src_rank = tid;
            auto* poll_ptr = flag_base[src_rank] + flag_off;
            const auto start_clock = clock64();
            while (ld_acq_sys(poll_ptr) == 0u) {
                if (clock64() - start_clock >= kTimeoutCycles)
                    break;
            }
        }
        __syncthreads();

        // ── Phase 2: Vectorized PULL + FP32 accumulation, kUnroll vectors in flight ──
        for (uint32_t base_vec = tid; base_vec < vecs_per_tile; base_vec += kNumThreads * kUnroll) {
            // Load phase: issue all (kUnroll × kNumRanks) 128-bit P2P loads up front.
            uint4 reg[kUnroll][kNumRanks];
            uint32_t lin[kUnroll];
            bool valid[kUnroll];

            #pragma unroll
            for (uint32_t u = 0; u < kUnroll; ++ u) {
                const uint32_t vec_offset = base_vec + u * kNumThreads;
                const uint32_t elem_offset = vec_offset * kVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;
                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;
                valid[u] = (vec_offset < vecs_per_tile) and
                           (global_row < runtime_m_per_rank) and (global_col < shape_n);
                lin[u] = global_row * shape_n + global_col;
                if (valid[u]) {
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumRanks; ++ s)
                        reg[u][s] = *reinterpret_cast<const uint4*>(slot_base[s] + lin[u]);
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

                auto* out_ptr = output + lin[u];
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

        // ── Phase 3: Reset the remote flags consumed for this tile (wait_eq_reset) ──
        // Each flag[rank_idx][m][n] is consumed only by this rank → no cross-consumer race.
        __syncthreads();
        if (tid < kNumRanks) {
            const uint32_t src_rank = tid;
            st_rel_sys(flag_base[src_rank] + flag_off, 0u);
        }
    }
}

} // namespace deep_gemm
