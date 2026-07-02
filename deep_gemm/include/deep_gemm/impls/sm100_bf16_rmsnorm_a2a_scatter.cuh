#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cuda_device_runtime_api.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/fused_qkv_norm_a2a.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;
using namespace deep_gemm::math;

// Host-side mirror of scatter maps (layout-compatible: CUtensorMap[8]).
struct FusedQKVNormA2AScatterMaps {
    cute::TmaDescriptor maps[8];
};

// ============================================================================================
//  sm100_bf16_rmsnorm_a2a_scatter_impl — v2a Kernel 2
//
//  Reads local GEMM output buffer [shape_m, shape_n] (Kernel 1's output),
//  applies RMSNorm on Q/K segments (optional), then A2A-transpose-scatters to peer HBM.
//
//  Pipeline per tile:
//    1. TMA load from local_buffer → SMEM (swizzled, via mbarrier)
//    2. RMSNorm elementwise in SMEM (swizzle-addressed read/modify/write)
//       - Read per-row x² sum from sum_buffer
//       - Apply: out = x * rsqrt(sum/dim + eps) * weight (Q/K only, V=identity)
//    3. TMA store scatter → peer HBM (scatter_maps[dst_rank])
//
//  Warp layout (128T = 4 warps):
//    W0: TMA load issue + TMA store issue (elect_one)
//    W0-W3: All participate in norm elementwise + barriers
//
//  GQA-aware scatter:
//    N_total = q_dim + 2*kv_dim
//    Q segment [0, q_dim):            dst_rank = n_col / local_q_n,  base_n = n_col % local_q_n
//    K segment [q_dim, q_dim+kv_dim): dst_rank = (n-q_dim) / local_kv_n, base_n += local_q_n
//    V segment [q_dim+kv_dim, ...):   dst_rank = (n-q_dim-kv_dim) / local_kv_n, base_n += local_q_n + local_kv_n
// ============================================================================================

template <uint32_t BLOCK_M,      // 128
          uint32_t BLOCK_N,      // 128
          uint32_t kSwizzleCDMode, // 128 (bytes)
          uint32_t kNumTMAStages, // 2
          uint32_t kNumRanks,
          bool kDoNormQ, bool kDoNormK,
          typename cd_dtype_t>    // nv_bfloat16
__global__ void __launch_bounds__(128, 1)
sm100_bf16_rmsnorm_a2a_scatter_impl(
    const uint32_t shape_m,     // bs * local_seq
    const uint32_t shape_n,     // N_total = q_dim + 2*kv_dim
    const uint32_t bs,
    const uint32_t seq,
    const uint32_t local_seq,
    const uint32_t q_dim,       // q_nheads * head_dim
    const uint32_t kv_dim,      // kv_nheads * head_dim
    const uint32_t local_q_n,   // (q_nheads/sp) * head_dim
    const uint32_t local_kv_n,  // (kv_nheads/sp) * head_dim
    const float eps,
    const float* __restrict__ norm_q_weight,    // [q_dim], fp32 (nullptr if !kDoNormQ)
    const float* __restrict__ norm_k_weight,    // [kv_dim], fp32 (nullptr if !kDoNormK)
    const float* __restrict__ sum_buffer,       // [shape_m, 2] per-row x² sum (Q=0, K=1)
    const cd_dtype_t* __restrict__ local_buffer, // [shape_m, shape_n] bf16, Kernel 1 output
    const __grid_constant__ cute::TmaDescriptor tensor_map_local,  // TMA descriptor for local_buffer
    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
    const __grid_constant__ FusedQKVNormA2AScatterMaps scatter_maps) {

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    constexpr uint32_t kNumThreads = 128;
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);  // 8 for bf16
    // STORE_BLOCK_M = BLOCK_M = 128, STORE_BLOCK_N = swizzle/sizeof = 64 for bf16
    constexpr uint32_t STORE_BLOCK_M = BLOCK_M;
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);  // 64

    // ── Runtime variables ──
    const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t num_total_tiles = num_m_blocks * num_n_blocks;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const uint32_t local_n_total = local_q_n + 2 * local_kv_n;

    const auto workspace = layout::FusedQKVNormA2AWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, bs, seq, local_n_total, sizeof(cd_dtype_t));

    // ── Shared memory layout ──
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // SMEM CD: double-buffered for load/store pipeline
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    // TMA load barriers (one per stage)
    auto load_barriers = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE_PER_STAGE * kNumTMAStages + i * sizeof(Barrier));
    });

    // ── Initialize barriers ──
    if (thread_idx == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumTMAStages; ++i) {
            load_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // ── Initial NVLink barrier ──
    constexpr uint32_t kInitBarrierTag = 61;
    comm::nvlink_barrier<kNumRanks, 128, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // ── Persistent tile loop ──
    uint32_t tma_stage_idx = 0;

    for (uint32_t tile_idx = blockIdx.x; tile_idx < num_total_tiles; tile_idx += 128) {
        const uint32_t m_block_idx = tile_idx / num_n_blocks;
        const uint32_t n_block_idx = tile_idx % num_n_blocks;
        const uint32_t global_m = m_block_idx * BLOCK_M;
        const uint32_t n_col = n_block_idx * BLOCK_N;

        // ── Determine segment (Q/K/V), dst_rank, and norm params ──
        uint32_t dst_rank, base_n_idx;
        bool is_q = (n_col < q_dim);
        bool is_k = (!is_q && n_col < q_dim + kv_dim);
        bool do_norm = false;
        const float* norm_weight = nullptr;
        uint32_t norm_dim = 0;
        uint32_t sum_idx = 0;  // 0 for Q, 1 for K

        if (is_q) {
            dst_rank = n_col / local_q_n;
            base_n_idx = n_col % local_q_n;
            if constexpr (kDoNormQ) { do_norm = true; norm_weight = norm_q_weight; norm_dim = q_dim; sum_idx = 0; }
        } else if (is_k) {
            uint32_t rel = n_col - q_dim;
            dst_rank = rel / local_kv_n;
            base_n_idx = rel % local_kv_n + local_q_n;
            if constexpr (kDoNormK) { do_norm = true; norm_weight = norm_k_weight; norm_dim = kv_dim; sum_idx = 1; }
        } else {  // V
            uint32_t rel = n_col - q_dim - kv_dim;
            dst_rank = rel / local_kv_n;
            base_n_idx = rel % local_kv_n + local_q_n + local_kv_n;
        }

        // Output M coordinates (pre-attn: rank's seq shard → dst's seq offset)
        const uint32_t b = global_m / local_seq;
        const uint32_t s_local = global_m - b * local_seq;
        const uint32_t base_m_idx = b * seq + rank_idx * local_seq + s_local;

        // ── Step 1: TMA load from local_buffer → SMEM (swizzled) ──
        // Wait for previous TMA store to finish using this stage's SMEM
        if (warp_idx == 0) cute::tma_store_wait<kNumTMAStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumThreads, 0);

        // Issue TMA load for each STORE_BLOCK_N atom
        // tensor_map_local describes [shape_m, shape_n] 2D with swizzle=128B
        // TMA load puts data in swizzled SMEM layout automatically
        if (warp_idx == 0 and cute::elect_one_sync()) {
            constexpr uint32_t BLOCK_INNER_ATOM = kSwizzleCDMode / sizeof(cd_dtype_t);
            #pragma unroll
            for (uint32_t s = 0; s < BLOCK_N / STORE_BLOCK_N; ++s) {
                // TMA load: tile at (global_m, n_col + s*STORE_BLOCK_N) → SMEM atom s
                // SMEM offset: atom s starts at s * STORE_BLOCK_M * STORE_BLOCK_N
                cute::SM90_TMA_LOAD_2D::copy(
                    &tensor_map_local,
                    reinterpret_cast<uint64_t*>(load_barriers[tma_stage_idx]),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                    reinterpret_cast<cd_dtype_t*>(smem_cd[tma_stage_idx]) + s * STORE_BLOCK_M * STORE_BLOCK_N,
                    n_col + s * STORE_BLOCK_N,
                    global_m);
            }
            // Arrive at load barrier after all TMA loads for this stage
            constexpr uint32_t kNumArrivalBytes = BLOCK_M * BLOCK_N * sizeof(cd_dtype_t);
            load_barriers[tma_stage_idx]->arrive_and_expect_tx(kNumArrivalBytes);
        }

        // Wait for TMA load to complete
        load_barriers[tma_stage_idx]->wait((tile_idx / 128) & 1 ^ 1);  // phase based on persistent iter
        ptx::tcgen05_after_thread_sync();

        // ── Step 2: RMSNorm elementwise in swizzled SMEM ──
        if (do_norm) {
            // Read per-row x² sum and compute rms
            // Each thread handles one "row" of the swizzled atom (like STSM in sm100_store_cd)
            // We use 4 warps × 32 lanes = 128 threads to cover 128 rows
            // Each row has STORE_BLOCK_N=64 bf16 elements, processed in 8 bank groups of 8

            #pragma unroll
            for (uint32_t s = 0; s < BLOCK_N / STORE_BLOCK_N; ++s) {
                auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                     s * STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);

                // Each warp handles STORE_BLOCK_M / kNumWarps = 32 rows
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
                    // Swizzle addressing (same as sm100_store_cd STSM)
                    auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                    col ^= row % (kSwizzleCDMode / 16);

                    auto smem_ptr = smem_base_ptr +
                                    warp_idx * 32 * kSwizzleCDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                    // Read 8 bf16 from SMEM
                    uint32_t values[4];  // 4 × uint32 = 4 × 2 bf16 = 8 bf16
                    ptx::ld_shared(smem_ptr, values[0], values[1], values[2], values[3]);

                    // Get global row for sum lookup
                    uint32_t global_row = global_m + warp_idx * 32 + (i / 8);
                    // Actually: row in tile = warp_idx*32 + (i/8) is wrong...
                    // row = i/8 + lane_idx (from kHasShortcut), so:
                    // tile_row = warp_idx * 32 + row  (each warp covers 32 rows)
                    // But row already includes lane_idx, so: tile_row = warp_idx * (STORE_BLOCK_M/kNumWarps) + row
                    // For STORE_BLOCK_M=128, kNumWarps=4: warp covers 32 rows
                    // row = i/8 + lane_idx, ranges 0..31 within warp's 32 rows
                    uint32_t tile_row = warp_idx * 32 + row;
                    uint32_t gm = global_m + tile_row;

                    if (gm < shape_m) {
                        float sum_val = sum_buffer[gm * 2 + sum_idx];
                        float rms = rsqrtf(sum_val / static_cast<float>(norm_dim) + eps);

                        // Apply norm to 8 bf16 values
                        nv_bfloat16* bf16_vals = reinterpret_cast<nv_bfloat16*>(values);
                        uint32_t n_base = n_col + s * STORE_BLOCK_N;
                        #pragma unroll
                        for (uint32_t e = 0; e < kNumElemsPerBankGroup; ++e) {
                            float x = __bfloat162float(bf16_vals[e]);
                            uint32_t weight_col = n_base + e;  // simplified; actual col mapping needed
                            float w = norm_weight[weight_col];
                            bf16_vals[e] = __float2bfloat16_rn(x * rms * w);
                        }

                        // Write back to SMEM
                        ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                    }
                }
            }
            __syncthreads();
        }

        // ── Step 3: TMA store scatter to peer HBM ──
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumThreads, 0);
        if (warp_idx == 0 and cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t s = 0; s < BLOCK_N / STORE_BLOCK_N; ++s) {
                cute::SM90_TMA_STORE_2D::copy(
                    &scatter_maps.maps[dst_rank],
                    reinterpret_cast<cd_dtype_t*>(smem_cd[tma_stage_idx]) + s * STORE_BLOCK_M * STORE_BLOCK_N,
                    base_n_idx + s * STORE_BLOCK_N,
                    base_m_idx);
            }
            cute::tma_store_arrive();
        }
        __syncwarp();

        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStages;
    }

    // ── Drain all scatter stores ──
    ptx::tma_store_wait<0>();
    cutlass::arch::NamedBarrier::sync(kNumThreads, 0);

    // ── Final NVLink barrier ──
    constexpr uint32_t kFinalBarrierTag = 62;
    comm::nvlink_barrier<kNumRanks, 128, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
