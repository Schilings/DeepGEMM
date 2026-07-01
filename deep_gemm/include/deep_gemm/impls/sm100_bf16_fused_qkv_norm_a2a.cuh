#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cuda_device_runtime_api.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/fused_qkv_norm_a2a.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;
using namespace deep_gemm::math;

// Host-side mirror of scatter maps for QKV (GQA-aware).
struct FusedQKVNormA2AScatterMaps {
    cute::TmaDescriptor maps[8];
};

// ============================================================================================
//  sm100_bf16_fused_qkv_norm_a2a_impl — Single kernel, norm-deferred
//
//  Based on bf16_gemm_a2a_transpose_nt with these changes:
//    1. Epilogue uses EpilogueX2Sum (pre_cast computes x² partial sum + atomic add)
//    2. GQA-aware scatter: Q/K/V segments scatter independently
//    3. After final barrier: compute rms + scatter rms to peers (fused in epilogue)
//
//  Data flow:
//    GEMM → epilogue: TMEM→SMEM→(pre_cast: x²sum)→cast→TMA scatter x to peer
//    nvlink_barrier (all tiles done, sum_buffer complete)
//    rms = rsqrt(sum/dim + eps) → scatter rms to peer's rms region (P2P store)
//    nvlink_barrier (rms globally visible)
//
//  Peer side (separate lightweight kernel or Python):
//    out = x * rms * weight (elementwise norm)
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          bool kDoNormQ, bool kDoNormK,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_fused_qkv_norm_a2a_impl(const uint32_t shape_m,
                                    const uint32_t shape_n,
                                    const uint32_t shape_k,
                                    const uint32_t bs,
                                    const uint32_t seq,
                                    const uint32_t local_seq,
                                    const uint32_t q_dim,
                                    const uint32_t kv_dim,
                                    const uint32_t local_q_n,
                                    const uint32_t local_kv_n,
                                    const float eps,
                                    float* __restrict__ sum_buffer,  // [shape_m, 2] local
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                    const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                                    const __grid_constant__ FusedQKVNormA2AScatterMaps scatter_maps) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 2, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;
    using ab_dtype_t = cutlass::bfloat16_t;

    // ── Constants (identical to GEMM-A2A-transpose) ──
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    constexpr uint32_t kNumNonEpiWarps = kNumNonEpilogueThreads / 32;
    constexpr uint32_t kLoadWarpIdx = 0;
    constexpr uint32_t kMMAWarpIdx = 1;
    constexpr uint32_t kReservedWarpIdx = 2;
    constexpr uint32_t kEpilogueWarpStart = kNumNonEpiWarps;

    constexpr uint32_t STORE_BLOCK_M = kSwapAB ? 16 : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kNumEpilogueThreads;
    constexpr uint32_t kNumEpiRegisters = 208;

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // ── Runtime variables ──
    const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t cta_rank = cute::block_rank_in_cluster();
    constexpr uint32_t kNumClusters = kNumSMs / kNumMulticast;
    const uint32_t cluster_idx = blockIdx.x / kNumMulticast;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const uint32_t local_n_total = local_q_n + 2 * local_kv_n;

    const auto workspace = layout::FusedQKVNormA2AWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, bs, seq, local_n_total, sizeof(comm_dtype_t));

    kNumMulticast > 1 ? cute::cluster_sync() : void();

    if (warp_idx == kLoadWarpIdx) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }

    // ── Shared memory layout (same as GEMM-A2A-transpose) ──
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<ab_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2);

    // ── Initialize barriers ──
    if (warp_idx == kMMAWarpIdx and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i) {
            full_barriers[i]->init(kNumMulticast);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kReservedWarpIdx) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── Initial NVLink barrier ──
    constexpr uint32_t kInitBarrierTag = 71;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // ── Build epilogue context for EpilogueX2Sum ──
    using epilogue_type_t = epilogue::transform::EpilogueX2Sum;
    typename epilogue_type_t::Context epi_ctx{
        .sum_buffer = sum_buffer,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
    };

    // ── Pipeline state ──
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // GQA-aware block scheduler
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        const uint32_t m_blocks_per_cluster = kNumMulticast;
        const uint32_t num_m_pairs = ceil_div(num_m_blocks, m_blocks_per_cluster);
        const uint32_t total_cluster_tiles = num_m_pairs * num_n_blocks;
        if (block_idx >= total_cluster_tiles) return false;
        const uint32_t m_pair_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx - m_pair_idx * num_n_blocks;
        m_block_idx = m_pair_idx * m_blocks_per_cluster + cta_rank;
        block_idx += kNumClusters;
        ++iter_idx;
        return true;
    };

    // ════════════════════════════════════════════════════════════════
    //  W0: TMA Load A+B (identical to GEMM-A2A-transpose)
    // ════════════════════════════════════════════════════════════════
    if (warp_idx == kLoadWarpIdx and cute::elect_one_sync()) {
        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            uint32_t load_m_idx = global_m, load_n_idx = n_idx;
            if constexpr (kNumMulticast > 1) {
                load_n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
            }
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  W1: MMA Issue (identical to GEMM-A2A-transpose)
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == kMMAWarpIdx and is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                                      UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        auto umma_arrive = [](const uint64_t* barrier) {
            constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };
        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N, k_block_idx > 0 or k > 0, runtime_instr_desc);
                    }
                }
                __syncwarp();
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }
        if constexpr (kNumMulticast > 1) {
            const auto iter_val = iter_idx - 1;
            if (iter_val >= 0) {
                const auto accum_phase_idx = (iter_val / kNumEpilogueStages) & 1;
                tmem_empty_barriers[iter_val % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    }

    else if (warp_idx == kReservedWarpIdx) {}

    // ════════════════════════════════════════════════════════════════
    //  W4-W7: Epilogue with EpilogueX2Sum + GQA-aware scatter
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx >= kEpilogueWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpiRegisters>();
        const auto epilogue_warp_idx = warp_idx - kEpilogueWarpStart;
        uint32_t tma_stage_idx = 0;
        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t b = global_m / local_seq;
            const uint32_t s_local = global_m - b * local_seq;
            const uint32_t n_col = n_block_idx * BLOCK_N;

            // GQA-aware: determine dst_rank and base_n_idx based on Q/K/V segment
            uint32_t dst_rank, base_n_idx;
            if (n_col < q_dim) {
                // Q segment
                dst_rank = n_col / local_q_n;
                base_n_idx = n_col % local_q_n;
            } else if (n_col < q_dim + kv_dim) {
                // K segment
                uint32_t rel = n_col - q_dim;
                dst_rank = rel / local_kv_n;
                base_n_idx = rel % local_kv_n + local_q_n;
            } else {
                // V segment
                uint32_t rel = n_col - q_dim - kv_dim;
                dst_rank = rel / local_kv_n;
                base_n_idx = rel % local_kv_n + local_q_n + local_kv_n;
            }
            const uint32_t base_m_idx = b * seq + rank_idx * local_seq + s_local;

            // Use EpilogueX2Sum instead of EpilogueIdentity — this is the ONLY change
            // from bf16_gemm_a2a_transpose_nt's epilogue (plus GQA-aware scatter index).
            epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                comm_dtype_t, epilogue::transform::EpilogueX2Sum>
            (smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N,
             base_m_idx, base_n_idx, 0,
             epilogue_warp_idx, lane_idx,
             tmem_empty_barriers[accum_stage_idx],
             scatter_maps.maps[dst_rank],
             epi_ctx);  // ← pass x²sum context
        }
        ptx::tma_store_wait<0>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
    }

    // ════════════════════════════════════════════════════════════════
    //  Final barrier: all tiles done, sum_buffer complete
    // ════════════════════════════════════════════════════════════════
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFinalBarrierTag = 72;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // ════════════════════════════════════════════════════════════════
    //  Post-barrier: compute rms + scatter rms to peers (fused in epilogue)
    //
    //  rms_q[row] = rsqrt(sum_buffer[row, 0] / q_dim + eps)
    //  rms_k[row] = rsqrt(sum_buffer[row, 1] / kv_dim + eps)
    //  Then scatter rms to each peer's rms region at the right seq offset.
    //
    //  Only SM 0 participates (lightweight, data is tiny: bs*local_seq*2 floats).
    //  Uses direct global stores (P2P via NVLink) — no TMA needed for this tiny data.
    // ════════════════════════════════════════════════════════════════
    if (blockIdx.x == 0) {
        // Each thread handles a few rows
        const uint32_t local_m = bs * local_seq;
        const uint32_t rows_per_thread = (local_m + kNumThreads - 1) / kNumThreads;

        for (uint32_t r = 0; r < rows_per_thread; ++r) {
            uint32_t row = r * kNumThreads + thread_idx;
            if (row >= local_m) break;

            // Compute rms
            float rms_q_val = 0.f, rms_k_val = 0.f;
            if constexpr (kDoNormQ) {
                float sum_q = sum_buffer[row * 2 + 0];
                rms_q_val = rsqrtf(sum_q / static_cast<float>(q_dim) + eps);
            }
            if constexpr (kDoNormK) {
                float sum_k = sum_buffer[row * 2 + 1];
                rms_k_val = rsqrtf(sum_k / static_cast<float>(kv_dim) + eps);
            }

            // Compute output row index: b*seq + rank_idx*local_seq + s_local
            // This is where this rank's data lands in each peer's rms region
            uint32_t b = row / local_seq;
            uint32_t s_local = row - b * local_seq;
            uint32_t out_row = b * seq + rank_idx * local_seq + s_local;

            // Scatter rms to all peers' rms regions
            // rms region is at sym_buffer offset 32, [bs*seq, 2] float32
            for (uint32_t d = 0; d < kNumRanks; ++d) {
                float* peer_rms_ptr = sym_buffer.template map<float*>(
                    workspace.get_rms_ptr<float>(), d);
                // Write rms_q and rms_k to peer's rms region at out_row
                peer_rms_ptr[out_row * 2 + 0] = rms_q_val;
                peer_rms_ptr[out_row * 2 + 1] = rms_k_val;
            }
        }
        __syncthreads();
        // Ensure rms stores are visible before the next barrier
        __threadfence_system();
    }

    // Second barrier: rms globally visible
    constexpr uint32_t kRmsBarrierTag = 73;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 1, kRmsBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // Deallocate TMEM
    if (warp_idx == kLoadWarpIdx)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
