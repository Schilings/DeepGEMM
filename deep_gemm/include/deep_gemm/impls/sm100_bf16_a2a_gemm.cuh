#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>

#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/layout/bf16_a2a_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// ============================================================================================
//  sm100_bf16_a2a_gemm_nt_impl — BF16 All2All + GEMM Fusion (Flux-style)
// ============================================================================================
//
//  Compute-only kernel with host-side CE DMA communication (same design as AG GEMM).
//
//  Architecture (kNumA2AThreads = 0, compute-only):
//
//  +----------------------------------------------------------------------+
//  |  Load A Warp (W0, elect_one):                                        |
//  |    Poll slot_state[src_rank][chunk] via ld_acq_sys, then TMA load A  |
//  |    Compute order: i, (i-1+n)%n, ..., (i+1)%n (ring order)           |
//  +----------------------------------------------------------------------+
//  |  Load B Warp (W1, elect_one):                                        |
//  |    TMA load B — no flag polling needed (weights always available).   |
//  +----------------------------------------------------------------------+
//  |  MMA Warp (W2, is_leader_cta):  UMMA tensor core issue.             |
//  |  Reserved  (W3):                TMEM allocator.                      |
//  +----------------------------------------------------------------------+
//  |  Epilogue (W4-W7, 128T):   TMEM -> smem -> TMA 2D store to output.  |
//  +----------------------------------------------------------------------+
//
//  Host-side communication (independent comm stream):
//    1. cudaMemsetAsync: clear slot_state flags
//    2. Copy local_x[rank_idx] -> slot[rank_idx], set flags (local ready)
//    3. For each remote rank j: copy j's local_x[rank_idx] -> slot[j], set flags
//    4. Kernel polls per-chunk flags before TMA loading A
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumA2AThreads,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
__global__ void __launch_bounds__(kNumA2AThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 2)
sm100_bf16_a2a_gemm_nt_impl(void* d,
                            const uint32_t shape_m_per_rank,
                            const uint32_t runtime_m_per_rank,
                            const uint32_t shape_n,
                            const uint32_t shape_k,
                            const uint32_t num_slots,
                            const uint32_t ready_chunk_rows,
                            const uint32_t num_ready_chunks,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    using ab_dtype_t = cutlass::bfloat16_t;

    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / kNumMulticast;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    constexpr uint32_t kNumA2AWarps = kNumA2AThreads / 32;
    constexpr uint32_t kGemmWarpBase = kNumA2AWarps;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumReadyChunksPerSlot = layout::BF16A2AGemmWorkspace::kNumReadyChunksPerSlot;
    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64,
                     "BF16 A2A+GEMM expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumA2AThreads == 0, "Flux-style A2A GEMM has no in-kernel comm threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128,
                     "Non-epi=128, Epi=128");
    DG_STATIC_ASSERT(kNumReadyChunksPerSlot == 4, "Unexpected ready chunk count");

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto workspace = layout::BF16A2AGemmWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_k, num_slots);

    // -- Prefetch TMA descriptors --
    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    // -- Shared memory layout --
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

    // -- Initialize barriers --
    if (warp_idx == kGemmWarpBase + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(2 * kNumMulticast);  // 2 producers (A+B) x kNumMulticast
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kGemmWarpBase + 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // -- Tile scheduling: A2A ring order --
    // Compute order for rank i: i, (i-1+n)%n, (i-2+n)%n, ..., (i+1)%n
    // Self-rank tiles first (always ready), then ring in reverse direction.
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
        const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
        const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        if constexpr (kNumMulticast > 1) {
            // 2-CTA cluster: pair consecutive blocks, same N, adjacent M
            const uint32_t pair_idx = block_idx / kNumMulticast;
            const uint32_t cta_in_pair = block_idx % kNumMulticast;
            const uint32_t m_pairs_per_rank = num_m_blocks_per_rank / kNumMulticast;
            const uint32_t tiles_per_rank = m_pairs_per_rank * num_n_blocks;
            const uint32_t logical_m_pair = pair_idx / num_n_blocks;
            n_block_idx = pair_idx - logical_m_pair * num_n_blocks;
            // A2A ring order: compute self first, then (i-1), (i-2), ..., (i+1)
            const uint32_t rank_step = logical_m_pair / m_pairs_per_rank;
            const uint32_t src_rank = (rank_idx + kNumRanks - rank_step) % kNumRanks;
            const uint32_t m_pair_within_rank = logical_m_pair % m_pairs_per_rank;
            m_block_idx = (src_rank * m_pairs_per_rank + m_pair_within_rank) * kNumMulticast + cta_in_pair;
        } else {
            const uint32_t tiles_per_rank = num_m_blocks_per_rank * num_n_blocks;
            const uint32_t rank_step = block_idx / tiles_per_rank;
            const uint32_t within = block_idx % tiles_per_rank;
            // A2A ring order: rank_idx, (rank_idx-1+n)%n, (rank_idx-2+n)%n, ...
            const uint32_t src_rank = (rank_idx + kNumRanks - rank_step) % kNumRanks;
            const uint32_t local_m_block = within / num_n_blocks;
            n_block_idx = within - local_m_block * num_n_blocks;
            m_block_idx = src_rank * num_m_blocks_per_rank + local_m_block;
        }
        block_idx += kNumSMs;
        ++ iter_idx;
        return true;
    };

    // -- W0: Load A Warp — poll per-chunk flag + TMA load A --
    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t src_rank = global_m / runtime_m_per_rank;
            const uint32_t local_m = global_m - src_rank * runtime_m_per_rank;
            const uint32_t slot_m = src_rank * shape_m_per_rank + local_m;
            // Per-chunk barrier polling: wait until all chunks covering this tile are ready
            const uint32_t chunk_start = local_m / ready_chunk_rows;
            const uint32_t chunk_end = cute::min<uint32_t>((local_m + BLOCK_M - 1) / ready_chunk_rows, num_ready_chunks - 1);
            #pragma unroll
            for (uint32_t chunk_idx = 0; chunk_idx < kNumReadyChunksPerSlot; ++ chunk_idx) {
                if (chunk_idx >= chunk_start and chunk_idx <= chunk_end)
                    while (ptx::ld_acq_sys(workspace.get_slot_state_ptr(src_rank, chunk_idx)) == 0);
            }
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, slot_m, kNumMulticast);
                if (is_leader_cta)
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * kNumMulticast);
                else
                    full_barriers[stage_idx]->arrive(0u);
            }
        }
    // -- W1: Load B Warp — TMA load B (no flag polling) --
    } else if (warp_idx == kGemmWarpBase + 1 and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            uint32_t n_idx = n_block_idx * BLOCK_N;
            if constexpr (kNumMulticast > 1)
                n_idx += cute::block_rank_in_cluster() * LOAD_BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, kNumMulticast);
                if (is_leader_cta)
                    full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE * kNumMulticast);
                else
                    full_barriers[stage_idx]->arrive(0u);
            }
        }
    // -- W2: MMA Issue Warp — UMMA tensor core --
    } else if (warp_idx == kGemmWarpBase + 2 and is_leader_cta) {
        auto instr_desc = cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto a_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
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
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        a_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                   k_block_idx > 0 or k > 0, runtime_instr_desc);
                    }
                }
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }
    // -- W3: Reserved / TMEM allocator (idle after init) --

    // -- Epilogue: TMEM -> smem -> TMA 2D store to output --
    } else if (warp_idx >= (kNumA2AThreads + kNumNonEpilogueThreads) / 32 and
               warp_idx < (kNumA2AThreads + kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (kNumA2AThreads + kNumNonEpilogueThreads) / 32;
        uint32_t tma_stage_idx = 0;
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();
            const uint32_t base_m_idx = m_block_idx * BLOCK_M;
            const uint32_t base_n_idx = n_block_idx * BLOCK_N;
            epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                GemmType::Normal, false,
                cd_dtype_t, epilogue::transform::EpilogueIdentity>
            (smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N,
             base_m_idx, base_n_idx, 0,
             epilogue_warp_idx, lane_idx,
             tmem_empty_barriers[accum_stage_idx],
             tensor_map_d);
        }
    }

    // -- Cleanup --
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
