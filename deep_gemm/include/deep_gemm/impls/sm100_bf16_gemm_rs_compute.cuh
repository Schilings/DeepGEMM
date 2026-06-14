#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;
using namespace deep_gemm::math;

// ============================================================================================
//  sm100_bf16_gemm_rs_compute_impl — GEMM-only kernel for dual-kernel GEMM+RS
// ============================================================================================
//
//  Part 1 of the dual-kernel architecture (v3):
//    - This kernel does ONLY GEMM computation + epilogue scatter write + flag signaling
//    - No Comm/Reduce warps — full 256T dedicated to GEMM for maximum throughput
//    - Epilogue scatter writes partial results to remote ranks via NVLink P2P
//    - Per-tile ready flags signal completion to the RS reduce kernel
//
//  Warp Layout (256T = 8 warps, same as standard 2SM GEMM):
//
//    W0: TMA Load A+B (elect_one)       — 32T, 40 regs
//    W1: MMA Issue (is_leader_cta)      — 32T, 40 regs
//    W2: Reserved / TMEM Allocator      — 32T, 40 regs
//    W3: Reserved                       — 32T, 40 regs
//    W4-W7: Epilogue Warps              — 128T, 208 regs
//
//  Register budget (SM100 Max = 64512):
//    40 x 128 (non-epi) + 208 x 128 (epilogue) = 5120 + 26624 = 31744
//    Plentiful headroom — ~252 regs/thread for non-epi, no spilling!
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_compute_impl(const uint32_t shape_m_per_rank,
                                  const uint32_t runtime_m_per_rank,
                                  const uint32_t shape_n,
                                  const uint32_t shape_k,
                                  cd_dtype_t* __restrict__ output,
                                  const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                  const __grid_constant__ cute::TmaDescriptor tensor_map_b) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 2, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;
    using ab_dtype_t = cutlass::bfloat16_t;

    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype for accumulation");

    // ── Constants ──
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

    // Warp layout (standard GEMM, no comm warps)
    constexpr uint32_t kNumNonEpiWarps = kNumNonEpilogueThreads / 32;   // 4
    constexpr uint32_t kNumEpiWarps = kNumEpilogueThreads / 32;          // 4
    constexpr uint32_t kLoadWarpIdx = 0;           // W0: unified TMA load (A+B)
    constexpr uint32_t kMMAWarpIdx = 1;            // W1
    constexpr uint32_t kReservedWarpIdx = 2;        // W2: TMEM Allocator
    constexpr uint32_t kEpilogueWarpStart = kNumNonEpiWarps;  // W4

    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K for BF16");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Non-epilogue must be 128 threads (4 warps)");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kNumEpilogueThreads;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Register budget (no comm warps = more for GEMM)
    constexpr uint32_t kNumNonEpiRegisters = 40;
    constexpr uint32_t kNumEpiRegisters = 208;

    // ── Runtime variables ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t num_n_slices = BLOCK_N / STORE_BLOCK_N;
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t cta_rank = cute::block_rank_in_cluster();
    constexpr uint32_t kNumClusters = kNumSMs / kNumMulticast;
    const uint32_t cluster_idx = blockIdx.x / kNumMulticast;
    const uint32_t sm_idx = cluster_idx;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // ── Prefetch TMA descriptors ──
    if (warp_idx == kLoadWarpIdx) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }

    // ── Shared memory layout ──
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
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(kNumMulticast);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kReservedWarpIdx) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── Initial NVLink barrier ──
    constexpr uint32_t kInitBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // ── Pipeline state ──
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ── Block scheduling: round-robin interleaved (same as single-kernel) ──
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        const uint32_t m_blocks_per_cluster = kNumMulticast;
        const uint32_t num_m_pairs_per_rank = num_m_blocks_per_rank / m_blocks_per_cluster;
        const uint32_t tiles_per_rank = num_m_pairs_per_rank * num_n_blocks;
        const uint32_t total_cluster_tiles = tiles_per_rank * kNumRanks;

        if (block_idx >= total_cluster_tiles)
            return false;

        const uint32_t local_tile_idx = block_idx / kNumRanks;
        const uint32_t rank_offset = block_idx % kNumRanks;
        const uint32_t dst_rank = (rank_offset + 1 < kNumRanks) ?
            (rank_idx + rank_offset + 1) % kNumRanks : rank_idx;

        const uint32_t local_m_pair_idx = local_tile_idx / num_n_blocks;
        n_block_idx = local_tile_idx - local_m_pair_idx * num_n_blocks;
        const uint32_t local_m_block_idx = local_m_pair_idx * m_blocks_per_cluster + cta_rank;
        m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;

        block_idx += kNumClusters;
        ++ iter_idx;
        return true;
    };

    // ════════════════════════════════════════════════════════════════
    //  Warp 0 (TMA Load): Load both A and B into smem
    // ════════════════════════════════════════════════════════════════
    if (warp_idx == kLoadWarpIdx and cute::elect_one_sync()) {
        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);

            uint32_t load_m_idx = global_m;
            uint32_t load_n_idx = n_idx;
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
    //  Warp 2 (MMA Issue): Execute UMMA FMA
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == kMMAWarpIdx and is_leader_cta) {
        auto instr_desc = kSwapAB ?
            cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                        UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>() :
            cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
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
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                            a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                            b_desc_base_lo, 0, k * UMMA_K);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        }
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

    // Warp 3 (Reserved): TMEM allocation only
    else if (warp_idx == kReservedWarpIdx) {
        // Reserved warp — TMEM allocation done above in barrier init
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 4~7 (Epilogue): TMEM → smem → scatter write + flag
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx >= kEpilogueWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumEpiRegisters>();

        const auto epilogue_warp_idx = warp_idx - kEpilogueWarpStart;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;

        constexpr uint32_t kElemsPerStore = 16 / sizeof(comm_dtype_t);
        constexpr uint32_t kRowBytesPerNSlice = STORE_BLOCK_N * sizeof(comm_dtype_t);
        constexpr uint32_t kStoresPerRow = STORE_BLOCK_N / kElemsPerStore;
        constexpr uint32_t kNumNSlices = BLOCK_N / STORE_BLOCK_N;

        uint32_t tma_stage_idx = 0;

        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const uint32_t dst_rank = m_block_idx / num_m_blocks_per_rank;
            const uint32_t local_m_block_idx = m_block_idx - dst_rank * num_m_blocks_per_rank;
            const uint32_t local_m = local_m_block_idx * BLOCK_M;

            // TMEM → smem → scatter write (same as single-kernel epilogue)
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumNSlices; ++ s) {
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Phase 1: TMEM → registers → smem
                    if (epilogue_thread_idx < STORE_BLOCK_M) {
                        auto* row_ptr = smem_base_ptr + epilogue_thread_idx * kRowBytesPerNSlice;

                        #pragma unroll
                        for (uint32_t st = 0; st < kStoresPerRow; ++ st) {
                            uint32_t tmem_col = accum_stage_idx * UMMA_N +
                                                s * STORE_BLOCK_N + st * kElemsPerStore;

                            if constexpr (cute::is_same_v<comm_dtype_t, float>) {
                                uint32_t f0, f1, f2, f3;
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_col, f0, f1, f2, f3);
                                cutlass::arch::fence_view_async_tmem_load();
                                ptx::st_shared(row_ptr + st * 16, f0, f1, f2, f3);
                            } else {
                                uint32_t f0, f1, f2, f3, f4, f5, f6, f7;
                                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_col, f0, f1, f2, f3, f4, f5, f6, f7);
                                cutlass::arch::fence_view_async_tmem_load();
                                ptx::st_shared(row_ptr + st * 16,
                                    math::cast_into_bf16_and_pack(f0, f1),
                                    math::cast_into_bf16_and_pack(f2, f3),
                                    math::cast_into_bf16_and_pack(f4, f5),
                                    math::cast_into_bf16_and_pack(f6, f7));
                            }
                        }
                    }

                    // Release TMEM stage
                    if (w == kNumMWaves - 1 and s == kNumNSlices - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // Phase 2: smem → global scatter write
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    {
                        uint32_t base_row = local_m + w * STORE_BLOCK_M;
                        uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        cute::tma_store_fence();

                        for (uint32_t row = epilogue_thread_idx; row < STORE_BLOCK_M; row += kNumUMMAStoreThreads) {
                            const uint32_t global_row = base_row + row;
                            if (global_row >= runtime_m_per_rank) break;

                            auto* smem_row = smem_base_ptr + row * kRowBytesPerNSlice;

                            if (dst_rank == rank_idx) {
                                // Self-rank: write directly to output
                                auto* dst = reinterpret_cast<void*>(
                                    output + global_row * shape_n + base_col);
                                ptx::tma_store_1d(dst, smem_row, kRowBytesPerNSlice);
                            } else {
                                // Remote rank: NVLink push to remote partial buffer
                                auto* local_ptr = workspace.get_partial_ptr<comm_dtype_t>(
                                    rank_idx, global_row, base_col);
                                auto* remote_ptr = sym_buffer.map(local_ptr, dst_rank);
                                ptx::tma_store_1d(remote_ptr, smem_row, kRowBytesPerNSlice);
                            }
                        }

                        cute::tma_store_arrive();
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

            // Wait all TMA stores, then set ready flag
            ptx::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if (dst_rank != rank_idx) {
                    auto* local_flag_ptr = workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx);
                    auto* remote_flag_ptr = reinterpret_cast<uint32_t*>(sym_buffer.map(local_flag_ptr, dst_rank));
                    ptx::st_rel_sys(remote_flag_ptr, 1u);
                } else {
                    auto* local_flag_ptr = workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx);
                    ptx::st_rel_sys(local_flag_ptr, 1u);
                }
            }
        }
    }

    // ── Final synchronization ──
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // Note: Ready flags are NOT reset here — they are reset by the RS reduce kernel
    // after consuming them. This is critical for the serial execution model where
    // torch.cuda.synchronize() is called between GEMM compute and RS reduce.
    // If flags were reset here, RS reduce would find all flags==0 and timeout.

    // Deallocate tensor memory
    if (warp_idx == kLoadWarpIdx)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
