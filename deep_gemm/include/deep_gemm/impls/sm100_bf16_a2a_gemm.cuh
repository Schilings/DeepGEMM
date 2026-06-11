#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/layout/bf16_a2a_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// ============================================================================================
//  sm100_bf16_a2a_gemm_nt_impl — BF16 All2All + GEMM Fusion (Ulysses SP)
// ============================================================================================
//
//  5-Warp Architecture (Ring-Push + Compute Overlap):
//
//  ┌────────────────────────────────────────────────────────────────────────┐
//  │  Push Warps (W0-W3, 128T):                                            │
//  │    Ring-push local chunks to remote ranks via NVLink.                  │
//  │    Push order: (i+1), (i+2), ..., (i+n-1), self.                      │
//  │    All SMs cooperate globally (strided copy). Atomic counter signal.  │
//  │    Runs CONCURRENTLY with GEMM pipeline.                              │
//  ├────────────────────────────────────────────────────────────────────────┤
//  │  Load A Warp (W4, elect_one):                                         │
//  │    Poll slot_state[src_rank] >= kNumSMs, then TMA load A from slot.  │
//  │    Compute order: i, (i-1+n)%n, (i-2+n)%n, ..., (i+1)%n             │
//  ├────────────────────────────────────────────────────────────────────────┤
//  │  Load B Warp (W5, elect_one):                                         │
//  │    TMA load B — no flag polling needed (weights always available).    │
//  ├────────────────────────────────────────────────────────────────────────┤
//  │  MMA Warp (W6, elect_one):  UMMA tensor core issue.                   │
//  │  Reserved  (W7):            TMEM allocator.                           │
//  ├────────────────────────────────────────────────────────────────────────┤
//  │  Epilogue (W8-W11, 128T):   TMEM → smem → TMA 2D store to output.   │
//  └────────────────────────────────────────────────────────────────────────┘
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumA2AThreads,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
__global__ void __launch_bounds__(kNumA2AThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_a2a_gemm_nt_impl(void* d,
                            const uint32_t shape_m_per_rank,
                            const uint32_t runtime_m_per_rank,
                            const uint32_t shape_n,
                            const uint32_t shape_k,
                            const uint32_t num_slots,
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                            const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;
    using ab_dtype_t = cutlass::bfloat16_t;

    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    constexpr uint32_t kNumA2AWarps = kNumA2AThreads / 32;
    constexpr uint32_t kGemmWarpBase = kNumA2AWarps;   // W4
    constexpr uint32_t kNumThreads = kNumA2AThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumEpilogueStages = 2;
    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64,
                     "BF16 A2A+GEMM expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumA2AThreads % 32 == 0 and kNumA2AThreads >= 128,
                     "Need at least 128 push threads (4 warps)");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128,
                     "Non-epi=128, Epi=128");

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

    // ── Prefetch TMA descriptors ──
    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_d);
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
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(
        smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 2 + kNumEpilogueStages * 2);

    // ── Initialize barriers (by MMA warp W6) ──
    // full_barriers: init(2) because BOTH Load A and Load B arrive separately
    // empty_barriers: init(1) because only MMA warp arrives
    if (warp_idx == kGemmWarpBase + 2 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(2);   // Load A + Load B each arrive
            empty_barriers[i]->init(1);  // MMA arrives
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kGemmWarpBase + 3) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    // ── Clear slot states + NVLink barrier ──
    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < kNumRanks; i += kNumSMs * kNumThreads)
        workspace.get_slot_state_ptr(i)[0] = 0;
    constexpr uint32_t kAfterStateCleanBarrierTag = 51;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kAfterStateCleanBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // ════════════════════════════════════════════════════════════════
    //  W0-W3 (128T): Push Warps — Ring-push to remote ranks
    // ════════════════════════════════════════════════════════════════
    //
    //  Push order: (i+1), (i+2), ..., (i+n-1), then self (i).
    //  Remote pushes go FIRST so remote ranks get data ASAP.
    //  Self copy goes LAST (GEMM computes self first anyway).
    //
    //  All SMs cooperate globally with strided copy for each chunk.
    //  Each SM atomicAdd(1) to remote flag when its portion is done.
    //  Slot ready when flag == kNumSMs.
    //
    if (warp_idx < kNumA2AWarps) {
        const uint64_t chunk_bytes = static_cast<uint64_t>(runtime_m_per_rank) * shape_k * sizeof(nv_bfloat16);
        const uint64_t num_vecs = chunk_bytes / sizeof(uint4);
        const uint64_t global_idx_base = static_cast<uint64_t>(sm_idx) * kNumA2AThreads + (warp_idx * 32u + lane_idx);
        const uint64_t global_total = static_cast<uint64_t>(kNumSMs) * kNumA2AThreads;

        #pragma unroll 1
        for (uint32_t step = 0; step < kNumRanks; ++ step) {
            // Push order: (i+1), (i+2), ..., (i+n-1), i
            const uint32_t dst_rank = (rank_idx + 1 + step) % kNumRanks;

            const auto* src_vec = workspace.template get_local_x_ptr<uint4>(dst_rank);
            auto* dst_vec = reinterpret_cast<uint4*>(
                dst_rank == rank_idx
                    ? workspace.get_slot_x_ptr(rank_idx)
                    : sym_buffer.map(workspace.get_slot_x_ptr(rank_idx), dst_rank));

            // Global strided copy: all SMs cooperate for max bandwidth
            for (uint64_t i = global_idx_base; i < num_vecs; i += global_total)
                dst_vec[i] = src_vec[i];

            // Per-SM sync (only push threads, not cross-SM)
            cutlass::arch::NamedBarrier::sync(kNumA2AThreads, 1);

            // Atomic counter signal
            if (dst_rank != rank_idx)
                __threadfence_system();
            if (thread_idx == 0) {
                if (dst_rank == rank_idx)
                    ptx::red_add_rel(workspace.get_slot_state_ptr(rank_idx), 1);
                else
                    ptx::red_add_rel(sym_buffer.map(workspace.get_slot_state_ptr(rank_idx), dst_rank), 1);
            }
        }
        // Push warps done — idle until kernel finishes
    }

    // ════════════════════════════════════════════════════════════════
    //  GEMM Pipeline — compute order: i, (i-1+n)%n, ..., (i+1)%n
    // ════════════════════════════════════════════════════════════════

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Tile scheduling: compute order = (i, i-1, i-2, ..., i+1)
    // Self-rank tiles first (always ready), then reverse ring.
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
        const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
        const uint32_t tiles_per_rank = num_m_blocks_per_rank * num_n_blocks;
        const uint32_t total_tiles = tiles_per_rank * kNumRanks;

        if (block_idx >= total_tiles)
            return false;

        // Map block_idx to (rank_step, within_rank_tile)
        const uint32_t rank_step = block_idx / tiles_per_rank;
        const uint32_t within = block_idx % tiles_per_rank;

        // Compute order: rank_idx, (rank_idx-1+n)%n, (rank_idx-2+n)%n, ...
        const uint32_t src_rank = (rank_idx + kNumRanks - rank_step) % kNumRanks;

        const uint32_t local_m_block = within / num_n_blocks;
        n_block_idx = within - local_m_block * num_n_blocks;
        m_block_idx = src_rank * num_m_blocks_per_rank + local_m_block;

        block_idx += kNumSMs;
        ++ iter_idx;
        return true;
    };

    // ── W4: Load A Warp — poll flag + TMA load A ──
    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t src_rank = global_m / runtime_m_per_rank;
            const uint32_t local_m = global_m - src_rank * runtime_m_per_rank;

            // Poll: wait until all SMs finished pushing this slot
            while (ptx::ld_acq_sys(workspace.get_slot_state_ptr(src_rank)) < kNumSMs);

            const uint32_t slot_m = src_rank * shape_m_per_rank + local_m;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, slot_m, 1);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
            }
        }
    }

    // ── W5: Load B Warp — TMA load B (no flag polling) ──
    else if (warp_idx == kGemmWarpBase + 1 and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
            }
        }
    }

    // ── W6: MMA Issue Warp — UMMA tensor core ──
    else if (warp_idx == kGemmWarpBase + 2 and is_leader_cta) {
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
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();
                using mma_t = ptx::SM100_MMA_F16BF16_SS;
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
    }

    // ── W7: Reserved / TMEM allocator (idle after init) ──

    // ── W8-W11: Epilogue — TMEM → smem → TMA 2D store to output ──
    else if (warp_idx >= (kNumA2AThreads + kNumNonEpilogueThreads) / 32 and
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

    // ── Cleanup ──
    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
