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
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

// ============================================================================================
//  sm100_bf16_a2a_gemm_nt_impl — BF16 All2All + GEMM Fusion (Ulysses SP: A2A + Wo)
// ============================================================================================
//
//  Communication: P2P All-to-All (each rank sends different data to each peer)
//  Computation:   Standard BF16 GEMM (NT layout)
//  Overlap:       Tile-level — GEMM starts processing tiles as soon as their
//                 source rank's data arrives in the local slot buffer.
//
//  Warp Layout (same as AG+GEMM):
//    W0-W3 (128T): A2A Comm warps — P2P scatter via NVLink
//    W4: TMA Load warp — loads A (from slots) + B, waits for slot_state
//    W5: MMA Issue warp — UMMA tensor core
//    W6: Reserved / TMEM allocator
//    W7+: Epilogue warps — TMEM → smem → TMA store to output
//
//  Key difference from AG+GEMM:
//    AG:  Ring relay — each step forwards one rank's data to next_rank
//    A2A: P2P direct — rank i writes chunk[j] to rank j's slot[i] (parallel scatter)
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
    constexpr uint32_t kGemmWarpBase = kNumA2AWarps;
    constexpr uint32_t kNumThreads = kNumA2AThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumEpilogueStages = 2;
    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64, "BF16 A2A+GEMM expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumA2AThreads % 32 == 0 and kNumA2AThreads >= 128, "Invalid A2A threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128, "Invalid GEMM thread layout");

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

    // ── Shared memory layout (same as AG+GEMM) ──
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
    if (warp_idx == kGemmWarpBase + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kGemmWarpBase + 2) {
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
    //  W0-W3: All-to-All Communication Warps (P2P Direct Scatter)
    // ════════════════════════════════════════════════════════════════
    //
    //  Each rank has local_x[num_ranks, M_per_rank, K].
    //  Chunk j (= local_x[j]) must be sent to rank j's slot[rank_idx].
    //
    //  P2P Direct: all ranks write concurrently to their destinations.
    //  No ring relay needed — NVLink Gen5 is fully connected.
    //
    constexpr uint32_t kA2AGridSyncIndex = 1;
    auto a2a_sync_scope = []() { cutlass::arch::NamedBarrier::sync(kNumA2AThreads, 1); };
    if (warp_idx < kNumA2AWarps) {
        auto copy_16B = [&](void* dst, const void* src, const uint64_t& num_bytes) {
            auto* dst_vec = reinterpret_cast<uint4*>(dst);
            const auto* src_vec = reinterpret_cast<const uint4*>(src);
            const uint64_t num_vecs = num_bytes / sizeof(uint4);
            const uint64_t global_thread_idx = static_cast<uint64_t>(sm_idx) * kNumA2AThreads + thread_idx;
            const uint64_t global_num_threads = static_cast<uint64_t>(kNumSMs) * kNumA2AThreads;
            for (uint64_t i = global_thread_idx; i < num_vecs; i += global_num_threads)
                dst_vec[i] = src_vec[i];
        };

        const uint64_t chunk_bytes = static_cast<uint64_t>(runtime_m_per_rank) * shape_k * sizeof(nv_bfloat16);

        // Phase 1: Self-copy (local chunk → local slot)
        // Rank i's chunk[rank_idx] stays local → slot[rank_idx]
        copy_16B(workspace.get_slot_x_ptr(rank_idx), workspace.get_local_x_ptr(rank_idx), chunk_bytes);
        comm::grid_sync<kNumSMs, kA2AGridSyncIndex>(workspace, sm_idx, thread_idx, a2a_sync_scope);
        if (sm_idx == 0 and thread_idx == 0)
            ptx::red_add_rel(workspace.get_slot_state_ptr(rank_idx), 1);

        // Phase 2: P2P scatter — send chunk[dst_rank] to each remote rank
        // We iterate over all other ranks and push our data to their slot[rank_idx].
        #pragma unroll 1
        for (uint32_t step = 0; step + 1 < kNumRanks; ++ step) {
            // dst_rank: which remote rank to send to
            const uint32_t dst_rank = (rank_idx + 1 + step) % kNumRanks;

            // Source: our local chunk destined for dst_rank
            const void* src = workspace.get_local_x_ptr(dst_rank);

            // Destination: slot[rank_idx] on dst_rank (via NVLink)
            auto* local_slot_ptr = workspace.get_slot_x_ptr(rank_idx);
            auto* remote_slot_ptr = sym_buffer.map(local_slot_ptr, dst_rank);

            copy_16B(remote_slot_ptr, src, chunk_bytes);

            // Grid sync to ensure all SMs finished this step's writes
            comm::grid_sync<kNumSMs, kA2AGridSyncIndex>(workspace, sm_idx, thread_idx, a2a_sync_scope);

            // Signal: set slot_state[rank_idx] on dst_rank
            if (sm_idx == 0 and thread_idx == 0) {
                __threadfence_system();
                auto* remote_state = sym_buffer.map(workspace.get_slot_state_ptr(rank_idx), dst_rank);
                *remote_state = 1;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  GEMM Pipeline (identical to AG+GEMM)
    // ════════════════════════════════════════════════════════════════

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        const uint32_t num_m_blocks = ceil_div(shape_m, BLOCK_M);
        const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        m_block_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx - m_block_idx * num_n_blocks;
        block_idx += kNumSMs;
        ++ iter_idx;
        return true;
    };

    // ── TMA Load Warp (W4): loads A from slots + B ──
    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            // Which slot does this M-tile come from?
            const uint32_t src_rank = global_m / runtime_m_per_rank;
            const uint32_t local_m = global_m - src_rank * runtime_m_per_rank;

            // Wait for this rank's data to arrive in our slot
            while (ptx::ld_acq_sys(workspace.get_slot_state_ptr(src_rank)) == 0);

            const uint32_t n_idx = n_block_idx * BLOCK_N;
            // TMA coordinate: slot_m indexes into the concatenated slots buffer
            const uint32_t slot_m = src_rank * shape_m_per_rank + local_m;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, slot_m, 1);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    }

    // ── MMA Issue Warp (W5) ──
    else if (warp_idx == kGemmWarpBase + 1 and is_leader_cta) {
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

    // ── Epilogue Warps (W8-W11): TMEM → smem → TMA store to output ──
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
