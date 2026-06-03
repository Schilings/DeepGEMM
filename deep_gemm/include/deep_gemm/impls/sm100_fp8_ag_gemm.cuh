#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/ag_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumAGThreads,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kGranK,
          typename cd_dtype_t>
__global__ void __launch_bounds__(kNumAGThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_ag_gemm_nt_impl(void* d,
                          const uint32_t shape_m_per_rank,
                          const uint32_t runtime_m_per_rank,
                          const uint32_t shape_n,
                          const uint32_t shape_k,
                          const uint32_t num_slots,
                          const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                          const __grid_constant__ cute::TmaDescriptor tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator1Sm;
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;

    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t kNumAGWarps = kNumAGThreads / 32;
    constexpr uint32_t kGemmWarpBase = kNumAGWarps;
    constexpr uint32_t kNumThreads = kNumAGThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128, "The first AG+GEMM version expects 128x128 tiles");
    DG_STATIC_ASSERT(kNumAGThreads % 32 == 0 and kNumAGThreads >= 128, "Invalid AG threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid GEMM non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Invalid epilogue threads");
    DG_STATIC_ASSERT(kGranK == 32 or kGranK == 128, "Invalid granularity K");

    constexpr uint32_t kNumSFAStagesPerLoad = kGranK == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranK == 32 ? 1 : 4;
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t shape_sfa_k = ceil_div(shape_k, kGranK * 4);
    const uint32_t shape_sfb_k = ceil_div(shape_k, kGranK * 4);

    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const uint32_t next_rank = (rank_idx + 1) % kNumRanks;

    const auto workspace = layout::AGGemmWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_k, kGranK, num_slots);

    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "Shared memory of A/B must be aligned to 1024 bytes");
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto with_sf_full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_full_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    auto tmem_empty_barriers = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    if (warp_idx == kGemmWarpBase + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(32);
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

    for (uint32_t i = sm_idx * kNumThreads + thread_idx; i < kNumRanks; i += kNumSMs * kNumThreads)
        workspace.get_slot_state_ptr(i)[0] = 0;
    constexpr uint32_t kAfterStateCleanBarrierTag = 21;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kAfterStateCleanBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    constexpr uint32_t kAGGridSyncIndex = 1;
    auto ag_sync_scope = []() { cutlass::arch::NamedBarrier::sync(kNumAGThreads, 1); };

    if (warp_idx < kNumAGWarps) {

        auto copy_16B = [&](void* dst, const void* src, const uint64_t& num_bytes) {
            auto* dst_vec = reinterpret_cast<uint4*>(dst);
            const auto* src_vec = reinterpret_cast<const uint4*>(src);
            const uint64_t num_vecs = num_bytes / sizeof(uint4);
            const uint64_t global_thread_idx = static_cast<uint64_t>(sm_idx) * kNumAGThreads + thread_idx;
            const uint64_t global_num_threads = static_cast<uint64_t>(kNumSMs) * kNumAGThreads;
            for (uint64_t i = global_thread_idx; i < num_vecs; i += global_num_threads)
                dst_vec[i] = src_vec[i];
        };

        copy_16B(workspace.get_slot_x_ptr(rank_idx), workspace.get_local_x_ptr(),

                 static_cast<uint64_t>(runtime_m_per_rank) * shape_k);
        copy_16B(workspace.get_slot_x_sf_ptr(rank_idx), workspace.get_local_x_sf_ptr(),
                 static_cast<uint64_t>(runtime_m_per_rank) * shape_sfa_k * sizeof(uint32_t));
        comm::grid_sync<kNumSMs, kAGGridSyncIndex>(workspace, sm_idx, thread_idx, ag_sync_scope);
        if (sm_idx == 0 and thread_idx == 0)
            ptx::red_add_rel(reinterpret_cast<uint32_t*>(workspace.get_slot_state_ptr(rank_idx)), 1);

        #pragma unroll 1
        for (uint32_t step = 0; step + 1 < kNumRanks; ++ step) {
            const uint32_t src_rank = (rank_idx + kNumRanks - step) % kNumRanks;
            while (ptx::ld_acq_sys(workspace.get_slot_state_ptr(src_rank)) == 0);

            auto* remote_x = sym_buffer.map(workspace.get_slot_x_ptr(src_rank), next_rank);
            auto* remote_sf = sym_buffer.map(workspace.get_slot_x_sf_ptr(src_rank), next_rank);
            copy_16B(remote_x, workspace.get_slot_x_ptr(src_rank),
                     static_cast<uint64_t>(runtime_m_per_rank) * shape_k);
            copy_16B(remote_sf, workspace.get_slot_x_sf_ptr(src_rank),
                     static_cast<uint64_t>(runtime_m_per_rank) * shape_sfa_k * sizeof(uint32_t));
            comm::grid_sync<kNumSMs, kAGGridSyncIndex>(workspace, sm_idx, thread_idx, ag_sync_scope);
            if (sm_idx == 0 and thread_idx == 0) {
                __threadfence_system();
                auto* remote_state = sym_buffer.map(workspace.get_slot_state_ptr(src_rank), next_rank);
                *remote_state = 1;
            }

        }
    }

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

    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t src_rank = global_m / runtime_m_per_rank;
            const uint32_t local_m = global_m - src_rank * runtime_m_per_rank;
            while (ptx::ld_acq_sys(workspace.get_slot_state_ptr(src_rank)) == 0);

            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                const uint32_t slot_m = src_rank * shape_m_per_rank + local_m;
                const uint32_t n_idx = n_block_idx * BLOCK_N;

                tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, slot_m, 1);
                tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
                uint32_t num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx],
                                            local_m, src_rank * shape_sfa_k + ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx],
                                            n_block_idx * BLOCK_N, ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad));
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == kGemmWarpBase + 1 and is_leader_cta) {
        constexpr uint32_t UMMA_M = LAYOUT_AD_M;
        constexpr uint32_t UMMA_N = BLOCK_N;
        constexpr uint32_t UMMA_K = 32;
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = make_sf_desc(nullptr);
        auto a_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            tcgen05_after_thread_sync();

            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (do_tmem_full_arrive)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
            };

            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                with_sf_full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync();

                using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;
                const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                if (sfa_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                }
                const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                if (sfb_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }
                __syncwarp();

                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const uint32_t sfa_id = (kGranK == 32 ? k : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranK == 32 ? k : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);
                        a_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        SM100_MMA_MXF8F6F4_SS::fma(a_desc, b_desc,
                                                   accum_stage_idx * kNumMWaves * BLOCK_N,
                                                   k_block_idx > 0 or k > 0,
                                                   runtime_instr_desc,
                                                   kTmemStartColOfSFA,
                                                   kTmemStartColOfSFB);
                    }
                }
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        if (iter_idx > 0) {
            const auto last_iter = iter_idx - 1;
            const auto accum_phase_idx = (last_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[last_iter % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx == kGemmWarpBase + 2) {
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                full_barriers[stage_idx]->wait(phase);
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    cutlass::arch::fence_view_async_shared();
                }
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= (kNumAGThreads + kNumNonEpilogueThreads) / 32 and
               warp_idx < (kNumAGThreads + kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (kNumAGThreads + kNumNonEpilogueThreads) / 32;
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() { tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages; };

        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            tcgen05_after_thread_sync();

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    const uint32_t m_idx = m_block_idx * BLOCK_M + w * WAVE_BLOCK_M;
                    const uint32_t n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);

                        uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                        epilogue_warp_idx * 32 * kSwizzleCDMode +
                                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                        uint32_t values[kNumElemsPerBankGroup];
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                        } else {
                            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::st_shared(smem_ptr,
                                           cast_into_bf16_and_pack(values[0], values[1]),
                                           cast_into_bf16_and_pack(values[2], values[3]),
                                           cast_into_bf16_and_pack(values[4], values[5]),
                                           cast_into_bf16_and_pack(values[6], values[7]));
                        }
                    }

                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_cd[tma_stage_idx], n_idx, m_idx);
                        cute::tma_store_arrive();
                    }
                }
            }
        }
    }

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
