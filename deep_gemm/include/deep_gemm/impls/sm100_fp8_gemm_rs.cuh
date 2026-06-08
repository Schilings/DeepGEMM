#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

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
//  sm100_fp8_gemm_rs_nt_impl —— FP8 GEMM + Reduce-Scatter (Push Only)
// ============================================================================================
//
//  【设计思想】
//
//  两阶段分离架构（方案 B: 统一 barrier 同步）：
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段1: FP8 GEMM + NVLink Push (本 kernel)                   │
//  │                                                              │
//  │  Ring 调度: rank i 先计算 rank (i+1) 的 chunk → push        │
//  │            再计算 rank (i+2) → push                         │
//  │            ...                                               │
//  │            最后计算自己 rank i 的 → 直接写本地 partial buf    │
//  │            每波计算掩盖上一波的 NVLink 通信                    │
//  │                                                              │
//  │  Epilogue: TMEM → registers → global store 到远端 partial   │
//  │  所有 tile 完成后: __threadfence_system + nvlink_barrier     │
//  │  跨 rank 同步（整个 kernel 只做一次）                         │
//  └──────────────────────────────────────────────────────────────┘
//             │
//             │ PDL (Programmatic Dependent Launch)
//             ↓
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段2: Reduce Epilogue (独立 kernel, 复用 bf16 版本)         │
//  │                                                              │
//  │  cudaGridDependencySynchronize() 等待阶段1完成               │
//  │  直接读取 partial buffer（无需轮询 ready flag）              │
//  │  element-wise 累加 → 写 output                               │
//  └──────────────────────────────────────────────────────────────┘
//
//  【FP8 特有逻辑】
//
//  - TMA 加载 Scale Factor A/B (SFA/SFB)
//  - Warp 2: SF warp transpose (UTCCP 所需布局变换)
//  - Warp 1: UTCCP 拷贝 SF 到 TMEM + SM100_MMA_MXF8F6F4_SS::fma
//  - 累加器始终为 FP32，通信精度由 comm_dtype_t 模板参数控制
//
//  【优势】（相比 per-tile fence + ready flag 方案）
//
//  1. __threadfence_system 整个 kernel 只执行一次（而非每 tile 一次）
//  2. Reduce kernel 无需自旋等 ready flag，进入即可直接读取
//  3. 简化的 epilogue: TMEM → registers → global store，无 flag 设置开销
//  4. 支持 comm_dtype_t 选择通信精度（BF16 省带宽 / FP32 保精度）
//  5. 两个 kernel 间通过 PDL 重叠，reduce 可在 GEMM 即将结束时开始
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
          uint32_t kGranK,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_rs_nt_impl(const uint32_t shape_m_per_rank,
                           const uint32_t runtime_m_per_rank,
                           const uint32_t shape_n,
                           const uint32_t shape_k,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_sfb) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::float_e4m3_t;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype for accumulation");

    // ── 常量定义 ──
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;

    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K for FP8");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kGranK == 32 or kGranK == 128, "Invalid granularity K");

    constexpr uint32_t kNumSFAStagesPerLoad = kGranK == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranK == 32 ? 1 : 4;

    // TMEM layout sizes for scale factors
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;

    // Epilogue stages (depends on TMEM capacity)
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    // TMEM columns
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // Store block shape for epilogue
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Shared memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);

    // ── 运行时变量 ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // ── Prefetch TMA descriptors ──
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
    }

    // ── Shared memory layout ──
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    // Note: smem_cd is used only for temporary staging if needed; for the direct-store epilogue
    // we don't actually use it, but keep allocation for barrier alignment
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    auto barrier_start_ptr = reinterpret_cast<Barrier*>(sf_start_ptr +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    auto full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto with_sf_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_full_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    // ── Initialize barriers ──
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(kNumMulticast);
            empty_barriers[i]->init(1);
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // 方案 B: 不需要 ready flag 清零和初始 barrier
    // 所有同步通过 kernel 结束前的 nvlink_barrier 统一完成
    constexpr uint32_t kInitBarrierTag = 41;
    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // ── Pipeline state ──
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ── Block scheduling: rotate through ranks for load-balanced communication ──
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        const uint32_t m_rank_wave = block_idx / (num_m_blocks_per_rank * num_n_blocks);
        const uint32_t rem = block_idx - m_rank_wave * num_m_blocks_per_rank * num_n_blocks;
        const uint32_t local_m_block_idx = rem / num_n_blocks;
        n_block_idx = rem - local_m_block_idx * num_n_blocks;
        const uint32_t dst_rank = (m_rank_wave + 1 < kNumRanks) ?
            (rank_idx + m_rank_wave + 1) % kNumRanks : rank_idx;
        m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;
        block_idx += kNumSMs;
        ++ iter_idx;
        return true;
    };

    // ════════════════════════════════════════════════════════════════
    //  Warp 0 (TMA Load Warp): Load A + B + SFA + SFB tiles into shared memory
    // ════════════════════════════════════════════════════════════════
    if (warp_idx == 0 and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);

            // Add multicast CTA offsets
            uint32_t load_m_idx = global_m;
            uint32_t load_n_idx = n_idx;
            if constexpr (kNumMulticast > 1) {
                load_m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                load_n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;

                // Issue A/B TMAs
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);

                uint32_t num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                // Issue SFA TMA at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    tma::copy<BLOCK_M, 1, 0>(
                        &tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx],
                        global_m, ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                // Issue SFB TMA at certain stages
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    tma::copy<BLOCK_N, 1, 0>(
                        &tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx],
                        n_idx, ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad));
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                // Arrive at full barriers
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 1 (MMA Issue Warp): Execute FP8 MMA with block scaling → TMEM
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == 1 and is_leader_cta) {
        // Make instruction descriptor for block-scaled FP8 MMA
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // UMMA arrive helper (supports multicast)
        auto umma_arrive = [](const uint64_t* barrier) {
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };

        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
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
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                // UTCCP copy scale factors to TMEM
                using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                    cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                if (sfa_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                }
                const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                if (sfb_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }
                __syncwarp();

                // Issue UMMA FP8 MMA with block scaling
                using mma_t = cute::conditional_t<kNumMulticast == 1, SM100_MMA_MXF8F6F4_SS, SM100_MMA_MXF8F6F4_2x1SM_SS>;
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const uint32_t sfa_id = (kGranK == 32 ? k : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranK == 32 ? k : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        b_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                            b_desc_base_lo, 0, k * UMMA_K);

                        #pragma unroll
                        for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                a_desc_base_lo, w * WAVE_BLOCK_M * BLOCK_K, k * UMMA_K);

                            if constexpr (kSwapAB) {
                                mma_t::fma(b_desc, a_desc,
                                           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                           k_block_idx > 0 or k > 0,
                                           runtime_instr_desc,
                                           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                                           kTmemStartColOfSFB);
                            } else {
                                mma_t::fma(a_desc, b_desc,
                                           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                           k_block_idx > 0 or k > 0,
                                           runtime_instr_desc,
                                           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                                           kTmemStartColOfSFB);
                            }
                        }
                    }
                }
                __syncwarp();
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        if constexpr (kNumMulticast > 1) {
            const auto iter_val = iter_idx - 1;
            if (iter_val >= 0) {
                const auto accum_phase_idx = (iter_val / kNumEpilogueStages) & 1;
                tmem_empty_barriers[iter_val % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 2 (SF Transpose Warp): Transpose SFA/SFB in smem for UTCCP
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == 2) {
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
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // Transpose for UTCCP at certain stages
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

                // Arrive
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 4~7 (Epilogue Warps): TMEM → registers → global store to remote
    // ════════════════════════════════════════════════════════════════
    //
    //  直接从 TMEM 读取 FP32 累加值，转换为 comm_dtype_t，全局 store 到远端 partial buffer
    //  - 每个线程处理自己 lane 对应的行
    //  - 128 个线程覆盖 128 行 (= STORE_BLOCK_M = BLOCK_M)
    //  - 逐列组迭代完成整个 BLOCK_N
    //  - 写完后设置 ready flag
    //
    else if (warp_idx >= kNumNonEpilogueThreads / 32 and
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        // Each thread handles one row (128 threads = 128 rows = STORE_BLOCK_M)
        const uint32_t my_row = epilogue_thread_idx;

        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            auto accum_stage_idx = (iter_idx - 1) % kNumEpilogueStages;
            auto accum_phase_idx = ((iter_idx - 1) / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const uint32_t dst_rank = m_block_idx / num_m_blocks_per_rank;
            const uint32_t local_m_block_idx = m_block_idx - dst_rank * num_m_blocks_per_rank;
            const uint32_t local_m = local_m_block_idx * BLOCK_M;
            const bool is_self_rank = (dst_rank == rank_idx);

            // ── TMEM → registers → global memory store ──
            // SM100_TMEM_LOAD_32dp32b4x: loads 4 × 32-bit (= 4 floats) per thread from TMEM
            constexpr uint32_t kElemsPerLoad = 4;
            constexpr uint32_t kNumIters = UMMA_N / kElemsPerLoad;

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                if (my_row < WAVE_BLOCK_M) {
                    #pragma unroll
                    for (uint32_t iter = 0; iter < kNumIters; ++ iter) {
                        uint32_t tmem_col = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + iter * kElemsPerLoad;
                        uint32_t f0, f1, f2, f3;
                        cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_col, f0, f1, f2, f3);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Compute destination address in partial buffer
                        uint32_t global_row = local_m + w * WAVE_BLOCK_M + my_row;
                        uint32_t global_col = n_block_idx * BLOCK_N + iter * kElemsPerLoad;

                        comm_dtype_t* dst_ptr = is_self_rank ?
                            workspace.get_partial_ptr<comm_dtype_t>(rank_idx, global_row, global_col) :
                            sym_buffer.map(
                                workspace.get_partial_ptr<comm_dtype_t>(rank_idx, global_row, global_col),
                                dst_rank);

                        // Store elements in communication format
                        if constexpr (cute::is_same_v<comm_dtype_t, float>) {
                            // FP32 communication: store raw FP32 values
                            *reinterpret_cast<uint32_t*>(dst_ptr + 0) = f0;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 1) = f1;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 2) = f2;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 3) = f3;
                        } else {
                            // BF16 communication: convert FP32 → BF16 and pack pairs
                            uint32_t bf16_pair0 = cast_into_bf16_and_pack(f0, f1);
                            uint32_t bf16_pair1 = cast_into_bf16_and_pack(f2, f3);
                            *reinterpret_cast<uint32_t*>(dst_ptr + 0) = bf16_pair0;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 2) = bf16_pair1;
                        }
                    }
                }
            }

            // Release TMEM stage for next MMA iteration
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
        }
    }

    // ── 方案 B: 所有 tile 完成后，统一做一次全局 fence + 跨 rank barrier ──
    // 确保所有 global store（包括 NVLink push）对远端可见
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
    __threadfence_system();

    // 跨 rank nvlink_barrier: 等所有 rank 的 GEMM kernel 都完成写入
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    // ── PDL: 通知后续 reduce kernel 本 GEMM kernel 即将完成 ──
    cudaTriggerProgrammaticLaunchCompletion();

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
