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
//  sm100_bf16_gemm_rs_fused_impl —— BF16 GEMM + Reduce-Scatter 全融合
// ============================================================================================
//
//  【设计思想】
//
//  单 kernel 全融合架构：GEMM 计算、NVLink Push、Local Reduce 在同一 kernel 内流水线重叠
//
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │ Warp 0:     TMA Load (A + B tiles → smem)                           │
//  │ Warp 1:     MMA (UMMA FMA → TMEM accumulator)                       │
//  │ Warp 2-3:   (reserved/idle)                                         │
//  │ Warp 4-7:   Epilogue (TMEM → smem → per-row TMA push to remote)    │
//  │             每完成一个 tile 的 TMA push 后，递增远端 ready counter    │
//  │ Warp 8-11:  ★ Reduce Warps (轮询本地 ready counter，在线 reduce)    │
//  │             counter == num_ranks 时读取所有 partial → reduce → output│
//  └──────────────────────────────────────────────────────────────────────┘
//
//  【数据流】
//
//  发送路径 (Epilogue Warps):
//    TMEM → registers → smem → TMA 1D bulk copy → 远端 partial_buf[my_rank]
//    → tma_store_wait (上一个 tile) → red.release.sys (remote counter++)
//
//  接收路径 (Reduce Warps):
//    ld.acquire.sys(local counter) 轮询
//    → counter == num_ranks → 读 partial_buf[0..N-1][tile] → FP32 累加
//    → 写 output[tile]
//
//  【优势】
//
//  1. 完全消除第二个 reduce kernel 的 launch + DRAM 重新读取开销
//  2. Reduce 与 GEMM compute + push 完全 overlap（不等 GEMM 结束）
//  3. 每个 tile 的 reduce 在数据到达后立即开始（最低延迟）
//  4. 大 shape 下：reduce 被 GEMM 完全掩盖，相当于零开销
//  5. 小 shape 下：减少 kernel launch gap + 内存带宽（partial 可能在 L2 中）
//
//  【Signaling 机制】
//
//  - Workspace 已有 ready_flags[num_ranks][m_blocks][n_blocks] (uint32_t)
//  - 但这里我们改变语义：ready_flags 变为 **arrival counter**
//    - 初始值 0
//    - 发送方对远端做 atomicAdd(+1) (release)
//    - 本 rank 自己写 partial 后也 +1
//    - 接收方轮询直到 counter == num_ranks (acquire)
//  - Kernel 开始前需要 reset 所有 counter 为 0
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumReduceThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads + kNumReduceThreads, 1)
sm100_bf16_gemm_rs_fused_impl(cd_dtype_t* __restrict__ output,
                              const uint32_t shape_m_per_rank,
                              const uint32_t runtime_m_per_rank,
                              const uint32_t shape_n,
                              const uint32_t shape_k,
                              const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                              const __grid_constant__ cute::TmaDescriptor tensor_map_b) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
    using ab_dtype_t = cutlass::bfloat16_t;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype for accumulation");

    // ── 常量定义 ──
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads + kNumReduceThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K for BF16");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // smem CD stage for TMA store
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // ── 运行时变量 ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t num_tiles_per_rank = num_m_blocks_per_rank * num_n_blocks;
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    // ── Reduce Warps 身份识别 ──
    constexpr uint32_t kReduceWarpStart = (kNumNonEpilogueThreads + kNumEpilogueThreads) / 32;
    constexpr uint32_t kReduceWarpEnd = kReduceWarpStart + kNumReduceThreads / 32;
    const bool is_reduce_warp = (warp_idx >= kReduceWarpStart and warp_idx < kReduceWarpEnd);

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // ── Prefetch TMA descriptors ──
    if (warp_idx == 0) {
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

    // ── Initialize barriers (non-reduce warps only) ──
    if (not is_reduce_warp) {
        if (warp_idx == 1 and cute::elect_one_sync()) {
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
        } else if (warp_idx == 2) {
            Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
        }
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── 跨 rank barrier 保证上一轮完成 + reset ready counters ──
    // 所有 warp 参与 barrier，但 ready counter reset 只由 reduce warps 做
    constexpr uint32_t kInitBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // ── Reset ready counters to 0 (distributed across all threads) ──
    // ready_flags 语义: arrival counter, 初始 0, 每个 rank push 完后 +1
    // 当 counter == num_ranks 时, reduce warps 可以开始 reduce
    {
        const uint32_t total_counters = num_m_blocks_per_rank * num_n_blocks;
        for (uint32_t i = thread_idx + sm_idx * kNumThreads; i < total_counters;
             i += kNumSMs * kNumThreads) {
            const uint32_t m_blk = i / num_n_blocks;
            const uint32_t n_blk = i - m_blk * num_n_blocks;
            // Reset: 我们用 slot_idx=0 作为 counter 位置
            // 重新定义: ready_ptr(slot=0, m_block, n_block) 作为该 tile 的到达计数
            auto* counter_ptr = workspace.get_ready_ptr(0, m_blk, n_blk);
            *counter_ptr = 0u;
        }
    }
    __syncthreads();  // Ensure counters are reset before any work begins

    // ── Pipeline state ──
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ── Block scheduling (same ring rotation as before) ──
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
    //  Warp 0 (TMA Load Warp): Load A + B tiles into shared memory
    // ════════════════════════════════════════════════════════════════
    if (warp_idx == 0 and cute::elect_one_sync()) {
        uint32_t block_idx = blockIdx.x, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);

            uint32_t load_m_idx = global_m;
            uint32_t load_n_idx = n_idx;
            if constexpr (kNumMulticast > 1) {
                load_m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
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
    //  Warp 1 (MMA Issue Warp): Execute UMMA FMA → TMEM accumulator
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == 1 and is_leader_cta) {
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
    //  Warp 4~7 (Epilogue Warps): TMEM → smem → TMA push + per-tile signal
    // ════════════════════════════════════════════════════════════════
    //
    //  每完成一个 tile 的所有 TMA push 后:
    //  1. tma_store_wait<0>() 确保该 tile 所有 TMA 完成
    //  2. __threadfence_system() 确保远端可见
    //  3. atomicAdd 递增目标 rank 的 ready counter
    //
    else if (warp_idx >= kNumNonEpilogueThreads / 32 and
             warp_idx < (kNumNonEpilogueThreads + kNumEpilogueThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;

        constexpr uint32_t kElemsPerStore = 16 / sizeof(comm_dtype_t);
        constexpr uint32_t kRowBytesPerNSlice = STORE_BLOCK_N * sizeof(comm_dtype_t);
        constexpr uint32_t kStoresPerRow = STORE_BLOCK_N / kElemsPerStore;
        constexpr uint32_t kNumNSlices = BLOCK_N / STORE_BLOCK_N;

        uint32_t tma_stage_idx = 0;

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

            // ── TMEM → smem → per-row TMA bulk store ──
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumNSlices; ++ s) {
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // ── Phase 1: TMEM → registers → smem ──
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

                    // Release TMEM stage as soon as last smem write
                    if (w == kNumMWaves - 1 and s == kNumNSlices - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // ── Phase 2: Per-row TMA 1D bulk copies (smem → remote partial) ──
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        uint32_t base_row = local_m + w * STORE_BLOCK_M;
                        uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        #pragma unroll 1
                        for (uint32_t row = 0; row < STORE_BLOCK_M; ++ row) {
                            comm_dtype_t* dst_ptr = is_self_rank ?
                                workspace.get_partial_ptr<comm_dtype_t>(rank_idx, base_row + row, base_col) :
                                sym_buffer.map(
                                    workspace.get_partial_ptr<comm_dtype_t>(rank_idx, base_row + row, base_col),
                                    dst_rank);

                            auto* src_ptr = smem_base_ptr + row * kRowBytesPerNSlice;
                            ptx::tma_store_1d(dst_ptr, src_ptr, kRowBytesPerNSlice);
                        }
                        cute::tma_store_arrive();
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

            // ── Per-tile signal: ensure all TMA for this tile complete, then signal ──
            // 使用 red.release.sys 保证跨 GPU 可见性（与 reduce warps 的 ld.acquire.sys 配对）
            // 普通 atomicAdd 只有 device scope，对远端 GPU 的 system-scope acquire load 不保证可见
            if (epilogue_warp_idx == 0) {
                cute::tma_store_wait<0>();
                __threadfence_system();
                if (cute::elect_one_sync()) {
                    auto* local_counter = workspace.get_ready_ptr(0, local_m_block_idx, n_block_idx);
                    if (is_self_rank) {
                        // Self-rank: also need sys scope for pairing with ld.acquire.sys
                        asm volatile("red.release.sys.global.add.u32 [%0], %1;"
                                     :: "l"(local_counter), "r"(1u));
                    } else {
                        auto* remote_counter = sym_buffer.map(local_counter, dst_rank);
                        asm volatile("red.release.sys.global.add.u32 [%0], %1;"
                                     :: "l"(remote_counter), "r"(1u));
                    }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 8~11 (Reduce Warps): 在线轮询 counter + reduce + write output
    // ════════════════════════════════════════════════════════════════
    //
    //  设计:
    //  - 所有 SM 的 Reduce Warps 协作处理本 rank 的所有 tile
    //  - 使用 tile_idx 静态分配（每个 SM 负责一批 tile）
    //  - 对每个 tile: 自旋等待 counter == num_ranks，然后读 partial + reduce + write
    //  - 向量化读写（16 bytes / iteration）
    //
    else if (is_reduce_warp) {
        const uint32_t reduce_warp_idx = warp_idx - kReduceWarpStart;
        const uint32_t reduce_thread_idx = reduce_warp_idx * 32 + lane_idx;

        // Total reduce threads across all SMs
        const uint32_t total_reduce_threads = kNumSMs * kNumReduceThreads;
        const uint32_t global_reduce_thread_idx = sm_idx * kNumReduceThreads + reduce_thread_idx;

        // Vectorized reduce: 16 bytes at a time
        constexpr uint32_t kVecBytes = 16;
        constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // BF16: 8, FP32: 4

        // Process tiles: each tile is BLOCK_M × BLOCK_N
        constexpr uint32_t kElemsPerTile = BLOCK_M * BLOCK_N;
        constexpr uint32_t kVecsPerTile = kElemsPerTile / kVecSize;

        // Static tile distribution across SMs
        for (uint32_t tile_idx = sm_idx; tile_idx < num_tiles_per_rank; tile_idx += kNumSMs) {
            const uint32_t tile_m_block = tile_idx / num_n_blocks;
            const uint32_t tile_n_block = tile_idx - tile_m_block * num_n_blocks;

            // ── Spin-wait until all ranks have pushed their data for this tile ──
            // Each warp's lane 0 independently polls the counter, then syncwarp broadcasts
            if (lane_idx == 0) {
                auto* counter_ptr = workspace.get_ready_ptr(0, tile_m_block, tile_n_block);
                uint32_t count;
                do {
                    // Acquire load: ensures subsequent data reads see the pushed data
                    asm volatile("ld.acquire.sys.global.b32 %0, [%1];" : "=r"(count) : "l"(counter_ptr));
                } while (count < kNumRanks);
            }
            __syncwarp();

            // ── Reduce: read all partial buffers and accumulate ──
            const uint32_t base_row = tile_m_block * BLOCK_M;
            const uint32_t base_col = tile_n_block * BLOCK_N;

            // Each reduce thread processes a subset of vectors in this tile
            for (uint32_t vec_local = reduce_thread_idx; vec_local < kVecsPerTile; vec_local += kNumReduceThreads) {
                // Map vector index to (row, col) within tile
                const uint32_t elem_local = vec_local * kVecSize;
                const uint32_t local_row = elem_local / BLOCK_N;
                const uint32_t local_col = elem_local - local_row * BLOCK_N;

                const uint32_t row = base_row + local_row;
                const uint32_t col = base_col + local_col;

                // Skip out-of-bounds
                if (row >= runtime_m_per_rank or col >= shape_n)
                    continue;

                // Accumulate across all ranks
                float acc[kVecSize];
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    acc[i] = 0.0f;

                #pragma unroll 1
                for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
                    const auto* partial_ptr = workspace.get_partial_ptr<comm_dtype_t>(src_rank, row, col);
                    const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
                    uint4 data = *vec_ptr;
                    const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        acc[i] += static_cast<float>(comm_data[i]);
                    }
                }

                // Write output
                auto* out_ptr = output + row * shape_n + col;
                if constexpr (cute::is_same_v<cd_dtype_t, comm_dtype_t> and sizeof(cd_dtype_t) == 2) {
                    // BF16 output: pack and write as uint4
                    uint4 result;
                    auto* result_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        result_bf16[i] = cutlass::bfloat16_t(acc[i]);
                    }
                    *reinterpret_cast<uint4*>(out_ptr) = result;
                } else {
                    // General case: write elements
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        out_ptr[i] = cd_dtype_t(acc[i]);
                    }
                }
            }
        }
    }

    // ── 所有 warp 完成后释放资源 ──
    // Reduce warps 自然完成（处理完所有 tile 后退出循环）
    // Non-reduce warps 在 epilogue 完成后也已就绪
    //
    // 【为什么不需要 final nvlink_barrier】
    // 不能在此使用 __syncthreads() / grid_sync / nvlink_barrier，因为：
    //   1. Reduce warps 依赖其他 rank 的 epilogue push（跨 rank 数据依赖）
    //   2. grid_sync 需要所有 SM 的 thread 0 参与原子操作
    //   3. 如果某个 SM 的 reduce warps 还在 spin-wait → 该 SM 的 __syncthreads() 死锁
    //      → grid_sync 全局阻塞 → 所有 SM epilogue 停止 push → 形成死锁环
    //
    // 连续调用安全性由 **下一个 kernel 的 init nvlink_barrier** 保证：
    //   - 同一 stream 内 kernel 严格顺序执行
    //   - Kernel N+1 的 init barrier 在所有 rank 都发 signal 后才通过
    //   - 这保证了所有 rank 的 Kernel N 已完成（包括 epilogue 的 atomicAdd）
    //   - Counter reset 在 init barrier 之后执行，因此是安全的

    // 只在 block 内同步（不做跨 SM/跨 rank 同步），确保 TMEM dealloc 不会和其他 warp 冲突
    __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
