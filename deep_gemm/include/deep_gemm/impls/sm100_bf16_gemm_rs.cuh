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
//  sm100_bf16_gemm_rs_nt_impl —— BF16 GEMM + Reduce-Scatter (Push Only)
// ============================================================================================
//
//  【设计思想】
//
//  两阶段分离架构（方案 B: 统一 barrier 同步）：
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段1: GEMM + NVLink Push (本 kernel)                       │
//  │                                                              │
//  │  Ring 调度: rank i 先计算 rank (i+1) 的 chunk → push        │
//  │            再计算 rank (i+2) → push                         │
//  │            ...                                               │
//  │            最后计算自己 rank i 的 → 直接写本地 partial buf    │
//  │            每波计算掩盖上一波的 NVLink 通信                    │
//  │                                                              │
//  │  Epilogue: TMEM → smem → per-row TMA bulk store (CE)        │
//  │  所有 tile 完成后: tma_store_wait + threadfence_system       │
//  │  + nvlink_barrier 跨 rank 同步（整个 kernel 只做一次）        │
//  └──────────────────────────────────────────────────────────────┘
//             │
//             │ PDL (Programmatic Dependent Launch)
//             ↓
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段2: Reduce Epilogue (独立 kernel)                         │
//  │                                                              │
//  │  cudaGridDependencySynchronize() 等待阶段1完成               │
//  │  直接读取 partial buffer（无需轮询 ready flag）              │
//  │  element-wise 累加 → 写 output                               │
//  └──────────────────────────────────────────────────────────────┘
//
//  【优势】
//
//  相比 per-tile fence + ready flag 方案：
//  1. GEMM kernel 线程全部用于计算和通信，无空转
//  2. __threadfence_system 整个 kernel 只执行一次（而非每 tile 一次）
//  3. TMA 异步写远端，epilogue 不阻塞下一轮 MMA 发射
//  4. Reduce kernel 无需自旋等 ready flag，进入即可直接读取
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
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_nt_impl(const uint32_t shape_m_per_rank,
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
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K for BF16");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    // STORE_BLOCK_N is computed based on comm_dtype_t since we write comm_dtype_t to smem for TMA store
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // smem CD stage sized for comm_dtype_t (what gets TMA-stored to remote partial buffer)
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
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA for full barriers
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs for empty barriers
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── 方案 B: 不再使用 per-tile ready flag ──
    // 所有同步通过 kernel 结束前的 nvlink_barrier 统一完成
    // 此处只需一次跨 rank barrier 保证上一轮（如果有）的 reduce 已完成
    constexpr uint32_t kInitBarrierTag = 41;
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
    //
    //  Wave 0: compute chunk for rank (i+1)%N → push via NVLink
    //  Wave 1: compute chunk for rank (i+2)%N → push via NVLink
    //  ...
    //  Wave N-1: compute chunk for rank i (self) → local write (no communication)
    //
    //  Result: N compute waves overlap N-1 communication phases
    //
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

                // Issue TMAs with correct swizzle modes (A: K-major, B: K-major for NT layout)
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);

                // Arrive at full barriers
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
        // Make instruction descriptor (swap A/B operand order when kSwapAB)
        auto instr_desc = kSwapAB ?
            cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                        UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>() :
            cute::UMMA::make_instr_desc<ab_dtype_t, ab_dtype_t, float,
                                        UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();

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
                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
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
    //  Warp 4~7 (Epilogue Warps): TMEM → smem → per-row TMA bulk store to remote
    // ════════════════════════════════════════════════════════════════
    //
    //  优化设计: 使用 TMA bulk copy (cp.async.bulk) 替代标量 global store
    //  - TMEM → registers → shared memory (linear row-major layout)
    //  - Per-row TMA 1D bulk copy → 远端 partial buffer (异步 DMA，不占 SM store buffer)
    //  - Partial buffer is row-major with stride = shape_n (不一定连续)，
    //    所以每行需要独立的 bulk copy 到正确的全局地址
    //  - 双缓冲 smem pipeline: 前一轮 bulk copies 与当前 TMEM→smem 重叠
    //  - 方案 B: 不再 per-tile fence/flag，所有 tile 完成后 kernel 结束前统一
    //    threadfence_system + nvlink_barrier 跨 rank 同步
    //
    //  性能优势 (vs 原方案每元素标量 store + per-element fence):
    //  1. TMA bulk copy 走 DMA 引擎 (CE)，不 stall SM 的 LSU
    //  2. 每行 128~256 bytes 的 bulk transfer，CE 可 pipeline 128 行的请求
    //  3. __threadfence_system 整个 kernel 只 1 次（而非 per-tile）
    //  4. MMA warp 不再被 epilogue 反压——smem 写入极快，TMEM 立即释放
    //  5. Reduce kernel 无需轮询 ready flag，直接读取
    //
    else if (warp_idx >= kNumNonEpilogueThreads / 32 and
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // Each warp's 32 lanes map to 32 consecutive TMEM rows automatically.

        // Number of comm_dtype_t elements per 128-bit (16-byte) store
        constexpr uint32_t kElemsPerStore = 16 / sizeof(comm_dtype_t);  // BF16: 8, FP32: 4
        // Bytes per row in partial buffer for this tile's N-slice
        constexpr uint32_t kRowBytesPerNSlice = STORE_BLOCK_N * sizeof(comm_dtype_t);
        // Number of 128-bit stores per row to cover STORE_BLOCK_N
        constexpr uint32_t kStoresPerRow = STORE_BLOCK_N / kElemsPerStore;
        // N-slices to cover BLOCK_N
        constexpr uint32_t kNumNSlices = BLOCK_N / STORE_BLOCK_N;

        // smem layout: one row per thread, total STORE_BLOCK_M rows × STORE_BLOCK_N cols
        // smem size per stage = STORE_BLOCK_M * kRowBytesPerNSlice (= SMEM_CD_SIZE_PER_STAGE)
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

                    // Wait previous TMA stores in this pipeline stage to complete
                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // ── Phase 1: TMEM → registers → smem (linear layout) ──
                    // Each thread writes one row of STORE_BLOCK_N elements
                    // epilogue_thread_idx ∈ [0, STORE_BLOCK_M) maps to row offset within wave
                    if (epilogue_thread_idx < STORE_BLOCK_M) {
                        auto* row_ptr = smem_base_ptr + epilogue_thread_idx * kRowBytesPerNSlice;

                        #pragma unroll
                        for (uint32_t st = 0; st < kStoresPerRow; ++ st) {
                            uint32_t tmem_col = accum_stage_idx * UMMA_N +
                                                s * STORE_BLOCK_N + st * kElemsPerStore;

                            if constexpr (cute::is_same_v<comm_dtype_t, float>) {
                                // FP32: load 4 FP32, store as-is (16 bytes)
                                uint32_t f0, f1, f2, f3;
                                cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_col, f0, f1, f2, f3);
                                cutlass::arch::fence_view_async_tmem_load();
                                ptx::st_shared(row_ptr + st * 16, f0, f1, f2, f3);
                            } else {
                                // BF16: load 8 FP32, convert to 8 BF16 (16 bytes)
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

                    // Release TMEM stage as soon as the last smem write is complete
                    if (w == kNumMWaves - 1 and s == kNumNSlices - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // ── Phase 2: Issue per-row TMA 1D bulk copies (smem → remote global) ──
                    // Partial buffer has stride = shape_n (not STORE_BLOCK_N), so rows
                    // are non-contiguous in global memory — must issue one bulk copy per row.
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

                    // Advance TMA store pipeline stage
                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

        }

        // ── 所有 tile 处理完毕，等待最后的 TMA stores 完成 ──
        // 方案 B: 不再 per-tile fence/flag，kernel 结束前统一同步
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<0>();
    }

    // ── 方案 B: kernel 结束前统一 fence + 跨 rank barrier ──
    // 确保所有 TMA stores 对远端可见，然后所有 rank 同步
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
    __threadfence_system();

    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    // ── PDL: 通知后续 reduce kernel 本 GEMM kernel 即将完成 ──
    // 此时所有 rank 的 partial data 已全部就绪，reduce kernel 可直接读取
    cudaTriggerProgrammaticLaunchCompletion();

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

// ============================================================================================
//  sm100_bf16_reduce_epilogue_impl —— Reduce Epilogue (PDL 依赖启动)
// ============================================================================================
//
//  【设计思想】
//
//  本 kernel 作为 GEMM+Push kernel 的下游，通过 PDL 机制实现零间隙调度。
//  方案 B: GEMM kernel 结束前已执行 threadfence_system + nvlink_barrier（跨 rank 同步），
//  因此本 kernel 进入时所有 partial data 已全部就绪，无需轮询任何 ready flag。
//
//  数据流:
//    partial_buffer[rank_0][row][col] + ... + partial_buffer[rank_N-1][...] → output
//
//  优势:
//  - 不需要 per-tile 轮询 ready flag —— 所有数据进入时已保证就绪
//  - 所有线程立即开始有效计算，无资源浪费
//  - 可以利用全部 SM 资源进行 reduce
//  - GPU 硬件可以在 GEMM kernel 快结束时就开始调度本 kernel
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumThreads = 256,
          typename cd_dtype_t = cutlass::bfloat16_t,
          typename comm_dtype_t = cd_dtype_t,
          bool kReduceInFP32 = true>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_bf16_reduce_epilogue_impl(cd_dtype_t* __restrict__ output,
                                const uint32_t runtime_m_per_rank,
                                const uint32_t shape_n,
                                const void* __restrict__ workspace_base,
                                const uint32_t shape_m_per_rank) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)

    // ── 等待前序 GEMM kernel 完成 ──
    // 方案 B: GEMM kernel 结束前已做 threadfence_system + nvlink_barrier，
    //         所有 rank 的 partial data 已全部就绪，无需轮询 ready flag。
    cudaGridDependencySynchronize();

    // workspace elem_size based on comm_dtype_t (what's actually stored in the partial buffer)
    const auto workspace = layout::GemmRSWorkspace(
        const_cast<void*>(workspace_base), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_thread_idx = sm_idx * kNumThreads + thread_idx;
    const uint32_t total_threads = kNumSMs * kNumThreads;

    // ── 向量化 reduce ──
    // Vector size depends on comm_dtype_t:
    //   BF16: 8 elements per uint4 (16 bytes / 2 bytes)
    //   FP32: 4 elements per uint4 (16 bytes / 4 bytes)
    constexpr uint32_t kVecBytes = 16;  // Always load 16 bytes at a time (uint4)
    constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);
    const uint32_t total_elements = runtime_m_per_rank * shape_n;
    const uint32_t total_vecs = total_elements / kVecSize;

    // Accumulation type: FP32 when kReduceInFP32, otherwise same as comm_dtype_t
    using accum_t = cute::conditional_t<kReduceInFP32, float, comm_dtype_t>;

    // 主循环：每个线程处理若干个向量（无需轮询 ready flag）
    for (uint32_t vec_idx = global_thread_idx; vec_idx < total_vecs; vec_idx += total_threads) {
        const uint32_t elem_base = vec_idx * kVecSize;
        const uint32_t row = elem_base / shape_n;
        const uint32_t col = elem_base - row * shape_n;

        // Accumulator
        accum_t acc[kVecSize];
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i)
            acc[i] = accum_t(0);

        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            // 直接读取 —— GEMM kernel 的 nvlink_barrier 已保证所有数据就绪
            const auto* partial_ptr = workspace.get_partial_ptr<comm_dtype_t>(src_rank, row, col);
            const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
            uint4 data = *vec_ptr;
            // 解包并累加
            const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i) {
                if constexpr (kReduceInFP32) {
                    acc[i] += static_cast<float>(comm_data[i]);
                } else {
                    acc[i] += comm_data[i];
                }
            }
        }

        // Convert accumulator to output dtype and write
        uint4 result;
        auto* result_out = reinterpret_cast<cd_dtype_t*>(&result);
        if constexpr (sizeof(cd_dtype_t) == sizeof(comm_dtype_t)) {
            // Same size: pack into uint4 directly
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i) {
                result_out[i] = cd_dtype_t(acc[i]);
            }
            auto* out_vec_ptr = reinterpret_cast<uint4*>(output + row * shape_n + col);
            *out_vec_ptr = result;
        } else {
            // Different sizes: write elements individually
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i) {
                output[(row * shape_n + col) + i] = cd_dtype_t(acc[i]);
            }
        }
    }

    // ── 处理剩余元素（非对齐的尾部） ──
    const uint32_t remaining_start = total_vecs * kVecSize;
    for (uint32_t elem_idx = remaining_start + global_thread_idx;
         elem_idx < total_elements;
         elem_idx += total_threads) {
        const uint32_t row = elem_idx / shape_n;
        const uint32_t col = elem_idx - row * shape_n;
        if (row >= runtime_m_per_rank) continue;

        accum_t acc = accum_t(0);
        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            const auto* partial_ptr = workspace.get_partial_ptr<comm_dtype_t>(src_rank, row, col);
            if constexpr (kReduceInFP32) {
                acc += static_cast<float>(*partial_ptr);
            } else {
                acc += *partial_ptr;
            }
        }
        output[row * shape_n + col] = cd_dtype_t(acc);
    }

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
