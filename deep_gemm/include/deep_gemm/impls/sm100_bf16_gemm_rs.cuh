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
//  两阶段分离架构：
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段1: GEMM + NVLink Push (本 kernel)                       │
//  │                                                              │
//  │  调度顺序: rank i 先计算发往 rank i+1 的 chunk → push       │
//  │           再计算 rank i+2 → push                            │
//  │           ...                                                │
//  │           最后计算自己 rank i 的 → 直接写本地 partial buf     │
//  │                                                              │
//  │  N 次计算掩盖 N-1 次通信                                     │
//  │  Epilogue: TMEM → smem → TMA store 到远端 partial buffer    │
//  │  完成后设置 ready flag (st_rel_sys)                          │
//  └──────────────────────────────────────────────────────────────┘
//             │
//             │ PDL (Programmatic Dependent Launch)
//             ↓
//  ┌──────────────────────────────────────────────────────────────┐
//  │ 阶段2: Reduce Epilogue (独立 kernel)                         │
//  │                                                              │
//  │  cudaGridDependencySynchronize() 等待阶段1完成               │
//  │  从 partial buffer 读取各 rank 的数据                        │
//  │  element-wise 累加 → 写 output                               │
//  └──────────────────────────────────────────────────────────────┘
//
//  【优势】
//
//  相比 RS Warp 内嵌方案：
//  1. GEMM kernel 线程全部用于计算和通信，无空转
//  2. Reduce kernel 不需要自旋等待，进入时数据已就绪
//  3. TMA 异步写远端，epilogue 不阻塞下一轮 MMA 发射
//  4. 两个 kernel 间通过 PDL 重叠，reduce 可在 GEMM 即将结束时开始
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
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
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

    // ── Clean ready flags (cross-rank sync) ──
    for (uint32_t i = sm_idx * kNumThreads + thread_idx;
         i < kNumRanks * workspace.get_num_m_blocks_per_rank() * workspace.get_num_n_blocks();
         i += kNumSMs * kNumThreads) {
        auto* ready_base = workspace.get_ready_ptr();
        ready_base[i] = 0;
    }
    constexpr uint32_t kAfterReadyCleanBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kAfterReadyCleanBarrierTag>(
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
    //  Warp 4~7 (Epilogue Warps): TMEM → registers → global store to remote
    // ════════════════════════════════════════════════════════════════
    //
    //  简化设计: 直接从 TMEM 读取 FP32 累加值，转 BF16，全局 store 到远端 partial buffer
    //  - 每个线程处理自己 lane 对应的行
    //  - 128 个线程覆盖 128 行 (= STORE_BLOCK_M = BLOCK_M for our tile)
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
            // TMEM layout: 128 rows (one per thread in 32dp addressing) × UMMA_N columns
            // SM100_TMEM_LOAD_32dp32b4x: loads 4 × 32-bit (= 4 floats) per thread from TMEM
            // Each float is one accumulator element at (thread's row, tmem_addr column)
            //
            // We iterate over N in chunks of 4 floats, convert to comm_dtype_t, and store.
            // comm_dtype_t controls the communication precision:
            //   - bfloat16_t: saves NVLink bandwidth (2 bytes/elem), slight precision loss in reduce
            //   - float:      full precision communication (4 bytes/elem), no reduce precision loss
            constexpr uint32_t kElemsPerLoad = 4;  // SM100_TMEM_LOAD_32dp32b4x gives 4 floats
            constexpr uint32_t kNumIters = UMMA_N / kElemsPerLoad;  // BLOCK_N / 4

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                if (my_row < WAVE_BLOCK_M) {
                    #pragma unroll
                    for (uint32_t iter = 0; iter < kNumIters; ++ iter) {
                        uint32_t tmem_col = accum_stage_idx * UMMA_N + iter * kElemsPerLoad;
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
                            // FP32 communication: store raw FP32 values (16 bytes per 4 elements)
                            *reinterpret_cast<uint32_t*>(dst_ptr + 0) = f0;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 1) = f1;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 2) = f2;
                            *reinterpret_cast<uint32_t*>(dst_ptr + 3) = f3;
                        } else {
                            // BF16 communication: convert FP32 → BF16 and pack pairs (8 bytes per 4 elements)
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

            // ── Ensure all stores are visible, then set ready flag ──
            __threadfence_system();

            if (epilogue_thread_idx == 0) {
                uint32_t* remote_ready_ptr = is_self_rank ?
                    workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx) :
                    sym_buffer.map(
                        workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx),
                        dst_rank);
                ptx::st_rel_sys(remote_ready_ptr, 1u);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

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

// ============================================================================================
//  sm100_bf16_reduce_epilogue_impl —— Reduce Epilogue (PDL 依赖启动)
// ============================================================================================
//
//  【设计思想】
//
//  本 kernel 作为 GEMM+Push kernel 的下游，通过 PDL 机制实现零间隙调度。
//  进入时调用 cudaGridDependencySynchronize()，确保 partial buffer 数据已全部就绪。
//  然后执行向量化的 element-wise reduce（各 rank 的 partial results 累加）。
//
//  数据流:
//    partial_buffer[rank_0][m_block][n_block] + ... + partial_buffer[rank_N-1][...] → output
//
//  优势:
//  - 不需要自旋等待 ready flag（PDL 保证进入时数据已就绪）
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

    // ── 等待前序 GEMM kernel 完成（PDL 保证本 rank 的 GEMM grid 已结束）──
    // NOTE: PDL 只保证本 rank 的 GEMM kernel 完成，不保证远程 rank 的 NVLink 写入已完成。
    //       因此必须在下游轮询 ready flag，确保数据已实际写入。
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

    // Pre-compute tile indices for ready flag polling
    const uint32_t num_m_blocks = workspace.get_num_m_blocks_per_rank();
    const uint32_t num_n_blocks = workspace.get_num_n_blocks();

    // 主循环：每个线程处理若干个向量
    for (uint32_t vec_idx = global_thread_idx; vec_idx < total_vecs; vec_idx += total_threads) {
        const uint32_t elem_base = vec_idx * kVecSize;
        const uint32_t row = elem_base / shape_n;
        const uint32_t col = elem_base - row * shape_n;

        // Compute tile indices for ready flag
        const uint32_t m_block = row / BLOCK_M;
        const uint32_t n_block = col / BLOCK_N;

        // Accumulator
        accum_t acc[kVecSize];
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i)
            acc[i] = accum_t(0);

        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            // Poll ready flag: wait until the partial data for this tile is written
            auto* ready_ptr = workspace.get_ready_ptr(src_rank, m_block, n_block);
            while (ptx::ld_acq_sys(ready_ptr) == 0);

            // 向量化读取 (16 bytes)
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
        // Output vector size may differ from comm vector size (e.g., comm=FP32, output=BF16)
        // We need to handle size mismatch: kVecSize elements of comm_dtype → kVecSize elements of cd_dtype
        // Since both work on kVecSize elements but output uint4 may not hold all of them,
        // we write element by element when sizes differ
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

        const uint32_t m_block = row / BLOCK_M;
        const uint32_t n_block = col / BLOCK_N;

        accum_t acc = accum_t(0);
        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            auto* ready_ptr = workspace.get_ready_ptr(src_rank, m_block, n_block);
            while (ptx::ld_acq_sys(ready_ptr) == 0);

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
