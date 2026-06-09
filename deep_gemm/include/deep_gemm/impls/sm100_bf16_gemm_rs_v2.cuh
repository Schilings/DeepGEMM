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
//  sm100_bf16_gemm_rs_v2_impl —— BF16 GEMM + Reduce-Scatter V2 (Pull-based Tile Overlap)
// ============================================================================================
//
//  【设计思想 — 借鉴 Flux + MegaMoe】
//
//  单 Kernel 融合：计算与通信在 Tile 粒度上完全 Pipeline 化
//
//  ┌──────────────────────────────────────────────────────────────┐
//  │  GEMM Warps (W0~W3):                                        │
//  │    W0: TMA Load (A+B tiles → smem)                          │
//  │    W1: MMA Issue (UMMA FMA → TMEM accumulator)              │
//  │    W2~3: Epilogue (TMEM → smem → local output buffer)       │
//  │          + per-tile ready flag signaling                     │
//  │                                                              │
//  │  Comm Warps (W4~W7):                                        │
//  │    Pull-based Reduce-Scatter:                                │
//  │    - Poll per-tile ready flags from ALL ranks               │
//  │    - TMA Load remote tile from peer's output buffer → smem  │
//  │    - SMEM → registers → FP32 accumulate → final output      │
//  │    - Fully pipelined with GEMM: comm warp processes tile_i  │
//  │      while GEMM warp computes tile_{i+k}                    │
//  └──────────────────────────────────────────────────────────────┘
//
//  【通信模型: Pull + Reduce (Bandwidth-Optimal)】
//
//  每个 Rank 的最终输出 = sum(partial[r][my_chunk]) for r in 0..N-1
//
//  - 本 rank 计算完整的 C = A × B，输出存入 local output buffer
//  - 设置 per-tile ready flag 通知其他 rank "这个 tile 的数据就绪了"
//  - Comm warps 从每个 peer rank pull 属于自己的 chunk
//  - 拉回后在 FP32 精度下逐 tile 累加到最终输出
//
//  总通信量 = (N-1)/N × total_output_size（与 NCCL ring RS 相同，bandwidth-optimal）
//
//  【Tile 调度: M-dimension Swizzle】
//
//  GEMM 按 M 维度分 N 个 chunk（每个 rank 一个 chunk）。
//  Rank i 优先计算属于 rank (i+1) 的 chunk → 让 rank (i+1) 的 comm warp
//  可以尽早开始拉取。这保证了计算与通信的最大 overlap。
//
//  【优势 vs 旧方案（Push + 两阶段 PDL）】
//
//  1. 单 kernel：无 kernel launch 间隙，无需 PDL
//  2. Tile 粒度 overlap：不等全量 GEMM 完成，逐 tile 通信
//  3. Bandwidth-optimal：通信量 = (N-1)/N × data（vs 旧方案 (N-1) × data）
//  4. Pull 模式 TMA Load 效率高：连续读取整个 tile，无 per-row 碎片
//  5. GEMM 占满 SM 算力：epilogue 轻量（写本地 + set flag），不做远程通信
//  6. Comm warps 独立运行：不干扰 MMA 流水线
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumGemmThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumCommThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumGemmThreads + kNumEpilogueThreads + kNumCommThreads, 1)
sm100_bf16_gemm_rs_v2_impl(const uint32_t shape_m_per_rank,
                           const uint32_t runtime_m_per_rank,
                           const uint32_t shape_n,
                           const uint32_t shape_k,
                           cd_dtype_t* __restrict__ output,
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
    constexpr uint32_t kNumThreads = kNumGemmThreads + kNumEpilogueThreads + kNumCommThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    // Comm warp constants
    constexpr uint32_t kNumCommWarps = kNumCommThreads / 32;
    constexpr uint32_t kCommWarpStart = (kNumGemmThreads + kNumEpilogueThreads) / 32;

    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K for BF16");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // smem CD stage sized for comm_dtype_t (what gets written to local output buffer)
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Comm warp smem: buffer for pulling remote tiles
    // Each comm warp stage holds one BLOCK_M × STORE_BLOCK_N tile in comm_dtype_t
    constexpr uint32_t kNumCommStages = 2;
    constexpr uint32_t SMEM_COMM_SIZE_PER_STAGE = BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_COMM_SIZE = SMEM_COMM_SIZE_PER_STAGE * kNumCommStages;

    // ── 运行时变量 ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t num_n_slices = BLOCK_N / STORE_BLOCK_N;
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
    //
    // Layout:
    //   [0, SMEM_CD_SIZE)                                    : CD epilogue stages (for local write)
    //   [SMEM_CD_SIZE, +kNumStages*A)                        : A tiles
    //   [+kNumStages*A, +kNumStages*(A+B))                   : B tiles
    //   [+kNumStages*(A+B), +barriers)                       : Pipeline barriers
    //   [+barriers, +SMEM_COMM_SIZE)                         : Comm warp pull buffer
    //
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

    // Comm warp pull buffer (after barriers + tmem_ptr)
    auto smem_comm_base = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
                          + (kNumStages * 2 + kNumEpilogueStages * 2) * sizeof(Barrier) + sizeof(uint32_t);
    // Align to 128 bytes for TMA
    smem_comm_base = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(smem_comm_base) + 127) & ~127ull);
    auto smem_comm = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<comm_dtype_t*>(smem_comm_base + i * SMEM_COMM_SIZE_PER_STAGE);
    });

    // ── Initialize barriers ──
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
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── Initial NVLink barrier: ensure previous iteration's data is consumed ──
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

    // ── Block scheduling: M-Swizzle for maximum overlap ──
    //
    //  Rank i computes tiles in order:
    //    chunk for rank (i+1)%N first → so rank (i+1) can pull earliest
    //    chunk for rank (i+2)%N next  → ...
    //    chunk for rank i (self) last  → self doesn't need remote pull
    //
    //  Within each chunk, blocks are distributed round-robin across SMs.
    //
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        const uint32_t m_rank_wave = block_idx / (num_m_blocks_per_rank * num_n_blocks);
        const uint32_t rem = block_idx - m_rank_wave * num_m_blocks_per_rank * num_n_blocks;
        const uint32_t local_m_block_idx = rem / num_n_blocks;
        n_block_idx = rem - local_m_block_idx * num_n_blocks;
        // Swizzle: compute other ranks' chunks first
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

        // Final wait for safe barrier destruction
        if constexpr (kNumMulticast > 1) {
            const auto iter_val = iter_idx - 1;
            if (iter_val >= 0) {
                const auto accum_phase_idx = (iter_val / kNumEpilogueStages) & 1;
                tmem_empty_barriers[iter_val % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 2~3 (Epilogue Warps): TMEM → smem → local partial buffer + set ready flag
    // ════════════════════════════════════════════════════════════════
    //
    //  Key difference from V1: We write to LOCAL partial buffer only (slot = rank_idx),
    //  then set a per-tile ready flag so comm warps on other ranks can pull.
    //  No cross-rank NVLink writes here — all remote communication is done by comm warps.
    //
    else if (warp_idx >= kNumGemmThreads / 32 and
             warp_idx < (kNumGemmThreads + kNumEpilogueThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumGemmThreads / 32;
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

            // Compute local coordinates
            const uint32_t dst_rank = m_block_idx / num_m_blocks_per_rank;
            const uint32_t local_m_block_idx = m_block_idx - dst_rank * num_m_blocks_per_rank;
            const uint32_t local_m = local_m_block_idx * BLOCK_M;

            // ── TMEM → smem → local partial buffer (via TMA bulk store) ──
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumNSlices; ++ s) {
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

                    // Wait previous TMA stores
                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Phase 1: TMEM → registers → smem (linear layout)
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

                    // Phase 2: Issue per-row TMA 1D bulk copies to LOCAL partial buffer
                    // We always write to our own rank's slot in the symmetric buffer.
                    // Other ranks will pull from us via comm warps.
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        uint32_t base_row = local_m + w * STORE_BLOCK_M;
                        uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                        #pragma unroll 1
                        for (uint32_t row = 0; row < STORE_BLOCK_M; ++ row) {
                            // Always write to our own rank's partial buffer slot
                            // Use rank_idx as slot for our contribution to dst_rank's output
                            comm_dtype_t* dst_ptr = workspace.get_partial_ptr<comm_dtype_t>(
                                rank_idx, base_row + row, base_col);

                            auto* src_ptr = smem_base_ptr + row * kRowBytesPerNSlice;
                            ptx::tma_store_1d(dst_ptr, src_ptr, kRowBytesPerNSlice);
                        }
                        cute::tma_store_arrive();
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

            // ── After all N-slices of this tile are stored, set per-tile ready flag ──
            // Wait for all TMA stores of this tile to land, then signal
            if (epilogue_warp_idx == 0) {
                cute::tma_store_wait<0>();
                if (cute::elect_one_sync()) {
                    // Set ready flag: other ranks can now pull this tile from us
                    // ready_ptr layout: [dst_rank][m_block_within_rank][n_block]
                    // We write to slot [rank_idx] at position [local_m_block_idx][n_block_idx]
                    // which tells dst_rank: "rank_idx's partial for your (local_m_block_idx, n_block_idx) is ready"
                    auto* ready_ptr = workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx);
                    __threadfence_system();  // Ensure TMA writes are visible across NVLink
                    ptx::st_rel_sys(ready_ptr, 1u);
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 4~7 (Comm Warps): Pull-based Reduce-Scatter
    // ════════════════════════════════════════════════════════════════
    //
    //  Each rank is responsible for producing the FINAL output of its own chunk.
    //  For each tile (m_block, n_block) in my chunk:
    //    output[m, n] = sum over all ranks r: partial[r][m_block][n_block]
    //
    //  My own rank's partial is already in local buffer (written by epilogue warps).
    //  I pull (N-1) partials from peer ranks and accumulate in FP32.
    //
    //  Scheduling:
    //    Comm warps iterate over my chunk's tiles (num_m_blocks_per_rank × num_n_blocks).
    //    For each tile, they wait for ALL ranks' ready flags, then pull + reduce.
    //    Tiles are distributed round-robin across SMs.
    //
    else if (warp_idx >= kCommWarpStart) {
        const uint32_t comm_warp_local_idx = warp_idx - kCommWarpStart;
        const uint32_t comm_thread_local_idx = comm_warp_local_idx * 32 + lane_idx;
        const uint32_t total_comm_threads = kNumCommThreads;

        // My chunk: tiles belonging to rank_idx
        const uint32_t total_my_tiles = num_m_blocks_per_rank * num_n_blocks;

        // Vectorized reduce: 16 bytes at a time
        constexpr uint32_t kVecBytes = 16;
        constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);

        // Elements per tile
        const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;
        const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

        // Iterate over tiles assigned to this SM
        for (uint32_t tile_idx = sm_idx; tile_idx < total_my_tiles; tile_idx += kNumSMs) {
            const uint32_t my_m_block = tile_idx / num_n_blocks;
            const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;

            // ── Wait for ALL ranks' ready flags for this tile ──
            // We need partial data from all N ranks (including self)
            // self is guaranteed ready once epilogue finishes for this tile
            if (comm_warp_local_idx == 0 and lane_idx < kNumRanks) {
                // Each lane polls one rank's ready flag
                const uint32_t poll_rank = lane_idx;
                if (poll_rank != rank_idx) {
                    // Poll remote rank's ready flag for my tile
                    // Remote rank stores ready at its own workspace.get_ready_ptr(poll_rank, my_m_block, my_n_block)
                    // But we need to read from the REMOTE rank's buffer via sym_buffer mapping
                    auto* local_ready_ptr = workspace.get_ready_ptr(poll_rank, my_m_block, my_n_block);
                    auto* remote_ready_ptr = sym_buffer.map(local_ready_ptr, poll_rank);

                    // Spin-wait with acquire semantics (30s timeout)
                    constexpr int64_t kTimeoutCycles = 30ll * 2000000000ll;
                    const auto start_clock = clock64();
                    while (ptx::ld_acq_sys(remote_ready_ptr) == 0u) {
                        if (clock64() - start_clock >= kTimeoutCycles) {
                            printf("GEMM-RS V2 comm warp timeout: rank=%d, poll_rank=%d, tile=(%d,%d)\n",
                                   rank_idx, poll_rank, my_m_block, my_n_block);
                            DG_DEVICE_ASSERT(false and "Comm warp ready flag timeout");
                        }
                    }
                }
            }
            // Sync all comm warps after polling
            cutlass::arch::NamedBarrier::sync(kNumCommThreads, 2);

            // ── Pull + Reduce: accumulate all ranks' partials ──
            // Each comm thread handles a slice of the tile's elements
            const uint32_t base_row = my_m_block * BLOCK_M;
            const uint32_t base_col = my_n_block * BLOCK_N;

            for (uint32_t vec_offset = comm_thread_local_idx; vec_offset < vecs_per_tile; vec_offset += total_comm_threads) {
                // Convert vec_offset to row/col within tile
                const uint32_t elem_offset = vec_offset * kVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;

                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;

                if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                    continue;

                // FP32 accumulator
                float acc[kVecSize];
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    acc[i] = 0.0f;

                // Accumulate from all ranks
                #pragma unroll 1
                for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
                    // Read from src_rank's partial buffer
                    const comm_dtype_t* partial_ptr;
                    if (src_rank == rank_idx) {
                        // Local: read directly
                        partial_ptr = workspace.get_partial_ptr<comm_dtype_t>(src_rank, global_row, global_col);
                    } else {
                        // Remote: map through sym_buffer for NVLink P2P read
                        auto* local_ptr = workspace.get_partial_ptr<comm_dtype_t>(src_rank, global_row, global_col);
                        partial_ptr = sym_buffer.map(local_ptr, src_rank);
                    }

                    // Vectorized load
                    const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
                    uint4 data = *vec_ptr;
                    const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);

                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        acc[i] += static_cast<float>(comm_data[i]);
                    }
                }

                // Write final output
                // Output is at: output[global_row * shape_n + global_col]
                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    // Pack FP32 → BF16 and write as uint4
                    uint4 result;
                    auto* out_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        out_bf16[i] = cd_dtype_t(acc[i]);
                    }
                    auto* out_ptr = reinterpret_cast<uint4*>(output + global_row * shape_n + global_col);
                    *out_ptr = result;
                } else {
                    // FP32 output
                    uint4 result;
                    auto* out_f32 = reinterpret_cast<float*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i) {
                        out_f32[i] = acc[i];
                    }
                    auto* out_ptr = reinterpret_cast<uint4*>(output + global_row * shape_n + global_col);
                    *out_ptr = result;
                }
            }
        }
    }

    // ── Final synchronization ──
    // All warps must finish before kernel exits.
    // Comm warps have written final output; epilogue warps have written partials + flags.
    // We need a grid-wide + NVLink barrier to ensure:
    //   1. All ranks have finished pulling (so nobody reads stale ready flags next iteration)
    //   2. Ready flags can be safely reset for next call
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    // Reset ready flags for next iteration (only for my rank's slot)
    {
        const uint32_t total_flags = num_m_blocks_per_rank * num_n_blocks;
        for (uint32_t flag_idx = thread_idx; flag_idx < total_flags; flag_idx += kNumThreads) {
            const uint32_t mb = flag_idx / num_n_blocks;
            const uint32_t nb = flag_idx - mb * num_n_blocks;
            auto* ready_ptr = workspace.get_ready_ptr(rank_idx, mb, nb);
            *ready_ptr = 0u;
        }
    }

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
