#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

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
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
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
    using Allocator = cute::TMEM::Allocator1Sm;
    using ab_dtype_t = cutlass::bfloat16_t;

    // ── 常量定义 ──
    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64,
                     "The BF16 GEMM+RS version expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128,
                     "Invalid GEMM thread layout");

    constexpr uint32_t UMMA_M = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N = BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M;
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
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
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(cd_dtype_t), BLOCK_M, BLOCK_N);

    // ── Prefetch TMA descriptors ──
    if (warp_idx == 0 and cute::elect_one_sync()) {
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
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

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
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, global_m, 1);
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 1);
                full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 1 (MMA Issue Warp): Execute UMMA FMA → TMEM accumulator
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == 1 and is_leader_cta) {
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

    // ════════════════════════════════════════════════════════════════
    //  Warp 2~3 (Epilogue Warps): TMEM → smem → TMA store to remote
    // ════════════════════════════════════════════════════════════════
    //
    //  关键改进: 使用 TMA store 异步写远端 partial buffer
    //  - TMA store 是非阻塞的，发射后 epilogue warp 可以立即处理下一个 wave
    //  - 使用 tma_store_arrive + tma_store_wait 做 store pipeline
    //  - 写完整个 block 后，用 st_rel_sys 设置 ready flag（替代 __threadfence_system）
    //
    else if (warp_idx >= kNumNonEpilogueThreads / 32 and
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueThreads / 32;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() { tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages; };
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

            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                    // ── Step 1: TMEM → smem (de-swizzle + BF16 conversion) ──
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    const uint32_t n_idx = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                        epilogue_warp_idx * 32 * kSwizzleCDMode +
                                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                        uint32_t values[kNumElemsPerBankGroup];
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr, values[0], values[1], values[2], values[3]);
                            cutlass::arch::fence_view_async_tmem_load();
                            ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                        } else {
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
                    // Release TMEM stage for next MMA iteration
                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // ── Step 2: smem → remote partial buffer (cp.async.bulk store async) ──
                    //
                    // 使用 cp.async.bulk (tma_store_1d) 将 smem 数据异步写到目标 rank 的 partial buffer
                    // 这是 fire-and-forget 的 async store，不需要 tensor map descriptor
                    // 因为目标地址可能是远端 NVLink 地址，无法用单一 tensor map 覆盖
                    //
                    // 模式: fence → sync → one-thread bulk store → arrive
                    //
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        // 计算目标 partial buffer 的全局地址
                        auto* dst_partial_ptr = is_self_rank ?
                            workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m + w * WAVE_BLOCK_M, n_idx) :
                            sym_buffer.map(
                                workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m + w * WAVE_BLOCK_M, n_idx),
                                dst_rank);

                        // cp.async.bulk: smem → gmem (可以是远端 NVLink 地址)
                        ptx::tma_store_1d(
                            dst_partial_ptr,
                            smem_cd[tma_stage_idx],
                            STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t));
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }
            }

            // ── Step 3: Set ready flag after all waves of this block are stored ──
            //
            // 使用 release semantics store（替代 __threadfence_system + volatile store）
            // 这保证了 TMA store 的数据在 ready flag 被远端看到之前一定对远端可见
            //
            if (epilogue_thread_idx == 0) {
                // Wait all TMA stores for this block to complete
                ptx::tma_store_wait();

                auto* remote_ready_ptr = is_self_rank ?
                    workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx) :
                    sym_buffer.map(
                        workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx),
                        dst_rank);
                // Release store: guarantees all prior stores are visible before this write
                ptx::st_rel_sys(remote_ready_ptr, 1u);
            }
        }
    }

    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

    // ── PDL: 通知后续 reduce kernel 本 GEMM kernel 即将完成 ──
    // 对于 cooperative launch，kernel 退出即为隐式完成信号
    // 对于非 cooperative launch，需要显式触发:
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
          typename cd_dtype_t = cutlass::bfloat16_t>
__global__ void __launch_bounds__(kNumThreads, 1)
sm100_bf16_reduce_epilogue_impl(cd_dtype_t* __restrict__ output,
                                const uint32_t runtime_m_per_rank,
                                const uint32_t shape_n,
                                const void* __restrict__ workspace_base,
                                const uint32_t shape_m_per_rank) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)

    // ── 等待前序 GEMM kernel 完成 ──
    // PDL: cudaGridDependencySynchronize 阻塞直到前序 kernel 信号完成
    // 此后 partial buffer 中所有 rank 的数据都已可见，无需再查 ready flag
    cudaGridDependencySynchronize();

    const auto workspace = layout::GemmRSWorkspace(
        const_cast<void*>(workspace_base), kNumRanks, shape_m_per_rank, shape_n, sizeof(cd_dtype_t), BLOCK_M, BLOCK_N);

    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t global_thread_idx = sm_idx * kNumThreads + thread_idx;
    const uint32_t total_threads = kNumSMs * kNumThreads;

    // ── 向量化 reduce ──
    // 每 4 个 BF16 元素打包为一个 uint64_t 进行向量化访存
    // 从各 rank 的 partial buffer 读取数据，FP32 累加后转回 BF16 写出
    constexpr uint32_t kVecSize = 8; // 8 个 BF16 = 16 bytes = uint4
    const uint32_t total_elements = runtime_m_per_rank * shape_n;
    const uint32_t total_vecs = total_elements / kVecSize;

    // 主循环：每个线程处理若干个向量
    for (uint32_t vec_idx = global_thread_idx; vec_idx < total_vecs; vec_idx += total_threads) {
        const uint32_t elem_base = vec_idx * kVecSize;
        const uint32_t row = elem_base / shape_n;
        const uint32_t col = elem_base - row * shape_n;

        // 累加各 rank 的 partial results
        float acc[kVecSize];
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i)
            acc[i] = 0.0f;

        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col);
            // 向量化读取 8 个 BF16 (16 bytes)
            const auto* vec_ptr = reinterpret_cast<const uint4*>(partial_ptr);
            uint4 data = *vec_ptr;
            // 解包并累加
            const auto* bf16_data = reinterpret_cast<const cd_dtype_t*>(&data);
            #pragma unroll
            for (uint32_t i = 0; i < kVecSize; ++ i) {
                acc[i] += static_cast<float>(bf16_data[i]);
            }
        }

        // 转回 BF16 并写出
        uint4 result;
        auto* result_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
        #pragma unroll
        for (uint32_t i = 0; i < kVecSize; ++ i) {
            result_bf16[i] = cd_dtype_t(acc[i]);
        }
        auto* out_vec_ptr = reinterpret_cast<uint4*>(output + row * shape_n + col);
        *out_vec_ptr = result;
    }

    // ── 处理剩余元素（非对齐的尾部） ──
    const uint32_t remaining_start = total_vecs * kVecSize;
    for (uint32_t elem_idx = remaining_start + global_thread_idx;
         elem_idx < total_elements;
         elem_idx += total_threads) {
        const uint32_t row = elem_idx / shape_n;
        const uint32_t col = elem_idx - row * shape_n;
        if (row >= runtime_m_per_rank) continue;

        float acc = 0.0f;
        #pragma unroll 1
        for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
            const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(src_rank, row, col);
            acc += static_cast<float>(*partial_ptr);
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
