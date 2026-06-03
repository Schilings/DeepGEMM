#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/gemm_rs.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kNumRSThreads,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t>
__global__ void __launch_bounds__(kNumRSThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_nt_impl(void* y,
                           const uint32_t shape_m_per_rank,
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

    constexpr uint32_t kSwizzleAMode = 128;
    constexpr uint32_t kSwizzleBMode = 128;
    constexpr uint32_t kSwizzleCDMode = 128;
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumRSWarps = kNumRSThreads / 32;
    constexpr uint32_t kGemmWarpBase = kNumRSWarps;
    constexpr uint32_t kNumThreads = kNumRSThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    DG_STATIC_ASSERT(BLOCK_M == 128 and BLOCK_N == 128 and BLOCK_K == 64, "The first BF16 GEMM+RS version expects 128x128x64 tiles");
    DG_STATIC_ASSERT(kNumRSThreads % 32 == 0 and kNumRSThreads >= 128, "Invalid RS threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128 and kNumEpilogueThreads == 128, "Invalid GEMM thread layout");

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
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

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

    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
    }

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

    for (uint32_t i = sm_idx * kNumThreads + thread_idx;
         i < kNumRanks * workspace.get_num_m_blocks_per_rank() * workspace.get_num_n_blocks();
         i += kNumSMs * kNumThreads) {
        auto* ready_base = workspace.get_ready_ptr();
        ready_base[i] = 0;
    }
    constexpr uint32_t kAfterReadyCleanBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kAfterReadyCleanBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
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

    if (warp_idx == kGemmWarpBase and cute::elect_one_sync()) {
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
    } else if (warp_idx == kGemmWarpBase + 1 and is_leader_cta) {
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
    } else if (warp_idx >= (kNumRSThreads + kNumNonEpilogueThreads) / 32 and
               warp_idx < (kNumRSThreads + kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        const auto epilogue_warp_idx = warp_idx - (kNumRSThreads + kNumNonEpilogueThreads) / 32;
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
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
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
                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    constexpr uint32_t kNumBankGroupsPerRow = STORE_BLOCK_N / kNumElemsPerBankGroup;
                    auto* remote_partial_ptr = sym_buffer.map(
                        workspace.get_partial_ptr<cd_dtype_t>(rank_idx, local_m + w * WAVE_BLOCK_M, n_idx), dst_rank);
                    #pragma unroll 1
                    for (uint32_t vec_idx = epilogue_thread_idx; vec_idx < WAVE_BLOCK_M * kNumBankGroupsPerRow; vec_idx += kNumUMMAStoreThreads) {
                        const uint32_t row = vec_idx / kNumBankGroupsPerRow;
                        const uint32_t bank_group_idx = vec_idx - row * kNumBankGroupsPerRow;
                        const uint32_t col = bank_group_idx * kNumElemsPerBankGroup;
                        if (local_m + w * WAVE_BLOCK_M + row < runtime_m_per_rank and n_idx + col < shape_n) {
                            const uint32_t warp_row = row / 32;
                            const uint32_t row_in_warp = row - warp_row * 32;
                            const uint32_t swizzled_bank_group_idx = bank_group_idx ^ (row_in_warp % (kSwizzleCDMode / kNumBankGroupBytes));
                            const auto* smem_vec_ptr = reinterpret_cast<const uint4*>(
                                reinterpret_cast<const uint8_t*>(smem_cd[tma_stage_idx]) +
                                warp_row * 32 * kSwizzleCDMode +
                                row_in_warp * (kNumBankGroupBytes * 8) +
                                swizzled_bank_group_idx * kNumBankGroupBytes);
                            auto* remote_vec_ptr = reinterpret_cast<uint4*>(remote_partial_ptr + row * shape_n + col);
                            *remote_vec_ptr = ptx::ld_shared(smem_vec_ptr);
                        }
                    }
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1 and epilogue_thread_idx == 0) {
                        __threadfence_system();
                        auto* remote_ready_ptr = sym_buffer.map(
                            workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx), dst_rank);
                        *remote_ready_ptr = 1;
                    }
                }
            }
        }
    } else if (warp_idx < kNumRSWarps) {
        auto to_float = [](const cd_dtype_t& value) {
            if constexpr (cute::is_same_v<cd_dtype_t, float>) return value;
            else return static_cast<float>(value);
        };
        auto from_float = [](const float& value) {
            if constexpr (cute::is_same_v<cd_dtype_t, float>) return value;
            else return cd_dtype_t(value);
        };
        auto* y_ptr = static_cast<cd_dtype_t*>(y);
        const uint32_t num_local_tiles = num_m_blocks_per_rank * num_n_blocks;
        for (uint32_t tile_idx = sm_idx * kNumRSWarps + warp_idx;
             tile_idx < num_local_tiles;
             tile_idx += kNumSMs * kNumRSWarps) {
            const uint32_t local_m_block_idx = tile_idx / num_n_blocks;
            const uint32_t n_block_idx = tile_idx - local_m_block_idx * num_n_blocks;
            const uint32_t n_base = n_block_idx * BLOCK_N;
            for (uint32_t elem_idx = lane_idx; elem_idx < BLOCK_M * BLOCK_N; elem_idx += 32) {
                const uint32_t row = elem_idx / BLOCK_N;
                const uint32_t col = elem_idx - row * BLOCK_N;
                if (local_m_block_idx * BLOCK_M + row >= runtime_m_per_rank or n_base + col >= shape_n)
                    continue;
                float acc = 0.0f;
                #pragma unroll 1
                for (uint32_t src_rank = 0; src_rank < kNumRanks; ++ src_rank) {
                    auto* ready_ptr = workspace.get_ready_ptr(src_rank, local_m_block_idx, n_block_idx);
                    while (ptx::ld_acq_sys(ready_ptr) == 0);
                    const auto* partial_ptr = workspace.get_partial_ptr<cd_dtype_t>(
                        src_rank, local_m_block_idx * BLOCK_M + row, n_base + col);
                    acc += to_float(*partial_ptr);
                }
                y_ptr[(local_m_block_idx * BLOCK_M + row) * shape_n + n_base + col] = from_float(acc);
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
