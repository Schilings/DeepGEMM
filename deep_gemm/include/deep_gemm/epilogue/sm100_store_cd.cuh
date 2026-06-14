#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

//
// 即 UMMA 计算完成后，将累加结果从 TMEM 搬回全局显存
//
// 核心流水线：STSM（TMEM → SMEM）→ TMA store（SMEM → global memory）
//            ┌─────────┐     ┌─────────┐     ┌──────────────┐
//            │  TMEM   │ ──→ │  SMEM   │ ──→ │ Global D 矩阵 │
//            │(累加器) │     │(TMA buf)│     │ (bfloat16)   │
//            └─────────┘     └─────────┘     └──────────────┘
//                  ↑ STSM 读       ↑ TMA store 写
//
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
               const uint32_t& tmem_base_addr,
               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
               const cute::TmaDescriptor& tensor_map_cd) {

    // =========================================================================
    // Step 0: 编译期校验
    // =========================================================================
    // TMA store 要求 D 矩阵必须 swizzled，且维度对齐
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    // =========================================================================
    // Step 1: 流水线控制 — stage_idx 在连续 tile 间共享
    // =========================================================================
    // 同一 CTA 处理多个 tile 时，TMA store 流水级在 tile 间复用，
    // 每次 advance 切换到下一个流水级，循环使用 kNumTMAStoreStages 个 stage
    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // =========================================================================
    // Step 2: M 维 wave 循环
    // =========================================================================
    // 输出 tile (BLOCK_M × BLOCK_N) 在 M 维被拆成 kNumMWaves 个 wave，
    // 每个 wave 处理 STORE_BLOCK_M 行，以流水线方式交错执行
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {

        // =====================================================================
        // Step 3: N 维 store atom 循环
        // =====================================================================
        // 每个 wave 内，N 维拆成 kNumStores 个 atom，每个 atom 处理 STORE_BLOCK_N 列
        // 每次迭代推进一次流水（advance_store_pipeline），实现 STSM 与 TMA store 的 overlap
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            // 当前 stage 的 SMEM 基地址（byte 级指针）
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            // =================================================================
            // Step 3a: 等待当前 SMEM 流水级可用
            // =================================================================
            // 只有 warp 0 负责等待 TMA store 完成（SMEM 被 DMA 释放），
            // 然后所有线程 NamedBarrier 同步，确保 SMEM 可安全写入
            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            // 当前 atom 的全局坐标（M 维 = 基地址 + wave 偏移，N 维 = 基地址 + store 偏移）
            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);

            // =================================================================
            // Step 3b: STSM — 从 TMEM 读出数据并写入 SMEM
            // =================================================================
            // 遍历当前 atom 的每个 bank group
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                // ------ 地址计算 ------
                // TMEM 地址：基址 + M wave 偏移 + N store 偏移 + element 偏移
                // TMEM 中数据按 (M, N) 行主序排布，每行 BLOCK_N 个元素
                uint32_t tmem_addr = tmem_base_addr +                                  // 当前 accum stage 的 TMEM 基址
                                     w * BLOCK_N +                                     // M 维 wave 偏移
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;    // N 维偏移 + element 偏移

                // SMEM 地址：按 bank group + swizzle 排布，经过比特异或 swizzle 避免 bank conflict
                //                  ┌── warp 内按 lane_idx 索引 ──┐
                //  layout 概念:  (lane, row_within_atom, col_within_atom)
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                // Swizzle: col 与 row 做比特异或，打散访问模式，消除 bank conflict
                col ^= row % (kSwizzleCDMode / 16);

                // SMEM 地址 = warp 偏移 (每个 warp 32 线程 × swizzle 字节) + row×bank_bytes + col×bank_bytes
                auto smem_ptr = smem_base_ptr +                                             // stage 基址
                                epilogue_warp_idx * 32 * kSwizzleCDMode +                   // warp 偏移
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // atom 内偏移

                // ------ 数据搬运：TMEM load → STSM (TMEM → SMEM) ------
                // 使用 SM100 的 tcgen05 TMEM load 指令，直接从 tensor memory 读到寄存器，
                // 然后用共享内存写指令 (st_shared) 写入 SMEM
                uint32_t values[kNumElemsPerBankGroup];
                if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                    // FP32 输出：每次 load 4 个 float（128 bit）
                    // 使用 SM100_TMEM_LOAD_32dp32b4x: 32 线程协作读取 32 深度的 TMEM
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3]);
                    cutlass::arch::fence_view_async_tmem_load();
                    ptx::st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                } else {
                    // BF16 输出：每次 load 8 个 bf16（128 bit），然后 cast+pack 成成对的 32-bit 写 SMEM
                    // 使用 SM100_TMEM_LOAD_32dp32b8x: 32 线程，8×32=256 个 bf16/iter
                    cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                        values[0], values[1], values[2], values[3],
                        values[4], values[5], values[6], values[7]);
                    cutlass::arch::fence_view_async_tmem_load();
                    // 将相邻两个 FP32 值 cast 为 BF16 后 pack 成一个 uint32_t 写入 SMEM
                    // 性能关键：减少 SMEM 写入量（pack 后减半）
                    ptx::st_shared(
                        smem_ptr,
                        math::cast_into_bf16_and_pack(values[0], values[1]),
                        math::cast_into_bf16_and_pack(values[2], values[3]),
                        math::cast_into_bf16_and_pack(values[4], values[5]),
                        math::cast_into_bf16_and_pack(values[6], values[7])
                    );
                }
            }

            // =================================================================
            // Step 3c: 通知 TMEM 已空（仅在最后一个 atom 的最后一个 wave）
            // =================================================================
            // 完成整个 tile 的 STSM 后，通知 UMMA（accum_stage 的下游）可以重用该 TMEM 区
            // 只需在最后一次执行，因为整个 accum_stage 的 TMEM 此时才全部读空
            if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            // =================================================================
            // Step 3d: TMA store — 将 SMEM 中的数据异步写回全局 D 矩阵
            // =================================================================
            // 所有线程同步保证 SMEM 写完，然后 warp 0 中单个线程发起 TMA store
            // TMA store 是异步的：发起后立即返回，DMA 在后台搬运，kernel 继续处理下一阶段
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                // 根据 GEMM 类型选择 TMA 指令：
                //   Batched: 3D (M, N, batch) store/reduce
                //   Normal:  2D (M, N) store/reduce
                // 根据 kWithAccumulation 选择：
                //   累加模式: TMA_REDUCE_ADD（原子加，用于 D = α·A·B + β·D）
                //   覆盖模式: TMA_STORE（直接写入）
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

} // namespace deep_gemm::epilogue
