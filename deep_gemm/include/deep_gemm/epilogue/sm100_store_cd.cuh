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
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t); // 8 
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
    //
    // 数据分块全景（以 BLOCK_M=128, BLOCK_N=128, STORE_BLOCK_M=128, STORE_BLOCK_N=64 为例）
    //
    // ╔══════════════════════════════════ Level 1: 输出 tile (128×128 BF16) ═══╗
    // ║                                                                        ║
    // ║        N=0              N=63  N=64              N=127                  ║
    // ║        ┌─────────────────────┬─────────────────────┐                   ║
    // ║        │     atom (0,0)      │     atom (0,1)      │ M=0               ║
    // ║        │   STORE_BLOCK_N=64  │   STORE_BLOCK_N=64  │                   ║
    // ║        │                     │                     │                   ║
    // ║        │                     │                     │                   ║
    // ║        └─────────────────────┴─────────────────────┘ M=127             ║
    // ║        迭代: (w=0,s=0) → (w=0,s=1)  每步推进 stage_idx                  ║
    // ║                                                                        ║
    // ║  ════════════════ Level 2: 单个 atom (128×64) 的 warp 分区 ═══════      ║
    // ║         ◄─── 128B = 64 个 BF16 ───►                                    ║
    // ║    ┌────────────────────────────────┐                                  ║
    // ║    │ lane 0..31 → 32 行             │ warp 0 (epilogue_warp_idx=0)     ║
    // ║    │                                │        rows 0..31                ║
    // ║    ├────────────────────────────────┤                                  ║
    // ║    │ lane 0..31 → 32 行             │ warp 1 (epilogue_warp_idx=1)     ║
    // ║    │                                │        rows 32..63               ║
    // ║    ├────────────────────────────────┤                                  ║
    // ║    │ lane 0..31 → 32 行             │ warp 2 (epilogue_warp_idx=2)     ║
    // ║    │                                │        rows 64..95               ║
    // ║    ├────────────────────────────────┤                                  ║
    // ║    │ lane 0..31 → 32 行             │ warp 3 (epilogue_warp_idx=3)     ║
    // ║    │                                │        rows 96..127              ║
    // ║    └────────────────────────────────┘                                  ║
    // ║    4 warp × 32 lane = 128 线程, 每 lane 写 1 行, warp 间不重叠           ║
    // ║                                                                        ║
    // ║  ═════════════ Level 3: 单线程 = 1 行 × 64 元素写 SMEM ═══════════      ║
    // ║    TMEM_LOAD(128bit) → 寄存器 → st_shared(SMEM)                        ║
    // ║    整 atom STSM 完成后 → TMA store 异步写回全局 D                       ║
    // ╚════════════════════════════════════════════════════════════════════════╝
    //
    // 🔄 swap-AB 场景 (STORE_BLOCK_M=16, STORE_BLOCK_N=128): M 维拆成 8 个 wave
    //    N 维不拆分, 迭代: (w=0,s=0) → (w=1,s=0) → ... → (w=7,s=0)
    //
    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {

        // =====================================================================
        // Step 3: N 维 store atom 循环
        // =====================================================================
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            // 当前 stage 的 SMEM 基地址（byte 级指针）
            // ⚠️ TMEM 与 SMEM CD 不是 1:1 对应，而是 1:N（1 个 TMEM stage → N 个 SMEM CD atom）
            //
            // 根本原因：SMEM CD 的 N 维容量受 swizzle stripe (128B) 限制，远小于 TMEM 的 N 维
            //   TMEM     N 维 = BLOCK_N = 128 列 FP32 = 512B/行
            //   SMEM CD  N 维 = kSwizzleCDMode = 128B = 64 列 BF16/行
            //   比例 = 128 / 64 = 2 → 1 个 TMEM tile 需要 2 个 SMEM CD atom
            //
            // 因此虽然两者都是双缓冲，但粒度不同：
            //   kNumEpilogueStages=2: UMMA 双缓冲 TMEM，tile 级
            //   kNumTMAStoreStages=2:  epilogue 双缓冲 SMEM CD，atom 级（1/N tile）
            //
            //     TMEM[0] (128×128 FP32)             SMEM CD (2 stage × 128×64 BF16)
            //     ┌─────────────────────┐            ┌──────────┐  ┌──────────┐
            //     │                     │    atom0   │ stage[0] │  │ stage[1] │
            //     │    accum_stage=0    │──────────▶│ 128×64   │  │ 128×64   │
            //     │                     │    atom1   │   TMA──▶│  │   TMA──▶│
            //     └─────────────────────┘            └──────────┘  └──────────┘
            //                                                ▲ 轮转复用 ▲
            //     TMEM[1] (128×128 FP32)           atom0: stage[?] ← 取决于 tile 间流水
            //     ┌─────────────────────┐          atom1: stage[?] ← advance 继续轮转
            //     │    accum_stage=1    │
            //     └─────────────────────┘
            //
            // tma_stage_idx 在 tile 间持续轮转（不重置），实现跨 tile 流水复用
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
            //
            // 每个 lane 处理一整行 64 个 BF16 元素，分 8 次迭代搬运（每次 8 个 = 1 bank group = 16B）
            //
            //   TMEM (连续地址)                           SMEM (swizzle 后，row=lane_idx)
            //   ┌── i=0: 元素  0.. 7 (128bit) ──┐      ┌─ col=0 → d0..d7  ─┐
            //   │  i=1: 元素  8..15             │      │  col=1 → d8..d15   │
            //   │  ...                          │ ──▶  │  ...                │  ← 同 1 行
            //   │  i=7: 元素 56..63             │      │  col=7 → d56..d63  │
            //   └───────────────────────────────┘      └────────────────────┘
            //
            //   8 次迭代 × 8 元素/次 = 64 元素 = STORE_BLOCK_N，lane 间无依赖
            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) { 
                // ------ 地址计算 ------
                // TMEM 地址：基址 + M wave 偏移 + N store 偏移 + element 偏移
                // TMEM 中数据按 (M, N) 行主序排布，每行 BLOCK_N 个元素
                uint32_t tmem_addr = tmem_base_addr +                                  // 当前 accum stage 的 TMEM 基址
                                     w * BLOCK_N +                                     // M 维 wave 偏移
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;    // N 维偏移 + element 偏移

                // SMEM 地址 = warp 偏移 + atom 内 (row, col)，详见上方全景图的 Level 2/3
                // warp 间不重叠，swizzle 消 bank conflict
                // i + lane_idx * 8
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes); 
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8; // True
                // i / 8 + lane_idx ->  row: 0 ~ 31
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                // i -> i ^ (row % 8) -> col: 0 ~ 7
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);
                //
                // XOR swizzle 原理（一行就懂）：
                //   col 全相同 → 全挤一个 bank → conflict
                //   row%8 各不相同 → 每个 lane 有个不同的"签名"
                //   col ^= row%8 → 用 row 的差异去扰动 col → 相同的 col 被散成不同的 bank
                //
                //   XOR 保证 {0^c, 1^c, ..., 7^c} 一定是 {0..7} 的排列（每种 c 对应一种洗牌方式）
                //   8 个 lane 的 row%8 恰好是 0..7 → 8 种排列 → 8 个不同 bank → 0-conflict
                // 
                //i=0 时，col 从 0→7，XOR row%8 的结果：
                // row%8=0: [0][1][2][3][4][5][6][7]    恒等排列
                // row%8=1: [1][0][3][2][5][4][7][6]    邻位交换
                // row%8=2: [2][3][0][1][6][7][4][5]    隔位交换
                // row%8=3: [3][2][1][0][7][6][5][4]    4位翻转
                // row%8=4: [4][5][6][7][0][1][2][3]    前后半交换
                // row%8=5: [5][4][7][6][1][0][3][2]
                // row%8=6: [6][7][4][5][2][3][0][1]
                // row%8=7: [7][6][5][4][3][2][1][0]    完全逆序
                // 
                // st.shared.b128（128bit = 16B）的调度粒度是 quarter-warp（8 lane），
                // 32 个活跃 lane → 4 次串行 memory transaction，bank conflict 分析按每 quarter-warp 独立计算。
                //
                // i=0 时 lane 0..31 的 swizzle 映射（kSwizzleCDMode=128, BF16）：
                //
                //     QW   lane  row  col^=row%8   bank
                //     ──   ────  ───  ──────────   ────
                //      0     0     0       0          0
                //      0     1     1       1          4
                //      0    ...   ...     ...        ...
                //      0     7     7       7         28     ← QW0: 8 lane 写 8 个不同 bank，0-way ✅
                //     ──   ────  ───  ──────────   ────
                //      1     8     8       0          0
                //      1     9     9       1          4
                //      1    ...   ...     ...        ...
                //      1    15    15       7         28     ← QW1: 同上，0-way ✅
                //     ──   ────  ───  ──────────   ────
                //      2   16~23  16~23  0~7       0,4,..,28  ← QW2: 0-way ✅
                //      3   24~31  24~31  0~7       0,4,..,28  ← QW3: 0-way ✅
                //
                // 无 swizzle（所有 lane col=0）：每个 QW 内 8 lane 全打 bank 0 → 8-way conflict ❌
                //                       4 QW × 8 cycle/QW = 32 cycle
                // 有 swizzle：每个 QW 内 8 lane 各打不同 bank → 0-way conflict ✅
                //             4 QW × 1 cycle/QW = 4 cycle（快 8 倍）
                // SMEM 地址 = warp 偏移 + row×bank_bytes + col×bank_bytes
                auto smem_ptr = smem_base_ptr +                                             // stage 基址
                                epilogue_warp_idx * 32 * kSwizzleCDMode +                   // warp 偏移（每 warp 独占 32 行 × 128B）
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // atom 内偏移（行 × 128B + 列 × 16B）

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
            // TMA 硬件在读出 SMEM 时自动做逆向 XOR swizzle，故全局 D 中的元素顺序正确：
            // SMEM 中 swizzle 只是为了消除 bank conflict，最终语义由 TMA 保证
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
