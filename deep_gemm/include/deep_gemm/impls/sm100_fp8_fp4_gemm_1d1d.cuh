#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// ============================================================================
// SM100 FP8×FP4 (或 FP8×FP8) GEMM Kernel — 1D-1D TMA 加载策略
// ============================================================================
// "1D-1D" 指 A、B 矩阵均沿 K 维度做 1D 分块加载（区别于 1D-2D 模式）
//
// 模板参数说明：
//   kMajorA / kMajorB        — A/B 矩阵的 UMMA 主维度（K 或 MN），决定 TMA 加载的坐标顺序
//   kGranKA / kGranKB        — A/B 的缩放因子(SF)量化粒度：每 kGranK 个元素共享 1 个 SF（32 或 128）
//   SHAPE_M/N/K              — 矩阵维度，0 表示运行时传入，非零则编译期固定以利优化
//   BLOCK_M/N/K              — GEMM tile 大小，如 128×128×128，由 heuristics 根据问题规模自动选择
//   kNumGroups                — 分组 GEMM 的组数（单次 GEMM = 1）
//   kSwizzleAMode/BMode/CDMode — smem swizzle 模式，用于避免 shared memory bank conflict
//   kNumStages                — 软件 pipeline 的 stage 数（通常 2~4），更多 stage 可隐藏延迟但占用更多 smem
//   kNumNonEpilogueThreads    — GEMM 计算部分线程数（warp 0/1/2），不含 epilogue
//   kNumEpilogueThreads       — Epilogue（写回结果）线程数
//   kNumMulticast             — TMA multicast CTA 数（1 = 单 CTA，2 = 2-CTA 共享 TMA 传输）
//   kIsMulticastOnA           — multicast 作用于 A 矩阵（true）还是 B 矩阵（false）
//   kNumSMs                   — 使用的 SM 数量，用于调度器分块
//   kSwapAB                   — 是否交换 A/B（MoE 场景下 M 维度小时交换以提高利用率）
//   kGemmType                 — GEMM 类型：Normal / MGroupedContiguous / MGroupedMasked / KGroupedContiguous / Batched
//   kWithAccumulation         — 是否累加 C 矩阵（D = C + AB）
//   a_dtype_t                 — A 矩阵数据类型（FP8 e4m3）
//   b_dtype_t                 — B 矩阵数据类型（FP4 e2m1 或 FP8 e4m3）
//   cd_dtype_t                — 输出数据类型（BF16 或 FP32）
//   epilogue_type_t           — Epilogue 处理类型 
// ============================================================================
template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB, //— A/B 矩阵的 UMMA 主维度（K 或 MN），决定 TMA 加载的坐标顺序
          uint32_t kGranKA, uint32_t kGranKB,                   //— A/B 的缩放因子(SF)量化粒度：每 kGranK 个元素共享 1 个 SF（32 或 128）
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K, //— 矩阵维度，0 表示运行时传入，非零则编译期固定以利优化
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, //— GEMM tile 大小，如 128×128×128，由 heuristics 根据问题规模自动选择
          uint32_t kNumGroups,                                  //— 分组 GEMM 的组数（单次 GEMM = 1）
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode, //— smem swizzle 模式，用于避免 shared memory bank conflict
          uint32_t kNumStages,                                          //— 软件 pipeline 的 stage 数（通常 2~4），更多 stage 可隐藏延迟但占用更多 smem
          uint32_t kNumNonEpilogueThreads,                              //— GEMM 计算部分线程数（warp 0/1/2），不含 epilogue
          uint32_t kNumEpilogueThreads,                                 //— Epilogue（写回结果）线程数
          uint32_t kNumMulticast,                                       //— TMA multicast CTA 数（1 = 单 CTA，2 = 2-CTA 共享 TMA 传输）
          bool kIsMulticastOnA,                                         //— multicast 作用于 A 矩阵（true）还是 B 矩阵（false） 
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType,                                           //— GEMM 类型：Normal / MGroupedContiguous / MGroupedMasked / KGroupedContiguous / Batched 
          bool kWithAccumulation,                                       //— 是否累加 C 矩阵（D = C + AB）
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t, 
          typename epilogue_type_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_fp4_gemm_1d1d_impl(int* grouped_layout,            // 分组 GEMM 的布局信息（MGrouped 模式下每组的 M 偏移）
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,  // 运行时矩阵维度（编译期未固定时使用）
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,   // A 矩阵的 TMA tensor map（HBM 布局描述符）
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,   // B 矩阵的 TMA tensor map
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,  // A 的缩放因子(SFA)的 TMA tensor map
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,  // B 的缩放因子(SFB)的 TMA tensor map
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {  // 输出 C/D 矩阵的 TMA tensor map
//#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    // 2-CTA multicast 需要 2Sm TMEM 分配器，单 CTA 用 1Sm 分配器
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // ========================================================================
    // MMA（矩阵乘累加）配置
    // ========================================================================
    constexpr uint32_t LAYOUT_AD_M = 128;                          // UMMA A/D 矩阵的 M 维行对齐粒度（硬件要求 128）
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;       // 2-CTA 时 UMMA_M = 256（两个 CTA 各负责 128 行）
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;       // swap-AB 时 UMMA_N 对应原 M 维度
    constexpr uint32_t UMMA_K = 32;                                 // UMMA 指令的 K 维度固定为 32 元素
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);  // multicast 在 A 上时，每个 CTA 只加载半行
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);  // multicast 在 B 上时，每个 CTA 只加载半列
    // ⚠️ BLOCK_K = 128
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");           // K 方向固定 128（匹配 TMA 传输粒度）
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");
 
    // ========================================================================
    // 缩放因子 (SF) 配置
    // ========================================================================
    constexpr uint32_t kNumUTCCPAlignedElems = 128;                 // UTCCP 一次搬运 128 个 SF 元素（4×32 转置布局）
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);  // SF 的 M 维度（向上对齐到 128）
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);  // SF 的 N 维度（向上对齐到 128）
    // kGranK=32 时每个 K block 都需加载 SF，kGranK=128 时每 4 个 K block 才需加载一次
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT((kGemmType != GemmType::KGroupedContiguous) or kGranKA == kGranKB, "K-grouped SF requires kGranKA == kGranKB");

    // ========================================================================
    // Epilogue（结果写回）配置
    // ========================================================================
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;                      // tmem 累加器 pipeline 2 级
    constexpr uint32_t kNumTMAStoreStages = 2;                       // TMA store pipeline 2 级
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);  // 每次 store 的 M 子块大小
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);        // 每次 store 的 N 子块大小
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;  // 参与 UMMA store 的线程数
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // ========================================================================
    // 共享内存 (smem) 大小计算
    // ========================================================================
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);  // 每阶段输出 smem 大小
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;                   // 输出 smem 总大小
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);           // 每阶段 A smem 大小
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);           // 每阶段 B smem 大小
    // BLOCK_K = 128 -> 32个元素 对应一个 SF -> 4个SF UE8M0格式 -> 1个uint32
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);                      // 每阶段 SFA smem（1 个 uint32 对齐 128 元素）
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);                      // 每阶段 SFB smem
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // ========================================================================
    // 张量内存 (tmem) 配置 — 列分配与偏移
    // ========================================================================
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;   // 累加器占用列数（2 级 × UMMA_N 列）
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;                 // SFA 占用列数（每 32 个 SF 占 1 列）
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;                 // SFB 占用列数
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();  // 对齐后的总列数
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;            // SFA 在 tmem 中的起始列
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;  // SFB 在 tmem 中的起始列
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // 2-CTA multicast 时需先同步整个 cluster，确保 TMEM 分配前所有 CTA 就绪
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // ========================================================================
    // 基础工具变量
    // ========================================================================
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;  // cluster 内第 0 个 CTA 为 leader
    const auto warp_idx = cutlass::canonical_warp_idx_sync();        // 当前 warp 在 CTA 内的编号
    const auto lane_idx = ptx::get_lane_idx();                       // 当前线程在 warp 内的编号 (0~31)

    // ========================================================================
    // 预取 TMA 描述符 — warp 0 在最开始就预取，减少后续 TMA 发射延迟
    // ========================================================================
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // ========================================================================
    // 运行时矩阵维度：编译期未固定（=0）时使用运行时传入值
    // ========================================================================
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    // SF 的 K 维度数量 = ceil(shape_k / (kGranK * 4))
    // kGranK 为基础量化粒度，×4 是因为 4 个 UE8M0 SF 打包成 1 个 uint32
    // BLOCK_K = 128 -> 32个元素 对应一个 SF -> 4个SF UE8M0格式 -> 1个uint32
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    // ========================================================================
    // 共享内存布局：CD | A[stages] | B[stages] | SFA[stages] | SFB[stages] | Barriers | TMEM ptr
    // ========================================================================
    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = reinterpret_cast<uint8_t*>(smem_b[kNumStages]);
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // ========================================================================
    // Barrier 布局：full[kNumStages] | empty[kNumStages] | with_sf_full[kNumStages]
    //             | tmem_full[kNumEpilogueStages] | tmem_empty[kNumEpilogueStages] | tmem_ptr
    // ========================================================================
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_sfb[kNumStages]);;
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });                  // TMA 数据到达 barrier
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });     // 消费者释放 barrier
    auto with_sf_full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); }); // TMA + SF转置都完成的 barrier
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); }); // tmem 累加器就绪 barrier
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });  // tmem 释放 barrier
    auto tmem_ptr_in_smem  = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);  // TMEM 分配结果指针

    // ========================================================================
    // 初始化 barrier 和 TMEM 分配
    // ========================================================================
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);                // TMA 完成后 arrive（1 个生产者）
            empty_barriers[i]->init(1);               // 消费者完成后 arrive（1 个消费者）
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);  // 32 个 UTCCP 转置线程 arrive，2-CTA 时 ×2
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);                              // MMA 完成后 arrive
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);  // epilogue 线程完成后 arrive
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory — warp 2 负责分配 tmem 列
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion（用于 persistent kernel 模式）
    cudaGridDependencySynchronize();

    // ========================================================================
    // 块调度器 — 持久化调度 GEMM tiles（M×N 块）到各 CTA
    // ========================================================================
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kGranKA * 4>(
        shape_m, shape_n, shape_k, grouped_layout);

    // ========================================================================
    // Pipeline 控制：stage 轮转和 phase 翻转
    // ========================================================================
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ========================================================================
    // Warp 角色分派：
    //   warp 0  → TMA 加载（生产者）：从 HBM 搬运 A/B/SFA/SFB 到 smem
    //   warp 1  → MMA 发射（消费者）：发起 UTCCP + UMMA 指令
    //   warp 2  → UTCCP 转置（生产者）：smem 中 SF 的 4×32 转置
    //   warp ≥3 → Epilogue（消费者）：从 tmem 读结果，量化后写回 HBM
    // ========================================================================
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // ====================================================================
        // TMA load warp — 持久化调度，循环处理所有 tile
        // ====================================================================
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release — 等待消费者释放当前 stage
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets — multicast 模式下根据 CTA rank 偏移 M 或 N
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // ====================================================================
                // 发射 TMA 加载指令 — 根据 major 布局选择 inner/outer 维度顺序
                // ====================================================================
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                // 预计到达字节数（FP4 数据按 2 倍压缩，实际 TMA 传输量减半）
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
                                         SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

                // ====================================================================
                // 按需加载 SFA/SFB（kGranK=128 时每 4 个 K block 才需加载一次）
                // ====================================================================
                // No swizzling, so one TMA for one SF is enough
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::SF_K>(
                        shape_sfa_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad));
                    tma::copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
                        shape_sfb_k, 1, math::ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx);
                    tma::copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                // Arrive at full barriers — 通知消费者数据已就绪
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }

    } else if (warp_idx == 1 and is_leader_cta) {
        // ====================================================================
        // MMA issue warp — 只有 leader CTA 的 warp 1 发射 UMMA 指令
        // ====================================================================
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor — 创建 UMMA 指令描述符
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);  // SF 的 smem 描述符（运行时替换地址）

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // 预计算各 stage 的 smem 描述符低位（lo）偏移，避免循环内重复计算
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);

        // 不同的stage的base地址记在前几个lane内
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // 等待 tmem 累加器被 epilogue 释放
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival — 通知 TMA warp 当前 stage 已消费完毕
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }
 
            // ====================================================================
            // 发射 UMMA 指令 — 遍历 K 方向的 block
            // ====================================================================
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // 等待 TMA 数据 + SF 转置都完成
                with_sf_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                // 交换当前 stage 的 smem 描述符，同时为下一次预取
                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);

                if (cute::elect_one_sync()) {
                    // ------------------------------------------------------------------
                    // 1) UTCCP：将 smem 中的 SF 搬运到 tmem（4×32 转置布局）
                    // ------------------------------------------------------------------
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                    }

                    // ------------------------------------------------------------------
                    // 2) 发射 UMMA FMA 指令 — BLOCK_K 内按 UMMA_K=32 分成多次 FMA
                    // ------------------------------------------------------------------
                    using mma_t = cute::conditional_t<
                        kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        // kGranK=32 时每次 FMA 用不同的 SF；kGranK=128 时同一组 SF 复用 4 次
                        const uint32_t sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = kSwapAB ?
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id):
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        // 沿 K 方向推进 smem 描述符
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       kTmemStartColOfSFB, kTmemStartColOfSFA);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc,
                                       kTmemStartColOfSFA, kTmemStartColOfSFB);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);  // 最后一个 K block 时同时 arrive tmem_full
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }


    } else if (warp_idx == 2) {
        // ====================================================================
        // UTCCP transposer warp — 将 smem 中的 SF 做 4×32 转置，供 UTCCP 搬运到 tmem
        // ====================================================================
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            // 转置的目的：UTCCP 往 tmem 搬运时，要求同一 tmem 列的 4 个 token 的 SF 连续存放。
            // 转置前 (4×32):            转置后 (32×4):
            // token  0  1  2 ... 31     token 0  32  64  96  ← 连续4个→同一 tmem 列
            // token 32 33 34 ... 63     token 1  33  65  97
            // token 64 65 66 ... 95     token 2  34  66  98
            // token 96 97 98 ... 127    ...

            // 同一行的32个token连续     同一tmem列的4个token连续

            uint32_t values[4];
            // 读取 4 个值：lane_idx 对应的 4 个位置（交错读取避免 bank conflict）
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                // lane_idx：0~31，代表 32 列中的哪一列
                // lane_idx >> 3：lane 所在的"8 元素组"编号（0~3），即 4 行中哪一行的候选
                // i：0~3，代表要读 4 行
                // i ^ (lane_idx >> 3)：XOR 交换行号，这是转置的关键
                // 本质就是一个用 XOR 交织避免 smem bank conflict 的经典矩阵转置技巧。
                values[i] = ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx); 

                // ⚠️ 行：(i ^ (lane_idx >> 3) 
                // ⚠️ 列： lane_idx
            __syncwarp();


            // 写回转置后的 4×32 布局：同一列的 4 个元素连续存放 
            //  XOR 的真正目的是避免写的 bank conflict。
            // 0^0=0  0^1=1  0^2=2  0^3=3
            // 1^0=1  1^1=0  1^2=3  1^3=2
            // 2^0=2  2^1=3  2^2=0  2^3=1
            // 3^0=3  3^1=2  3^2=1  3^3=0

            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);

        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                    // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                    cutlass::arch::fence_view_async_shared();
                }

                // Arrive — 通知 MMA warp SF 转置已完成
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }

    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // ====================================================================
        // Epilogue warp groups — 从 tmem 读累加结果，量化后通过 TMA store 写回 HBM
        // ====================================================================
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival — 等待 tmem 累加器就绪
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            // swap-AB 和非 swap-AB 使用不同的 epilogue store 实现
            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<
                    BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue_type_t>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

// #else
//     if (blockIdx.x == 0 and threadIdx.x == 0)
//         DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
// #endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
