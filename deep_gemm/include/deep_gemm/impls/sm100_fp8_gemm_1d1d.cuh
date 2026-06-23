#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/epilogue_utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm100_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm100;

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, bool kWithAccumulation,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_type_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_fp8_gemm_1d1d_impl(int* grouped_layout, 
                         uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M; // kNumMWaves = 1 or 2
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t kNumUTCCPAlignedElems = 128; 
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K"); 
    DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 and 2 % kNumMWaves == 0, "Invalid block M");

    // kGranKA和kGranKB表示A和B的per-block量化维度
    // 如果 kGranKA=32，那么一个BlockK=128就是4个SF(UE8M0),对应1个UINT4，所以kNumSFAStagesPerLoad=1 每1次k 都要加载 1次 SF
    // 如果 kGranKA=128，那么一个BlockK=128就是1个SF(UE8M0)，对应0.25个UINT4，所以kNumSFAStagesPerLoad=4 每4次k 加载 1次 SF
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    // SF加载的步长是4，4个UE8M0对于一个UINT4，所以shape_k//（kGranKA * 4）
    const uint32_t shape_sfa_k = ceil_div(shape_k, kGranKA * 4);
    const uint32_t shape_sfb_k = ceil_div(shape_k, kGranKB * 4);

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = get_lane_idx();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // 2-CTA MMA
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1); //  BLOCK_M
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast); // BLOCK_N/2
    constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    // STORE_BLOCK_N: kSwizzleCDMode=128 是 TMA swizzle stripe 字节宽, ÷sizeof 转元素数 → BF16:64, FP32:32
    constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
    // epilogue 线程数 = STORE_BLOCK_M, 每个线程负责 M 维的一个 element 行
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    // 禁止 A-side multicast: FP8 block-scaled GEMM 只能做 B-side (N 维) 2-CTA
    // A 矩阵的 SFA 走 UTCCP 入 TMEM, SF 列对齐要求特殊, A 必须每 CTA 完整加载
    DG_STATIC_ASSERT(not kIsMulticastOnA or kNumMulticast == 1, "Invalid multicast");
    DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M, "Only support tensor memory layout A/D");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SF_BLOCK_M = constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for UMMA");

    // Automatically deduce the number of epilogue stages (1 or 2), according to the tensor memory size
    // TODO: test cases of `kNumMWaves == 2 and kNumEpilogueStages == 2`
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumEpilogueStages = (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1 : 2;

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // D/A/B shared memory
    // SMEM 布局: [smem_cd(num_store_stages) | smem_a(num_stages) | smem_b(num_stages) | sfa(num_stages) | sfb(num_stages) | barriers | tmem_ptr]
    auto smem_cd = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE); 
    });
    auto smem_a  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<a_dtype_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<b_dtype_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SFA/SFB shared memory
    auto sf_start_ptr = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Fill barriers
    // 将数据区之后的 smem 原始字节 reinterpret_cast 为 Barrier* 数组
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer +
        SMEM_CD_SIZE +
        kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
        kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
    // ── TMA ↔ MMA 流水线 barrier ──
    auto full_barriers              = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    // with_sf_full_barriers: TMA load (包含 A/B/SFA/SFB) 完成信号, 32 线程 arrive → init(kNumMulticast*32)
    auto with_sf_full_barriers      = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    // ── TMEM 流水线 barrier ──
    auto tmem_full_barriers         = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + i); });
    auto tmem_empty_barriers        = PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i); });

    // TMEM 地址存放在 shared memory 中，紧接在所有 barrier 之后
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    if (kNumMulticast > 1)
        cute::cluster_sync();

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Arrive only at the leader CTA
            with_sf_full_barriers[i]->init(kNumMulticast * 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * LOAD_BLOCK_M) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, 1, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, a_dtype_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma_copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, 1, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma_copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, 1, batch_idx);
                auto num_arrival_bytes = SMEM_A_SIZE_PER_STAGE / (std::is_same_v<a_dtype_t, cutlass::float_e4m3_t> ? 1 : 2) +
                                         SMEM_B_SIZE_PER_STAGE / (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

                // Issue SFA and SFB TMAs at certain stages
                // No swizzling, so one TMA for one SF is enough
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, full_barriers[stage_idx], smem_sfa[stage_idx], m_block_idx * BLOCK_M,
                                            scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::SF_K>(shape_sfa_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad)));
                    num_arrival_bytes += BLOCK_M * sizeof(uint32_t);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, full_barriers[stage_idx], smem_sfb[stage_idx], n_block_idx * BLOCK_N,
                                            scheduler.template get_global_idx<true, IndexType::SF_K>(shape_sfb_k, 1, ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad), m_block_idx));
                    num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
                }

                // Arrive at full barriers
                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp — only the leader CTA will do this
        // 创建 UMMA block-scaled 指令描述符 (FP8×FP8→FP32, 带 UE8M0 scale factor)
        // TODO: refactor `UMMA_M` calculation
        constexpr uint32_t UMMA_M = LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
        constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
        constexpr uint32_t UMMA_K = 32;
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                   UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");

        /*
        ═══════════════════════════════════════════════════════════════════════
        UMMA SmemDescriptor 构造 — 与 TMA 的 swizzle 协议匹配
        ═══════════════════════════════════════════════════════════════════════

        TMA 存入 SMEM 时按 kSwizzleAMode/kSwizzleBMode 打乱了物理字节顺序
        (如 swizzle-128B: atom 0=所有行×前64列, atom 1=所有行×后64列, 物理不连续)。

        make_umma_desc 将同一个 swizzle 模式编码进 SmemDescriptor, 包含 4 个关键字段:
          - layout_type_:        SWIZZLE_128B / SWIZZLE_64B / ... → 硬件由此知反解算法
          - start_address_:      SMEM 起始物理地址 >> 4
          - stride_byte_offset_: atom 间跨步 (MN 方向跳一个 atom 的字节距)
          - leading_byte_offset_: 行列间跨步 (K 方向跳一行/一列的字节距)

        UMMA::fma 发射时, SM100 硬件读取此描述符, 自动将 (逻辑行列) 映射到 (SMEM 物理地址),
        无需软件手动反解 swizzle。这就是 TMA(生产者) 和 UMMA(消费者) 的协议层。

        各 lane 预计算自己负责 stage 的 desc.lo:
          - kNumStages 个 lane 各管一个 stage 的 SMEM 偏移
          - /16 是因为 SmemDescriptor.start_address_ 以 16B 为单位
          - 其余 lane 置 0 (不会用到)
        后续通过 __shfl_sync 广播 stage_idx 对应 lane 的值, 无需查表。
        ═══════════════════════════════════════════════════════════════════════
        */
        auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
        // desc.lo = 低 32 位: start_address | layout_type | base_offset, 包含 stage 0 的 SMEM 起始地址
        // + lane_idx * SMEM_xxx_SIZE_PER_STAGE / 16: 各 lane 预存其负责 stage 的 desc.lo
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
            // Wait tensor memory empty barrier arrival
            // barrier.wait 之后发射 UMMA 前需要 tcgen05 fence, 确保上一轮 TMEM 写入可见
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            tcgen05_after_thread_sync();

            // Empty barrier arrival
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
            };

            // Launch MMAs
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA and SF-transpose arrival
                with_sf_full_barriers[stage_idx]->wait(phase);
                tcgen05_after_thread_sync(); // barrier.wait 之后 UMMA 前需要 fence

                // UTCCP: 将 UE8M0 scale factor 从 smem 拷贝到 TMEM 的 SF 区
                // SM100 block-scaled UMMA 要求 SF 在 TMEM 的指定列
                // kGranKA=32: 每 1 次 K 做 UTCCP; kGranKA=128: 每 4 次 K 做一次
                using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                    cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                if (sfa_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                    }
                }
                const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                if (sfb_stage_in_group_idx == 0 and cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                        replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                    }
                }
                __syncwarp();

                // Issue UMMA in the leader CTA
                // mma_t: FP8×FP8 block-scaled, 根据 kNumMulticast 选单 SM 或 2-CTA cluster
                using mma_t = cute::conditional_t<kNumMulticast == 1, SM100_MMA_MXF8F6F4_SS, SM100_MMA_MXF8F6F4_2x1SM_SS>;
                // 通过 __shfl_sync 广播: 取出 stage_idx 号 lane 预存的 desc.lo
                // 例: stage_idx=2 时, 所有 lane 会拿到 lane 2 上的 desc_lo (即 stage 2 的 SMEM 起始地址)
                // 这样无需内存查表, 一个 warp shuffle 指令就完成 stage 切换
                const auto& a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    // 内层 K 循环: 每 UMMA_K=32 个 K 元素发一条 UMMA, BLOCK_K=128 时循环 4 次
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        const uint32_t sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);
                        // 将编译期 instr_desc + 运行时 SF ID → 64 位立即数
                        const auto& runtime_instr_desc = make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        /*
                        advance_umma_desc_lo — swizzle 感知的 SMEM 地址推进

                        desc.lo (SmemDescriptor 低 32 位):
                          bits[15:0]:  start_address  (SMEM 物理地址 >> 4, 16B 对齐)
                          bits[22:16]: layout_type    (SWIZZLE_128B/64B/.../NONE)
                          bits[31:23]: base_offset    (原子内子偏移, 通常为 0)

                        推进公式: desc_lo = base_lo + ((offset + k_idx * stride_k) * sizeof(dtype)) >> 4

                        其中 stride_k:
                          - K-major: stride_k = 1 (K 方向逐元素连续)
                          - MN-major: stride_k = BLOCK_INNER_ATOM (K 方向跳跃整个 atom)

                        offset 参数:
                          - B 矩阵: offset=0, B 只有一层, 只需沿 K 推进 k * UMMA_K
                          - A 矩阵: offset = w * WAVE_BLOCK_M * BLOCK_K,
                            跨 M 维度 atom 的大跳 (MN-major 时整个 SMEM 的 M 段跳跃)
                        */
                        b_desc.lo = advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                        #pragma unroll
                        // M 维 wave 循环: FP8 A 矩阵在 M 维可拆为多个 wave, 每个 wave 做一次 fma
                        for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                            DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0, "Invalid swizzling offset");
                            /*
                            A 矩阵 (MN-major) 的 SMEM 物理布局 — 两级跳跃:

                            物理布局 (swizzle128 + MN-major, LOAD_BLOCK_M=128):
                            ┌──────────────────────────────┐  smem 起始
                            │ atom 0: 所有 128 行 × 64 列   │  (128×64, swizzled)
                            └──────────────────────────────┘
                            ┌──────────────────────────────┐  smem + 128×64
                            │ atom 1: 所有 128 行 × 64 列   │  (128×64, swizzled)
                            └──────────────────────────────┘
                            同一行的 col 63 和 col 64 之间隔了 ~128×64 个元素, 完全不连续.

                            advance_umma_desc_lo 的 offset + k_idx 两级寻址:
                              - offset = w * WAVE_BLOCK_M * BLOCK_K
                                wave 级大跳: 跨 M 维 atom (如 MN-major 时跳过整个 M 段)
                              - k_idx = k * UMMA_K
                                K 维微调: 在 atom 内沿 K 方向推进 sub-tile

                            UMMA 硬件通过 SmemDescriptor.layout_type_ 自动反解 swizzle,
                            所以软件只需给出正确的 SMEM 物理起始地址偏移, 不需要手动反排字节.
                            */
                            a_desc.lo = advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                                a_desc_base_lo,
                                w * WAVE_BLOCK_M * BLOCK_K, // ← offset: 跨 M atom 的大跳
                                k * UMMA_K); // ← k_idx: atom 内 K 偏移
                            // fma: D += A × B
                            //   参数3 = accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N: TMEM 列偏移
                            //   参数4 = k_block_idx>0 or k>0: 首 K-block 第一步清零, 后续累加
                            mma_t::fma(a_desc, b_desc,
                                       accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                                       k_block_idx > 0 or k > 0,
                                       runtime_instr_desc,
                                       kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                                       kTmemStartColOfSFB);
                        }
                    }
                }

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto& iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto& accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx == 2) {
        /*
        ═══════════════════════════════════════════════════════════════════════
        Warp 2 — UTCCP SMEM 转置 (三级流水线的第二级)
        ═══════════════════════════════════════════════════════════════════════

        数据流全貌:
          HBM                       SMEM                          SMEM                       TMEM
        ┌──────────┐    TMA     ┌─────────────┐   转置       ┌─────────────┐   UTCCP    ┌──────────┐
        │ sf[M][K] │ ────────→  │  row-major   │ ──────────→ │  K-major    │ ─────────→ │ col[k]   │
        │ row-major│ (warp 0)   │ sf[0..127]   │  (warp 2)   │ 32-lane交织  │ (warp 1)  │ 32b×N列  │
        └──────────┘            └─────────────┘              └─────────────┘            └──────────┘

        为什么需要转置？
        ─────────────────
        1. TMA 将 SFA 加载为 row-major: smem_sfa[0..127] 连续存放 128 个 uint32,
           即第 i 个位置 = sf[M_start + i] (行优先, 逐 M 排列).

        2. UTCCP (SM100_UTCCP_4x32dp128bit) 要求 SMEM 数据为 K-major 布局:
           - 每"行"宽度 = 128 bits = 4 个 uint32 (K 方向)
           - "行"数由 SMEM descriptor 的 SBO 控制
           - UTCCP 从 SMEM 源拷贝 4 行 × 32 列 (每列 128b) → 输出到 4 个连续 TMEM 列

        3. make_sf_desc 构造的 SMEM descriptor:
             layout = SWIZZLE_NONE, SBO = 8×16 = 128B, LBO = 0
           即: 每行 128b 宽, MN 方向相邻行间隔 128B, K 方向无跳跃.
           UTCCP 以此 layout 解释 SMEM → 需要 K-major 而非 row-major.

        三级 warp 流水线:
          warp 0 (TMA load):   从 HBM 搬 A/B/SF 数据到 SMEM
                   ↓ (full_barrier wait)
          warp 2 (transpose):  将 SF 从 row-major 转置为 K-major (原地改写 SMEM)
                   ↓ (fence_view_async_shared → with_sf_full_barrier arrive)
          warp 1 (UMMA issue):  UTCCP 拷贝 SF 到 TMEM → UMMA fma 计算

        ═══════════════════════════════════════════════════════════════════════
        转置算法: 32-lane warp XOR shuffle
        ═══════════════════════════════════════════════════════════════════════

        将 128 个 uint32 (SF_BLOCK_M 对齐到 128) 原地转置.
        32 个 lane 协作, 每个 lane 处理 4 个元素 (32×4=128).

        XOR 模式解析 (以 lane 0 为例):
          lane_idx=0, lane_idx>>3=0:
            i=0: 读 (0^0)*32+0 = 0×32+0 = 0,      写 0*4+(0^0) = 0
            i=1: 读 (1^0)*32+0 = 1×32+0 = 32,     写 0*4+(1^0) = 1
            i=2: 读 (2^0)*32+0 = 2×32+0 = 64,     写 0*4+(2^0) = 2
            i=3: 读 (3^0)*32+0 = 3×32+0 = 96,     写 0*4+(3^0) = 3
          → lane 0 从 [0, 32, 64, 96] 读, 写到 [0, 1, 2, 3]

          lane_idx=8, lane_idx>>3=1:
            i=0: 读 (0^1)*32+8 = 1×32+8 = 40,     写 8*4+(0^1) = 33
            i=1: 读 (1^1)*32+8 = 0×32+8 = 8,      写 8*4+(1^1) = 32
            i=2: 读 (2^1)*32+8 = 3×32+8 = 104,    写 8*4+(2^1) = 35
            i=3: 读 (3^1)*32+8 = 2×32+8 = 72,     写 8*4+(3^1) = 34
          → lane 8 从 [40, 8, 104, 72] 读, 写到 [33, 32, 35, 34]

        读模式: 每 8 个 lane 为一个 XOR 组, lane 内读跨 4 行 × 32 列的交错数据
        写模式: 按 lane 列优先写入, 形成 32 列 × 4 行的 dense K-major 布局

        转置前 (row-major): sf[0], sf[1], sf[2], ..., sf[127]
        转置后 (K-major):   适合 UTCCP 按 128b 宽行读取的排列,
                           lane 0=col0, lane 1=col1, ..., lane 31=col31,
                           每列 4 个值组成连续的 K 维宽行

        ═══════════════════════════════════════════════════════════════════════
        TMEM 中 SF 的摆放
        ═══════════════════════════════════════════════════════════════════════

        TMEM 共 512 列 (SM100), 每列 32 bits:
        ┌─────────────────────────┬────────────────┬────────────────┐
        │     Accumulator 区      │    SFA 区       │    SFB 区      │
        │  kNumAccumTmemCols 列   │  kNumSFATmem   │  kNumSFBTmem   │
        │  (epilogue stages ×     │  Cols =        │  Cols =        │
        │   waves × BLOCK_N)      │  SF_BLOCK_M/32  │  SF_BLOCK_N/32 │
        └─────────────────────────┴────────────────┴────────────────┘

        SFA: kNumSFATmemCols = 128/32 = 4 列
        1 次 UTCCP copy 将 128 个 uint32 (4×32dp128bit) → 4 个 TMEM 列.
        循环 i = 0..ceil_div(SF_BLOCK_M, 128) 覆盖所有 M 行.

        UMMA fma 引用:
          SFA 列 = kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32)
                 = kTmemStartColOfSFA + w * 4
          每个 M wave 分配 4 个 TMEM 列, 硬件通过 sfa_id 选择列内偏移.
        ═══════════════════════════════════════════════════════════════════════
        */
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
            __syncwarp();
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto& num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // 等待 warp 0 的 TMA 完成 — SFA/SFB 数据已在 SMEM 中
                full_barriers[stage_idx]->wait(phase);

                // 只在 SF 需要加载的 K-block 做转置 (同 TMA 加载频率)
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems);
                    // 异步代理 fence: 确保 warp 2 的 SMEM 写入对 warp 1 的 UTCCP 读取可见
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

                // 转置完成, 通知 warp 1 可以开始 UTCCP + UMMA
                with_sf_full_barriers[stage_idx]->arrive(0u);
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups — STSM (TMEM→SMEM) → TMA store (SMEM→global D)
        // 128 线程 (4 warp), warp 间按 SMEM swizzle 分区, 互不重叠
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

        // TMA checks
        constexpr uint32_t kNumBankGroupBytes = 16;
        constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
        DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
        DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;
        auto advance_store_pipeline = [&]() {
            tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
        };

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            DG_STATIC_ASSERT(kNumEpilogueThreads == 128, "Epilogue threads not enough");
            DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

            // Iterate over M waves
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                // Issue every swizzled atom and pipeline STSM and TMA store
                constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
                #pragma unroll
                for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
                    // Wait shared memory to be released
                    if (epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // The pipeline stage
                    const auto m_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), IndexType::MN>(shape_m, BLOCK_M, m_block_idx) + w * WAVE_BLOCK_M;
                    const auto n_idx = epilogue_type_t::apply_index_n<STORE_BLOCK_N>(n_block_idx * BLOCK_N + s * STORE_BLOCK_N);

                    // Store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(LAYOUT_AD_M, kSwizzleCDMode / kNumBankGroupBytes)`
                        //  - new: `(LAYOUT_AD_M * kSwizzleCDMode / kNumBankGroupBytes / 8, 8)`
                        // NOTES: "8" is the number of bank groups, "16" is the swizzling pattern
                        constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                        col ^= row % (kSwizzleCDMode / 16);

                        // TMEM 地址: 基址 + M wave 偏移 + N store 偏移 + element 偏移
                        // FP8 版 TMEM accumulator 有 kNumMWaves × BLOCK_N 列 (多 wave 时)
                        uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +               // Accumulator offset
                                             w * BLOCK_N +                                          // Wave offset
                                             s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;         // In-block offset
                        // SMEM 地址: warp 偏移 + swizzle 后的 (row, col), 消 bank conflict
                        auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +        // Base pointer
                                        epilogue_warp_idx * 32 * kSwizzleCDMode +                   // Warp offset
                                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;  // In-atom offset

                        // STSM: TMEM→寄存器→SMEM, SM100 tcgen05 TMEM load 指令
                        // FP32: SM100_TMEM_LOAD_32dp32b4x (4 float/thread), BF16: 32dp32b8x (8 float→pack 4 uint32)
                        uint32_t values[kNumElemsPerBankGroup];
                        if constexpr (cute::is_same_v<cd_dtype_t, float>) {
                            // For FP32 output, read and store
                            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
                            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                                values[0], values[1], values[2], values[3]);
                            cutlass::arch::fence_view_async_tmem_load();
                            st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
                        } else {
                            // For BF16 output, read, cast and store
                            DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid type");
                            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                values[0], values[1], values[2], values[3],
                                values[4], values[5], values[6], values[7]);
                            cutlass::arch::fence_view_async_tmem_load();
                            st_shared(smem_ptr,
                                      cast_into_bf16_and_pack(values[0], values[1]),
                                      cast_into_bf16_and_pack(values[2], values[3]),
                                      cast_into_bf16_and_pack(values[4], values[5]),
                                      cast_into_bf16_and_pack(values[6], values[7]));
                        }
                    }

                    // 通知 TMEM 已空: 整个 tile 的最后一个 atom 时才 arrive, 允许 UMMA 重用该 TMEM 区
                    if (w == kNumMWaves - 1 and s == BLOCK_N / STORE_BLOCK_N - 1) {
                        tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // TMA store: SMEM → Global D (异步 DMA, 发起后立即返回)
                    // warp 0 中单线程发射, kWithAccumulation 时用 REDUCE_ADD
                    cute::tma_store_fence();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
                    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                        if constexpr (kGemmType == GemmType::Batched) {
                            using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                            cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx],
                                             n_idx, m_idx, scheduler.current_group_idx);
                        } else {
                            using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                            cute_tma_t::copy(&tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
                        }
                        cute::tma_store_arrive();
                    }
                }
            }
        }
    }

    // Deallocate tensor memory
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
