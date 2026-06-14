#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB,
          GemmType kGemmType, bool kWithAccumulation, typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_impl(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    // Stage Merge: 当 stage 数 ≥ 8 且 A/B 均为 K-major（K 连续）时，将多个原始 stage 合并，
    // TMA 一次加载更多 K 元素 → stage 数和 umma_arrive() 调用数减半。
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // 合并后至少保留 kNumMinStages 个 stage
    constexpr uint32_t kNumMinStages = 8;
    // kNumStagesPerMerge: 合并倍数（例 16 stages → 8 stages, merge 倍数为 2）
    // 未触发合并时为 1（无合并）
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    // BLOCK_K: 合并后一次 TMA 加载的 K 元素数（= BLOCK_K_ × kNumStagesPerMerge）
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    // kNumStages: 合并后的实际 stage 数
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    // ClusterTransactionBarrier 是 CUTLASS 对 CUDA PTX mbarrier 硬件原语的封装。
    // init(N)/arrive()/wait() 等方法最终编译为 mbarrier.init/arrive/try_wait 等 PTX 指令。
    // mbarrier 是 SMEM 中的一个硬件对象，用于 TMA 加载完成同步和 warp 间流水线控制。
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // GEMM with accumulation must have FP32 output
    if constexpr (kWithAccumulation)
        DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    // SM100 上 UMMA 硬件要求 A 矩阵对齐到 LAYOUT_AD_M (128) 行。
    // 当 LOAD_BLOCK_M < 128 时（如 32 或 64），TMA 加载的实际行数少于 128，但 UMMA
    // 仍以 128 行为单位访问 smem，多出的部分会溢出到 B 的 smem 区域。此断言编译期
    // 验证 A 的 smem 加 B 所有 stage 的 smem 总空间足以容纳 UMMA 的访问范围。
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory out of bound for UMMA");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SMEM 布局: [smem_cd | smem_a(num_stages) | smem_b(num_stages) | barriers | tmem_ptr]
    // 将数据区之后的 smem 原始字节 reinterpret_cast 为 Barrier* 数组，所有 barrier 都从此处分配。
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers              = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;

    // TMEM 地址存放在 shared memory 中，紧接在所有 barrier 之后。
    // warp 2 负责分配 TMEM 并将地址写入此处，epilogue warps 后续从中读取地址用于 store。
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    // init(N) 表示 barrier 需要 N 次 arrive 后才会解除阻塞。
    // ── TMA ↔ MMA 流水线 barrier ──
    // full_barriers: TMA 加载完成信号。所有 CTA 的 TMA warp 都会 arrive → init(kNumMulticast)
    // empty_barriers: MMA 消费完成信号。只有 leader CTA 的 MMA warp 会 arrive → init(1)
    // ── TMEM 流水线 barrier ──
    // tmem_full_barriers: UMMA 累加到 TMEM 完成。只有 leader CTA 的 MMA warp arrive → init(1)
    // tmem_empty_barriers: epilogue 存出完成。所有 CTA 的 epilogue threads arrive → init(kNumMulticast * kNumUMMAStoreThreads)
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                // k_idx 仅作基准值（k * BLOCK_K），实际代码中未使用。
                // k_a_idx/k_b_idx 分开计算：因为 A/B 的 major 布局可能不同，在 grouped GEMM 中
                // MN-major 的矩阵需要在 K 维上叠加 group offset，所以需要各自独立计算 K 索引。
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, kNumMulticast, batch_idx);

                // Arrive at full barriers
                // SM100 通过 distributed shared memory（shared::cluster）实现跨 CTA 访问 barrier。
                // TMA 2SM load 将数据 multicast 到两个 CTA 的 smem，但完成信号只发给 leader CTA 的 barrier。
                // ClusterTransactionBarrier 是硬件支持的分布式原语：leader 负责追踪整个 cluster 的 TX 字节数，
                // 非 leader 通过 shared::cluster 地址空间对其远程 arrive。
                // Leader: arrive_and_expect_tx 设定预期接收的 TX 字节数（* kNumMulticast 因数据写入两份 smem）
                // 非 Leader: arrive(0u) 仅参与 barrier 同步，不设置 TX 期望
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        // 创建 UMMA 指令描述符：告诉 tensor core 输入为 bf16×bf16→fp32 的矩阵乘，
        // M/N 维度和 A/B 的 major 布局。swapAB 时 A/B 的 major 互换（N 维度也随之变）。
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        // BLOCK_ATOM_K: smem swizzle 在 K 维度上的原子大小（= BLOCK_K / kNumStagesPerMerge = 原始 BLOCK_K_）。
        // 即使 stage merge 后 BLOCK_K 变大，swizzle 模式不变，UMMA descriptor 必须用原始原子大小描述布局。
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        // SM100 UMMA SmemDescriptor 中所有地址/偏移以 16 字节为单位（>>4 编码）。
        // a_desc.lo 是 stage 0 的 smem 地址（16B 单位），累加每个 stage 的大小（/16 转 16B 单位）
        // 得到对应 stage 的 descriptor。每个 lane 管一个 stage，后续用 __shfl_sync 按需取用。
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
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            // tcgen05.fence::after_thread_sync：SM100 tensor core 是深度异步流水线的，barrier.wait
            // 只做线程级同步，不管 tensor core 的 TMEM 写入是否已提交。此 fence 确保本次 iteration
            // 开始前，上一轮 tensor core 的所有 TMEM 写入已可见，防止读到脏数据。
            ptx::tcgen05_after_thread_sync();

            // UMMA and empty barrier arrival alias
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
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

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                // 任何 barrier.wait 之后如果马上要发射 UMMA 指令，都需要 fence，只是这里守卫的是内层 K 循环。
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA in the leader CTA
                // mma_t: 根据 kNumMulticast 选择 UMMA 指令类型（单 SM / 2-CTA cluster），2x1SM 会从两个 SM 的分布式 smem 读取数据
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                // 将编译期 instr_desc 转为运行时 64 位立即数，传给 UMMA fma 指令
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    // 内层 K 循环: 每次 UMMA_K=16 个 K 元素发一条 UMMA 指令
                    // BLOCK_K=64 时循环 4 次，stage merge 后 BLOCK_K=128 时循环 8 次
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                        // atom_k_idx: 当前 UMMA 步落在哪个 swizzle atom 内
                        // BULOCK_ATOM_K=64 时，k=0..3 全在 atom 0，k=4..7 在 atom 1
                        uint32_t atom_k_idx = k * UMMA_K / BLOCK_ATOM_K;
                        // advance_umma_desc_lo(base, elem_offset, k_within_atom):
                        //   参数2 = atom_k_idx * M_rows * BLOCK_ATOM_K: 跨 atom 大跳（元素数）
                        //   参数3 = k * UMMA_K % BLOCK_ATOM_K:   atom 内 K 偏移 (0,16,32,48)
                        // smem 中 A[0:LOAD_BLOCK_M, k*16:(k+1)*16] 的起始位置 → UMMA descriptor lo
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, atom_k_idx * LOAD_BLOCK_M * BLOCK_ATOM_K,
                                        k * UMMA_K % BLOCK_ATOM_K);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, atom_k_idx * LOAD_BLOCK_N * BLOCK_ATOM_K,
                                        k * UMMA_K % BLOCK_ATOM_K);
                        // 发射 UMMA fma: D += A × B
                        //   参数3 = accum_stage_idx * UMMA_N: TMEM 中累加器的列偏移
                        //   参数4 = k_block_idx>0 or k>0: 首个 K-block 第一步清零累加器，后续累加
                        if (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       k_block_idx > 0 or k > 0, runtime_instr_desc);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    __syncwarp();

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
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

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<
                (not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
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

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
