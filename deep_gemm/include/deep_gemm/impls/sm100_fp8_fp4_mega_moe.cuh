#pragma once
//
// ============================================================
// Mega MoE SM100 Kernel - 图文详解
// ============================================================
//
// 【整体架构概述】
//
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                      输入 (Input)                            │
//                    │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
//                    │  │ tokens (FP8)│  │ topk_idx     │  │ topk_weights        │  │
//                    │  │ [M, hidden] │  │ [M, num_topk]│  │ [M, num_topk]       │  │
//                    │  └─────────────┘  └──────────────┘  └─────────────────────┘  │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                   EP Dispatch (跨GPU通信)                      │
//                    │                                                                    │
//                    │   GPU 0 ──NVLink──> GPU 1 ──NVLink──> GPU 2 ──NVLink──> GPU 3   │
//                    │      │                  │                  │                   │
//                    │      ▼                  ▼                  ▼                   │
//                    │   路由到正确          路由到正确          路由到正确              │
//                    │   的expert           的expert           的expert               │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                      L1 GEMM (FP8xFP4)                       │
//                    │                                                                    │
//                    │      tokens(FP8) @ weights_L1(FP4) ──> intermediate(FP8)        │
//                    │                                                                    │
//                    │      shape: [M, hidden] x [hidden, int_hidden*2]                │
//                    │               = [M, int_hidden*2]                              │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                      SwiGLU Activation                       │
//                    │                                                                    │
//                    │      gate, up = split(intermediate, 2)                        │
//                    │      output = SiLU(gate) * up                                   │
//                    │      (或 SwiGLU: gate * SiLU(up))                              │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                      L2 GEMM (FP8xFP4)                       │
//                    │                                                                    │
//                    │      intermediate(FP8) @ weights_L2(FP4) ──> output(BF16)     │
//                    │                                                                    │
//                    │      shape: [M, int_hidden] x [int_hidden, hidden]              │
//                    │               = [M, hidden]                                    │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                   EP Combine (跨GPU通信)                     │
//                    │                                                                    │
//                    │      每个token的topk结果需要汇聚回原始位置                      │
//                    │      GPU 0 <──NVLink── GPU 1 <──NVLink── GPU 2 <──NVLink── GPU 3 │
//                    └──────────────────────────────────────────────────────────────┘
//                                       │
//                    ┌──────────────────────────────────────────────────────────────┐
//                    │                      输出 (Output)                            │
//                    │                   y: [M, hidden] (BF16)                       │
//                    └──────────────────────────────────────────────────────────────┘
//
// 【线程分工】
//
//   ┌─────────────────────────────────────────────────────────────────────────────┐
//   │  Warp 0        │ Warp 1        │ Warp 2        │ Warp 3        │ ...     │
//   │  Dispatch #0   │ Dispatch #1   │ MMA Load A    │ MMA Load B    │ MMA     │
//   │  (EP通信)      │ (EP通信)      │ (tokens+SFA)  │ (weights+SFB) │ Issue   │
//   └─────────────────────────────────────────────────────────────────────────────┘
//   │                          │                          │                       │
//   │                          │                          │                       │
//   ▼                          ▼                          ▼                       ▼
//   1. 统计每个expert       2. TMA从远端GPU        3. TMA加载tokens        4. 执行UMMA
//      的token数量             拉取数据               和scaling factors       FMA指令
//
// 【共享内存布局】
//
//   smem_buffer:
//   ┌────────────────┬────────────────┬─────────────────────────────────────────┐
//   │ Expert Count   │ Send Buffer    │  GEMM Shared Memory                     │
//   │ [num_experts]  │ [dispatch_warps│  ┌──────────┬──────────┬──────────┐     │
//   │                │ x token_size]  │  │  C/D     │    A    │    B    │     │
//   │                │                │  │ (epilogue)│ (stages)│ (stages)│     │
//   │                │                │  └──────────┴──────────┴──────────┘     │
//   │                │                │  ┌──────────┬──────────┐                │
//   │                │                │  │   SFA   │   SFB   │ (scaling factors)│
//   │                │                │  └──────────┴──────────┘                │
//   └────────────────┴────────────────┴─────────────────────────────────────────┘
//
// 【Tensor Memory布局】 (SM100特有的on-chip memory)
//
//   TMEM columns:
//   ┌────────────────────────────────────────────────────────────────────────────┐
//   │ Accumulator (C)           │ SFA (Scaling Factor A)  │ SFB (Scaling Factor B) │
//   │ [UMMA_N * epilogue_stages]│ [SF_BLOCK_M / 32]       │ [SF_BLOCK_N / 32]       │
//   └────────────────────────────────────────────────────────────────────────────┘
//
// 【数据格式】
//
//   FP8 (e4m3): 用于activation，默认范围 [-448, 448]
//   FP4 (e2m1): 用于weights，默认值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//   UE8M0:     Packed FP32格式，4个值打包成1个uint32
//
// ============================================================

#include <cstdint>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <
    uint32_t kNumMaxTokensPerRank,       // 每个rank的最大token数量，用于预分配共享内存
    uint32_t kHidden,                    // 隐藏层维度 (hidden size)
    uint32_t kIntermediateHidden,        // FFN中间层维度 (通常是 hidden * 4 / 3 或类似值)
    uint32_t kNumExperts,                // MoE模型中专家的总数
    uint32_t kNumTopk,                   // 每个token选择的top-k个专家
    uint32_t kNumExpertsPerWave,         // 每wave处理的专家数量
    uint32_t BLOCK_M,                    // GEMM tile的M维度
    uint32_t BLOCK_N,                    // GEMM tile的N维度
    uint32_t BLOCK_K,                    // GEMM tile的K维度
    uint32_t STORE_BLOCK_M,             // MMA epilogue的store block M维度
    uint32_t SF_BLOCK_M,                // Scaling Factor块的M维度
    uint32_t SF_BLOCK_N,                // Scaling Factor块的N维度
    uint32_t kNumMaxPoolTokens,         // 池化token的最大数量
    uint32_t kNumPaddedSFPoolTokens,    // Padding后的SF池化token数量
    uint32_t kNumStages,                // TMA加载的pipeline stages数量
    uint32_t kNumDispatchThreads,       // 负责EP dispatch的线程数
    uint32_t kNumNonEpilogueThreads,    // 负责MMA非epilogue计算的线程数 (固定为128)
    uint32_t kNumEpilogueThreads,       // 负责epilogue和combine的线程数
    uint32_t kNumSMs,                   // 使用的SM数量
    uint32_t kNumRanks,                 // 并行ranks数量 (通常为GPU数量)
    float kActivationClamp,             // SwiGLU激活函数的clamp值
    bool kFastMath,                     // 是否启用fast math优化
    uint32_t L1_SHAPE_N = kIntermediateHidden * 2,  // L1 GEMM的N维度 (gate+up)
    uint32_t L1_SHAPE_K = kHidden,                 // L1 GEMM的K维度
    uint32_t L2_SHAPE_N = kHidden,                 // L2 GEMM的N维度
    uint32_t L2_SHAPE_K = kIntermediateHidden,     // L2 GEMM的K维度
    uint32_t kNumDispatchWarps = kNumDispatchThreads / 32,
    uint32_t kNumMMANonEpilogueWarps = kNumNonEpilogueThreads / 32,
    uint32_t kNumEpilogueWarps = kNumEpilogueThreads / 32,
    uint32_t kNumEpilogueWarpgroups = kNumEpilogueWarps / 4,
    uint32_t kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads,
    uint32_t kNumTokensPerWarp = 32 / kNumTopk,
    uint32_t kNumExpertsPerRank = kNumExperts / kNumRanks
>
CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void
sm100_fp8_fp4_mega_moe_impl(void* y,                                                          // 输出tensor (bfloat16, shape: [num_tokens, kHidden])
                            int* cumulative_local_expert_recv_stats,                           // 每个本地专家累计接收的token数
                            const uint32_t num_tokens,                                        // 当前batch的token数量
                            const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,   // 对称内存缓冲区 (用于EP通信)
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts,       // L1输入激活的TMA descriptor (FP8)
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_acts_sf,  // L1输入激活scaling factor的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights,   // L1权重(FP4)的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_weights_sf,// L1权重scaling factor的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l1_output,    // L1输出(FP8)的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts,       // L2输入激活的TMA descriptor (FP8)
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_acts_sf,  // L2输入激活scaling factor的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights,   // L2权重(FP4)的TMA descriptor
                            const __grid_constant__ cute::TmaDescriptor tensor_map_l2_weights_sf) {// L2权重scaling factor的TMA descriptor
//#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::TMEM::Allocator2Sm;

    // Template checks
    DG_STATIC_ASSERT(kNumDispatchThreads % 128 == 0, "Invalid number of dispatch threads");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Invalid number of MMA non-epilogue threads");
    DG_STATIC_ASSERT(kNumEpilogueThreads % 128 == 0, "Invalid number of MMA epilogue and combine threads");
    DG_STATIC_ASSERT(kNumExperts % kNumRanks == 0, "Invalid number of experts or ranks");

    // Thread indices
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t sm_idx = blockIdx.x;
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l1_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l1_weights_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l1_output);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts);
        cute::prefetch_tma_descriptor(&tensor_map_l2_acts_sf);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights);
        cute::prefetch_tma_descriptor(&tensor_map_l2_weights_sf);
    }

    // Workspace: 管理 EP 通信的元数据内存布局，从 sym_buffer 基地址开始的连续内存区域：
    //   [0..31]      Barrier 信号区 (32B)
    //                  [0..15]  4 × uint32_t grid sync 计数器
    //                  [16..20] uint32_t NVLink barrier 计数器
    //                  [20..27] 2 × int NVLink barrier 信号 (phase 0/1)
    //   [32..]       Expert 发送计数 (num_experts × uint64_t)
    //   [..]         Expert 接收计数 (num_ranks × num_experts_per_rank × uint64_t)
    //   [..]         Expert 接收计数求和 (num_experts_per_rank × uint64_t)
    //   [..]         L1 到达计数 (align(num_max_pool_blocks, 2) × uint32_t)
    //   [..]         L2 块到达掩码 (num_max_pool_blocks × uint64_t)
    //   [..]         Dispatch 拉取源 token-topk 索引 (num_experts_per_rank × num_ranks × num_max_recv_tokens × int)
    //   [..]         Combine 推送源元数据 (num_max_pool_tokens × TokenSrcMetadata)
    //   末尾 16B 对齐 (TMA descriptor 要求)
    const auto workspace = layout::Workspace(
        sym_buffer.get_base_ptr(), kNumRanks, kNumExperts, kNumMaxTokensPerRank, kNumTopk);

    // Token and buffer layouts (layout::Data: per-token 字节数, 是否TMA对齐)
    constexpr auto fp8_token_layout = layout::Data(kHidden);                           // FP8 token: kHidden 字节/token, TMA对齐
    constexpr auto bf16_token_layout = layout::Data(kHidden * sizeof(nv_bfloat16));    // BF16 token: kHidden×2 字节/token, TMA对齐
    constexpr auto fp8_intermediate_token_layout = layout::Data(kIntermediateHidden);  // FP8中间层token: kIntermediateHidden 字节/token, TMA对齐
    constexpr auto fp8_sf_layout = layout::Data(kHidden / 32);                         // FP8 scaling factor: 每32个元素1个SF, TMA对齐
    constexpr auto fp8_intermediate_sf_layout = layout::Data(kIntermediateHidden / 32); // FP8中间层SF, TMA对齐
    constexpr auto input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false); // 输入topk索引: kNumTopk个int64, 非TMA对齐
    constexpr auto input_topk_weights_layout = layout::Data(kNumTopk * sizeof(float), false); // 输入topk权重: kNumTopk个float, 非TMA对齐
    constexpr auto l1_topk_weights_layout = layout::Data(sizeof(float), false);            // L1 topk权重: 单个float, 非TMA对齐

    // Registered inputs (layout::Buffer: Data布局, rank数, 每rank最大token数, 基地址)
    // 各buffer在workspace末尾依次紧挨排列
    const auto input_token_buffer = layout::Buffer(                       // 输入token缓冲区 (FP8, 单rank)
        fp8_token_layout, 1, kNumMaxTokensPerRank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(                         // 输入scaling factor缓冲区 (FP8 SF, 单rank)
        fp8_sf_layout, 1, kNumMaxTokensPerRank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(                   // 输入topk索引缓冲区 (int64 × kNumTopk, 单rank)
        input_topk_idx_layout, 1, kNumMaxTokensPerRank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(              // 输入topk权重缓冲区 (float × kNumTopk, 单rank)
        input_topk_weights_layout, 1, kNumMaxTokensPerRank,
        input_topk_idx_buffer.get_end_ptr());

    // SF and its buffer configs
    constexpr uint32_t kGranK = 32;
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    DG_STATIC_ASSERT(SF_BLOCK_M == math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems), "Invalid SF_BLOCK_M");
    DG_STATIC_ASSERT(SF_BLOCK_N == BLOCK_N, "No padding is needed for SFB");

    // UTCCP 4x32 transpose index mapping within each 128-element group
    const auto transform_sf_token_idx = [](const uint32_t& token_idx_in_expert) {
        const uint32_t idx = token_idx_in_expert % BLOCK_M;
        return token_idx_in_expert / BLOCK_M * SF_BLOCK_M +
               (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
    };

    // L1 inputs
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, kNumMaxPoolTokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, kNumPaddedSFPoolTokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, kNumMaxPoolTokens,
        l1_sf_buffer.get_end_ptr());

    // L2 inputs
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, kNumMaxPoolTokens,
        l1_topk_weights_buffer.get_end_ptr()
    );
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, kNumPaddedSFPoolTokens,
        l2_token_buffer.get_end_ptr()
    );

    // Combine inputs
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, kNumTopk, kNumMaxTokensPerRank,
        l2_sf_buffer.get_end_ptr()
    );

    // Data types
    // NOTES: activations are FP8 (e4m3), weights are FP4 (e2m1)
    using a_dtype_t = cutlass::float_e4m3_t;
    using b_dtype_t = cutlass::detail::float_e2m1_unpacksmem_t;


    // MMA configs
    // NOTES: always swap A/B, 2-CTA MMA, and matrices are K-major
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;
    constexpr uint32_t UMMA_N = BLOCK_M;  // Swap AB
    constexpr uint32_t UMMA_K = 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2;  // Multicast on A
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
    DG_STATIC_ASSERT(BLOCK_M % 16 == 0, "Invalid block M");
    DG_STATIC_ASSERT(BLOCK_N == LAYOUT_AD_M, "Invalid block N");
    DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");

    // Swizzle configs
    constexpr uint32_t kSwizzleAMode = BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t kSwizzleBMode = BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t kSwizzleCDMode = 128;
    DG_STATIC_ASSERT(BLOCK_N % kSwizzleCDMode == 0, "Invalid block N");

    // Epilogue configs
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;

    // Shared memory
    constexpr uint32_t kSharedMemoryAlignment = 1024;
    extern __shared__ __align__(kSharedMemoryAlignment) uint8_t smem_buffer[];

    // Shared memory sizes
    // NOTES: FP8 CD output for L1 (2 TMA stages, BLOCK_N/2 post-SwiGLU), BF16 output for L2 (no TMA, a single stage)
    constexpr uint32_t L1_OUT_BLOCK_N = BLOCK_N / 2;
    constexpr uint32_t SMEM_EXPERT_COUNT_SIZE =
        math::constexpr_align<uint32_t>(kNumExperts * sizeof(uint32_t), kSharedMemoryAlignment);
    constexpr uint32_t SMEM_SEND_BUFFER_SIZE =
        math::constexpr_align(fp8_token_layout.get_num_bytes() * kNumDispatchWarps, kSharedMemoryAlignment);
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
    constexpr uint32_t SMEM_CD_L1_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(cutlass::float_e4m3_t) * kNumTMAStoreStages;
    constexpr uint32_t SMEM_CD_L2_SIZE =
        kNumEpilogueWarpgroups * STORE_BLOCK_M * BLOCK_N * sizeof(nv_bfloat16);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_L1_SIZE > SMEM_CD_L2_SIZE ? SMEM_CD_L1_SIZE : SMEM_CD_L2_SIZE;
    constexpr uint32_t SMEM_CD_L1_SIZE_PER_STAGE = SMEM_CD_L1_SIZE / kNumTMAStoreStages;
    constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
        SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % kSharedMemoryAlignment == 0 and
                     SMEM_A_SIZE_PER_STAGE % kSharedMemoryAlignment == 0 and
                     SMEM_B_SIZE_PER_STAGE % kSharedMemoryAlignment == 0,
                     "Shared memory of CD/A/B must be aligned to 1024 bytes");

    // Tensor memory size
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Assign shared memory for dispatch warps
    const auto smem_expert_count = reinterpret_cast<uint32_t*>(smem_buffer);
    const auto smem_send_buffers = layout::Buffer(
        fp8_token_layout, kNumDispatchWarps, 1,
        math::advance_ptr(smem_buffer, SMEM_EXPERT_COUNT_SIZE));

    // GEMM shared memory: C/D, A, B
    // NOTES: GEMM shared memory starts after the dispatch region, aligned to 1024 bytes
    auto smem_gemm_base = math::advance_ptr(
        smem_buffer, SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE
    );

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<uint8_t>(smem_gemm_base, i * SMEM_CD_L1_SIZE_PER_STAGE);
    });
    auto smem_cd_l2 = smem_cd[0];
    auto smem_a = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<a_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([=](const uint32_t& i) {
        return math::advance_ptr<b_dtype_t>(smem_gemm_base, SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // SF shared memory: SFA and SFB per pipeline stage
    auto sf_start_ptr = math::advance_ptr<uint8_t>(smem_gemm_base,
        SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto smem_sfa = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + i * SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(sf_start_ptr + kNumStages * SMEM_SFA_SIZE_PER_STAGE + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // Epilogue amax reduction shared memory
    auto smem_amax_reduction = reinterpret_cast<float2*>(smem_sfb[kNumStages]);

    // Barriers and tensor memory pointer
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_amax_reduction + STORE_BLOCK_M * kNumEpilogueWarps / 2);
    auto dispatch_barriers      = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto full_barriers          = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + i); });
    auto empty_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages + i); });
    auto tmem_full_barriers     = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + i); });
    auto tmem_empty_barriers    = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages + i); });
    auto combine_barriers       = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + i); });
    auto tmem_ptr_in_smem       = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumDispatchWarps + kNumStages * 2 + kNumEpilogueStages * 2 + kNumEpilogueWarps * 2);

    // A cluster sync is essential for 2CTA tensor memory allocation
    comm::cluster_sync_with_relaxed_arrive();

    // Initialization
    if (warp_idx == 0) {
        // Clean shared memory
        if (cute::elect_one_sync())
            ptx::st_shared_bulk(smem_expert_count, kNumExperts * sizeof(uint32_t));
    } else if (warp_idx == 1) {
        // Init m-barriers for dispatch
        #pragma unroll
        for (uint32_t i = lane_idx; i < kNumDispatchWarps; i += 32)
            dispatch_barriers[i]->init(1);
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Init GEMM barriers
        if (cute::elect_one_sync()) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumStages; ++ i) {
                // Arrive at all CTAs
                full_barriers[i]->init(2 * 2);
                empty_barriers[i]->init(1);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
                // Arrive at all CTAs
                tmem_full_barriers[i]->init(1);
                // Arrive only at the leader CTA
                tmem_empty_barriers[i]->init(2 * kNumEpilogueThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumEpilogueWarps * 2; ++ i)
                combine_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 3) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    // NOTES: Using `.relaxed` is allowed here since `fence_barrier_init` is `.release.cluster`,
    // and `barrier.cluster.wait.aligned` is by default `.acquire`
    comm::cluster_sync_with_relaxed_arrive();

    // Task scheduler
    auto scheduler = sched::MegaMoEScheduler<
        BLOCK_M, BLOCK_N, BLOCK_K,
        L1_SHAPE_N, L1_SHAPE_K,
        L2_SHAPE_N, L2_SHAPE_K,
        kNumExpertsPerRank,
        kNumExpertsPerWave,
        kNumSMs, kNumRanks>(workspace);

    // MMA pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Intra-SM Barrier indices
    constexpr uint32_t kDispatchBarrierIdx = 0;
    constexpr uint32_t kDispatchWithEpilogueBarrierIdx = 1;
    constexpr uint32_t kEpilogueFullBarrierIdx = 2;
    constexpr uint32_t kEpilogueWGBarrierStartIdx = 3;

    // NVLink barrier tags
    constexpr uint32_t kBeforeDispatchPullBarrierTag = 1;
    constexpr uint32_t kBeforeCombineReduceBarrierTag = 2;
    constexpr uint32_t kAfterWorkspaceCleanBarrierTag = 3;

    // Adjust registers
    constexpr uint32_t kNumDispatchRegisters = 48;
    constexpr uint32_t kNumNonEpilogueRegisters = 40;
    constexpr uint32_t kNumEpilogueRegisters = 208;
    DG_STATIC_ASSERT(kNumDispatchRegisters * kNumDispatchThreads +
                     kNumNonEpilogueRegisters * kNumNonEpilogueThreads +
                     kNumEpilogueRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // Grid sync index assignments (dispatch and epilogue use separate counters to avoid conflicts)
    constexpr uint32_t kDispatchGridSyncIndex = 0;
    constexpr uint32_t kEpilogueGridSyncIndex = 1;

    // ============================================================
    // 【线程角色分配】 - 每个warp有明确的职责
    //
    //  ┌─────────────────────────────────────────────────────────────────────────────┐
    //  │ Warp 0 ~ kNumDispatchWarps-1    : EP Dispatch Warps (跨GPU通信)              │
    //  │   - 统计每个expert的token数量                                               │
    //  │   - 通过NVLink从远端GPU拉取数据                                            │
    //  │   - 写入本地L1 buffer                                                       │
    //  ├─────────────────────────────────────────────────────────────────────────────┤
    //  │ Warp kNumDispatchWarps       : MMA Load Warp A (tokens + SFA)               │
    //  │   - 通过TMA从全局内存加载tokens                                             │
    //  │   - 加载对应的scaling factors                                               │
    //  ├─────────────────────────────────────────────────────────────────────────────┤
    //  │ Warp kNumDispatchWarps + 1   : MMA Load Warp B (weights + SFB)             │
    //  │   - 通过TMA从全局内存加载weights                                            │
    //  │   - 加载对应的scaling factors                                               │
    //  ├─────────────────────────────────────────────────────────────────────────────┤
    //  │ Warp kNumDispatchWarps + 2   : MMA Issue Warp                              │
    //  │   - 执行UMMA FMA指令                                                        │
    //  │   - 仅leader CTA运行                                                        │
    //  ├─────────────────────────────────────────────────────────────────────────────┤
    //  │ Warp kNumDispatchWarps + 3   : 空warp (预留)                                │
    //  ├─────────────────────────────────────────────────────────────────────────────┤
    //  │ Warp >= kNumDispatchWarps + kNumMMANonEpilogueWarps : Epilogue Warps       │
    //  │   - SwiGLU激活函数                                                          │
    //  │   - 结果写回全局内存                                                        │
    //  │   - Combine操作 (结果汇聚)                                                 │
    //  └─────────────────────────────────────────────────────────────────────────────┘
    //
    // Different warp roles
    if (warp_idx < kNumDispatchWarps) {
        // ============================================================
        // 【Dispatch Warp职责】
        //
        // 1. 统计阶段: 遍历所有token-topk对，统计每个expert被选中的次数
        //    ┌─────────────────────────────────────────────┐
        //    │ token[0] → expert[5] ──> expert_count[5]++  │
        //    │ token[0] → expert[2] ──> expert_count[2]++  │
        //    │ token[1] → expert[5] ──> expert_count[5]++  │
        //    │ token[1] → expert[8] ──> expert_count[8]++  │
        //    │ ...                                        │
        //    └─────────────────────────────────────────────┘
        //
        // 2. 广播阶段: 通过atomic将统计结果广播到workspace
        //
        // 3. 路由阶段: 计算每个token应该发往哪个rank的哪个expert
        //
        // 4. 拉取阶段: 通过NVLink从远端GPU拉取token数据和scaling factors
        //
        //    ┌────────────────────────────────────────────────────────────┐
        //    │ GPU 0                                                      │
        //    │   need tokens[0..3] for expert[2] from GPU 1              │
        //    │        ◄──────── NVLink pull ──────── GPU 1              │
        //    │                                                        │
        //    │   need tokens[4..7] for expert[6] from GPU 2            │
        //    │        ◄──────── NVLink pull ──────── GPU 2              │
        //    └────────────────────────────────────────────────────────────┘
        //
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumDispatchRegisters>();

        // Dispatch warps
        DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
        // kNumTokensPerWarp = 32 / kNumTopk
        constexpr uint32_t kNumActivateLanes = kNumTokensPerWarp * kNumTopk; // kNumActivateLanes = 32
        const auto read_topk_idx = [&](const auto& process) {
            // TODO: figure out better unrolling
            // Now, `unroll` is better than `unroll 8`
            #pragma unroll
            for (uint32_t i = (sm_idx * kNumDispatchWarps + warp_idx) * kNumTokensPerWarp;
                 i < num_tokens;
                 i += kNumSMs * kNumDispatchWarps * kNumTokensPerWarp) {
                // Allocate slots for each token-topk
                int expert_idx = -1;
                // ⚠️ 一个warp一次处理多个token的topk
                if (i + (lane_idx / kNumTopk) < num_tokens and lane_idx < kNumActivateLanes) {
                    expert_idx = static_cast<int>(
                        __ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + i * kNumTopk + lane_idx));
                    if (expert_idx >= 0)
                        process(i * kNumTopk + lane_idx, expert_idx);
                }
                __syncwarp();
            }
        };

        // Count experts' tokens
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
           atomicAdd_block(smem_expert_count + expert_idx, 1);
        });
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Get SM offset (~6.5 us)
        //
        // 目的：将本SM在每个expert上的token计数累加到全局workspace，同时取回本SM在该expert
        // 全局缓冲区中的起始槽位偏移（offset），供后续写入源索引时使用。
        //
        // 关键技巧：用一次 uint64_t 原子加同时完成两件事——
        //   send_value = (1ull << 32) | smem_expert_count[i]
        //     高32位 = 1：表示"本SM参与了对该expert的发送"（用于后续统计有多少SM参与了发送）
        //     低32位 = 本SM要发给expert i的token数量（由上一步Count experts' tokens统计得到）
        //
        // ptx::atomic_add 返回加操作之前的旧值，即：
        //   旧值低32位 = 在本SM之前，其他SM已经累加的token总数 = 本SM的起始槽位偏移
        //   （旧值高32位暂不使用，后续在Write expert count阶段才用到SM计数）
        //
        // 举例：3个SM都给expert 0发token，假设写入顺序为SM0→SM1→SM2
        //   SM0 发5个 → atomic_add返回0, smem_expert_count[0] = 0 (槽位0~4)
        //   SM1 发3个 → atomic_add返回5, smem_expert_count[0] = 5 (槽位5~7)
        //   SM2 发4个 → atomic_add返回8, smem_expert_count[0] = 8 (槽位8~11)
        //
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            // ⚠️  高位 + 1
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Write source indices (~2 us with 512 tokens)
        //
        // 目的：将每个token-topk的源索引写入远端rank的workspace，让远端rank在后续
        // 拉取阶段(pull phase)知道：需要从哪个rank、取哪个token的数据。
        //
        // 写入位置：workspace中每个(本地expert, 来源rank)组合有一段连续的索引缓冲区，
        // 槽位号由本SM的起始偏移递增分配。
        //
        // 具体步骤（对每个token-topk）：
        //   1. dst_rank_idx: 确定目标expert属于哪个rank（expert_idx / kNumExpertsPerRank）
        //   2. dst_slot_idx: atomicAdd_block递增smem_expert_count，从步骤Get SM offset
        //      拿到的起始偏移开始依次分配，例如起始偏移为5则第一个token分到槽位5，下一个6...
        //   3. dst_ptr: 计算远端workspace中的写入地址
        //      - expert_idx % kNumExpertsPerRank: 目标rank的本地expert索引
        //      - sym_buffer.rank_idx: 来源rank（即本GPU）
        //      - dst_slot_idx: 在该(本地expert, 来源rank)段中的槽位号
        //   4. sym_buffer.map(dst_ptr, dst_rank_idx): 将本地workspace地址映射为目标rank的
        //      NVLink远端地址，然后直接写入token_topk_idx（= token_idx * kNumTopk + topk_idx，
        //      表示"哪个token的第几个topk选择"）
        //
        read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
            const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;      // 目标expert所在的rank
            const auto dst_slot_idx = atomicAdd_block(smem_expert_count + expert_idx, 1); // 从起始偏移递增分配槽位
            // Dispatch拉取源索引区域: workspace中 num_experts_per_rank × num_ranks × num_max_recv_tokens_per_expert 的int数组
            // 按三维索引 [local_expert_idx][src_rank_idx][slot_idx] 寻址
            // 存储的是token-topk复合索引，远端rank据此知道从哪个rank拉取哪个token
            const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(
                expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
            *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;       // NVLink直接写入远端rank
        });

        // Grid sync: 所有SM必须完成源索引写入后，各rank才能安全地读取自己workspace中的
        // 索引缓冲区进入拉取阶段。grid_sync使用workspace中的grid sync计数器实现跨SM同步，
        // lambda提供CTA内部的barrier同步。
        comm::grid_sync<kNumSMs, kDispatchGridSyncIndex>(
            workspace, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); }
        );

        // Write expert count
        if (sm_idx == 0) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const auto dst_rank_idx = i / kNumExpertsPerRank;
                const auto dst_local_expert_idx = i % kNumExpertsPerRank;
                const auto expert_status = *workspace.get_expert_send_count_ptr(i);
                // ⚠️ get_expert_recv_count_ptr(j, current_expert_idx) 记录的既是 rank 又是 expert——它是一个二维表，
                // 语义是 "第 j 个 rank 发给当前本地第 current_expert_idx 个 expert 的 token 数"。
                /*
                Workspace 内存布局（连续排列）：
                ┌────────────────────────────────────────────────────────────┐
                │ expert_send_count[0..num_experts-1]                        │
                │   含义：全局第 i 个 expert 从本 rank 发出去了多少 token       │
                │   维度：num_experts × uint64                                │
                │   写入：dispatch 阶段本 rank 统计后写入                      │
                ├────────────────────────────────────────────────────────────┤
                │ expert_recv_count[0..num_ranks-1][0..num_experts_per_rank-1]│
                │   含义：第 j 个 rank 发给本 rank 第 e 个本地 expert 的 token数│
                │   维度：num_ranks × num_experts_per_rank × uint64           │
                │   写入：其他 rank 通过 NVLink 远端写入（mapped address）      │
                ├────────────────────────────────────────────────────────────┤
                │ expert_recv_count_sum[0..num_experts_per_rank-1]           │
                │   含义：本 rank 第 e 个本地 expert 从所有 rank 接收的 token 总数│
                │   维度：num_experts_per_rank × uint64                       │
                │   写入：其他 rank 通过 atomic_add 远端累加                    │
                └────────────────────────────────────────────────────────────┘
                */
                
                // ⚠️ All 2 All --> 得到每个rank发来的expert count数据 
                *sym_buffer.map(
                    workspace.get_expert_recv_count_ptr(sym_buffer.rank_idx, dst_local_expert_idx),
                    dst_rank_idx) = expert_status & 0xffffffff;
                // ⚠️ Reduce-Scatter，相当于对上面A2A的结果按rank累加，得到expert recv count sum
                ptx::atomic_add_sys(
                    sym_buffer.map(workspace.get_expert_recv_count_sum_ptr(dst_local_expert_idx), dst_rank_idx),
                    expert_status);
            }
        }
        ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

        // Barrier before pulling
        //
        // 目的：跨所有rank的NVLink屏障同步，确保所有rank都完成了expert count写入后，
        // 才开始拉取(pull)阶段。否则某rank可能读到其他rank尚未写入完毕的expert计数和源索引。
        // 执行流程（3步）：
        //   1. [sync_prologue=false] 跳过grid sync — 因为上方的barrier已经保证了CTA内部同步，
        //      且Write expert count阶段只有SM0参与写入，其他SM不会产生写冲突
        //   2. [NVLink barrier] 只有SM0的线程参与跨rank信令：
        //      - 每个线程向对应远端rank发送原子递增信号 (ptx::red_add_rel_sys)
        //      - 然后等待所有rank的信号到达本地 (自旋等待 signal_ptr == kNumRanks)
        //      - 使用phase交替机制避免重复使用同一个信号值
        //   3. [sync_epilogue=true] NVLink barrier后执行grid sync — 确保SM0完成
        //      跨rank同步后，再通知其他SM可以安全进入拉取阶段
        //
        // 模板参数：
        //   kDispatchGridSyncIndex - grid sync使用的计数器索引
        //   kBeforeDispatchPullBarrierTag - 屏障标签，区分不同阶段的NVLink barrier
        //
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kBeforeDispatchPullBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* After the grid sync above, there is no more writes by other SMs (except 0) */ false,
            /* After the NVLink barrier, there is a grid sync */ true
        );

        // 4.5.2 与 Dispatch Warp 同步
        //
        // 两次同步, 解决 SMEM 和 workspace 所有权切换:
        //
        // 第 1 次 (此处):
        //   dispatch 的 smem_send_buffers (pull TMA 1D 落地区)
        //   与 epilogue combine 阶段的 combine_load/combine_store 复用 smem
        //   同步后: dispatch pull 阶段和 epilogue 阶段互不干扰
        //
        // 第 2 次 (Dispatch 清理 workspace 前):
        //   epilogue combine 需要复用 smem 做 chunk 缓冲
        //   dispatch 需要等 combine 启动后才能安全清理 workspace
        //
        // 使用 sync_unaligned: dispatch (128 线程) 和 epilogue (256 线程)
        //   线程数不同且 warp 边界不对齐, 只能用 bar.sync unaligned 变体
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // Pull token data and SF from remote ranks into local L1 buffer
        uint32_t pull_mbarrier_phase = 0;
        // ⚠️ 一个warp一个smem，配一个mbarrier
        const auto pull_buffer = smem_send_buffers.get_rank_buffer(warp_idx).get_data_buffer(0);
        const auto pull_mbarrier = dispatch_barriers[warp_idx];

        // ⚠️从全局workspace缓存每个本地expert的token接收总数到warp级寄存器
        // fetch_expert_recv_count() 内部流程：
        //   1. 每个lane负责缓存 expert (i*32 + lane_idx) 的计数
        //   2. 自旋等待 workspace.expert_recv_count_sum 高32位 == kNumSMs * kNumRanks
        //      （确保所有rank的计数都已汇总完毕）
        //   3. 取低32位 = 该expert从所有rank接收到的token总数
        //   4. 存入 stored_num_tokens_per_expert[]，后续 get_num_tokens() 直接查寄存器
        // NVLink barrier已保证此处数据就绪
        scheduler.fetch_expert_recv_count();

        // Per-rank counts for current expert (re-loaded when expert changes)
        // kNumRanksPerLane: 每个lane需要缓存的rank数（向上取整到32的倍数）
        constexpr uint32_t kNumRanksPerLane = math::constexpr_ceil_div(kNumRanks, 32u);
        int current_expert_idx = -1;
        // ⚠️ per-lane级存储，所有rank发送到本rank的当前这个expert的token count
        uint32_t stored_rank_count[kNumRanksPerLane] = {}; 
        uint32_t expert_start_idx = 0, expert_end_idx = 0;  // 当前expert在全局token索引空间中的区间 [start, end)
        uint32_t expert_pool_block_offset = 0;              // 当前expert在L1 token pool中的BLOCK_M块偏移

        // 所有dispatch warp跨SM轮询拉取token，全局token索引空间按expert紧凑排列
        // 例: expert0收到10个token → 区间[0,10), expert1收到5个 → [10,15), expert2收到8个 → [15,23)
        // ⚠️ 按本rank的expert顺序排序token，token_idx是顺序索引
        // ⚠️ 已有每个expert的recv count
        constexpr uint32_t kNumGlobalWarps = kNumSMs * kNumDispatchWarps;

        for (uint32_t token_idx = sm_idx * kNumDispatchWarps + warp_idx; ; token_idx += kNumGlobalWarps) {
            // 在全局token索引空间中推进expert，直到找到token_idx所属的expert区间
            // 所有本地expert的token按顺序紧凑排列，while循环递推 [expert_start_idx, expert_end_idx)
            int old_expert_idx = current_expert_idx;
            while (token_idx >= expert_end_idx) {
                if (++ current_expert_idx >= kNumExpertsPerRank) 
                    break;

                // 更新当前expert在pool中的块偏移（上一个expert的token数按BLOCK_M向上取整）
                // ⚠️ math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M) 表示当前expert会被分为几个BLOCK_M块
                // ⚠️ expert_pool_block_offset 表示这个expert的最后一块block的全局偏移
                expert_pool_block_offset += math::ceil_div(expert_end_idx - expert_start_idx, BLOCK_M);
  
                expert_start_idx = expert_end_idx;
                // ⚠️ 获取当前expert的recv count sum
                expert_end_idx += scheduler.get_num_tokens(current_expert_idx);
            }
 
            // Finish all tokens
            if (current_expert_idx >= kNumExpertsPerRank)
                break;

            // Load per-rank counts when expert changes
            // ⚠️expert切换时，从workspace加载该expert各rank的接收计数到寄存器
            // ⚠️ per-lane级存储，所有rank发送到本rank的当前这个expert的token count
            if (old_expert_idx != current_expert_idx) {
                old_expert_idx = current_expert_idx; 
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    const uint32_t j = i * 32 + lane_idx;
                    // TODO: this is not coalesced
                    // ⚠️ get_expert_recv_count_ptr(j, current_expert_idx) 记录的既是 rank 又是 expert——它是一个二维表，
                    // 语义是 "第 j 个 rank 发给当前本地第 current_expert_idx 个 expert 的 token 数"。
                    stored_rank_count[i] = j < kNumRanks ?
                        static_cast<uint32_t>(*workspace.get_expert_recv_count_ptr(j, current_expert_idx)) : 0;
                }
            }

            // ========================================================================
            // ⚠️ Round-robin rank selection via iterative min-peeling
            // ========================================================================
            // 输出：当前 token 所属的 rank 索引
            uint32_t current_rank_in_expert_idx;

            // remaining[]: 每 lane 负责若干 rank，记录每个 rank 当前还有多少 token 未排位
            // 初始化为该 expert 从各 rank 接收到的 token 计数
            uint32_t remaining[kNumRanksPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                remaining[i] = stored_rank_count[i];

            // offset: 前几轮已排掉的 token 行数（按单 rank 计，即每轮的 length 累加）
            uint32_t offset = 0;

            // ⚠️ 当前 token 在该 expert 内的相对位置（0-based）
            uint32_t token_idx_in_expert = token_idx - expert_start_idx;

            // ⚠️ slot_idx初始值为当前 token 在该 expert 内的相对位置（0-based）
            uint32_t slot_idx = token_idx_in_expert;

            // 输出：当前 token 在其所属 rank 内的位置（0-based）
            uint32_t token_idx_in_rank;


            // ⚠️⚠️⚠️⚠️ 接下里的操作是来选择，这个token idx只是一个位置，还没确定这个token是来自哪个rank，也没确定是这个rank发送过来的第几个
            // ⚠️⚠️⚠️⚠️ 下面就是确定这两个信息！！！这样的结果就是每个expert的所有token的排布应该是按rank均衡的，例如来自[0 1 2 3 0 1 2 3]这样交错的rank
            while (true) {
                // ---- Step 1: 统计本轮信息 ----
                // 计算还有多少个 rank 有剩余 token，以及这些 rank 中的最小剩余量
                // ⚠️  per-lane级存储，还需要从多少个rank pull token过来
                uint32_t num_actives_in_lane = 0;    
                // ⚠️ 用户warp reduce，求出要pull的per-rank最小token数
                uint32_t min_in_lane = 0xffffffff;   

                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                    // 统计有剩余的 rank 数
                    num_actives_in_lane += remaining[i] > 0;    
                    // 跳过不发token的rank
                    if (remaining[i] > 0)
                        min_in_lane = cute::min(min_in_lane, remaining[i]);  // 求最小剩余
                }

                //  ⚠️ num_active_ranks: 本轮还有 token 的 rank 数
                const uint32_t num_active_ranks = __reduce_add_sync(0xffffffff, num_actives_in_lane);
                //  ⚠️ 取最小的 token 数，每个rank pull length个token
                const uint32_t length = __reduce_min_sync(0xffffffff, min_in_lane);
                

 
                // ---- Step 2: 判断目标 token 是否在本轮内 ----
                // ⚠️ 每个rank pull length个token，这一round pull num_round_tokens个token
                const uint32_t num_round_tokens = length * num_active_ranks;

                // ⚠️假设得到的length = 2， num_active_ranks = 4，那么就是
                //    那么对应数据[length, num_active_ranks]
                //    slot_idx/num_active_ranks 表示行号，表示这是这轮每个rank的第几个token
                //    slot_idx % num_active_ranks 表示列号，表示这是这轮要发送的第几个rank
                if (slot_idx < num_round_tokens) {
                    // 目标 token 在本轮内，解码为 (rank, token_in_rank)

                    // ⚠️ 轮序消费rank，一个rank 一次 pull 1个token，交错pull
                    const uint32_t slot_idx_in_round = slot_idx % num_active_ranks;
                    uint32_t num_seen_ranks = 0;
                    current_rank_in_expert_idx = 0;

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumRanksPerLane; ++ i) {
                        // ⚠️ 返一个mask，32个bit，标记哪些lane为true
                        const uint32_t mask = __ballot_sync(0xffffffff, remaining[i] > 0);
                        // ⚠️ __popc(mask)：统计 mask 中 1 的个数 = 活跃 lane 数 
                        // 表示 这次32个rank 中多少个rank是active的
                        const uint32_t num_active_lanes = __popc(mask);
                        // 轮询到的下一个rank是不是在这一批次的32个rank中
                        if (slot_idx_in_round >= num_seen_ranks and slot_idx_in_round < num_seen_ranks + num_active_lanes)
                            // ⚠️ 是的话就定位出来
                            // ⚠️ __fns(mask, 0, N)：找 mask 中第 N 个置位的 bit 位置 = 第 N 个活跃 rank 的 lane 编号
                            current_rank_in_expert_idx = i * 32 + __fns(mask, 0, slot_idx_in_round - num_seen_ranks + 1);
                        // 不是就下一批次
                        num_seen_ranks += num_active_lanes;
                    }
                    // ⚠️ 最后得到的current_rank_in_expert_idx是轮询到的rank
                    //    slot_idx/num_active_ranks 表示行号，表示这是这轮每个rank的第几个token
                    //    slot_idx % num_active_ranks 表示列号，表示这是这轮要发送的第几个rank
                    // token_idx_in_rank: 目标 token 在其所属 rank 内的行号
                    // offset 是前几轮已排掉的行数，slot_idx / num_active_ranks 是本轮内的行号
                    token_idx_in_rank = offset + (slot_idx / num_active_ranks);
                    break;
                }

                //  ⚠️ 没有break跳出循环，说明上面slot_idx >= num_round_tokens
                //  ⚠️ ⚠️⚠️⚠️⚠️⚠️
                //⚠️这轮 "剥" 的层太薄了，目标 token 还在更深的层里。
                //⚠️所以：
                //⚠️把 slot_idx 减掉当前层的 token 数（跳过当前层）
                //⚠️offset 累加当前层的行数（记录已跳过的行数）
                //⚠️remaining 各自扣掉 length（耗尽的 rank 自然归零，下轮不再参与）
                //⚠️slot_idx -= num_round_tokens;   // 减去本轮的 token 数
                // ⚠️累加本轮排掉的行数
                offset += length;                
                #pragma unroll
                for (uint32_t i = 0; i < kNumRanksPerLane; ++ i)
                    // 扣除本轮消耗的 length 个 token；已耗尽的 rank 自动变 0
                    remaining[i] -= cute::min(remaining[i], length);
            }

            // Read source token-topk index (written by remote dispatch via NVLink)
            // ⚠️ src_token_topk_idx布局[num_ranks, num_experts, max_tokens_per_expert]
            const uint32_t src_token_topk_idx = *workspace.get_src_token_topk_idx_ptr(
                current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank);
            // ⚠️ 该token在src_rank中的位置
            const uint32_t src_token_idx = src_token_topk_idx / kNumTopk;
            // ⚠️ 该token在src_rank中的topk位置
            const uint32_t src_topk_idx = src_token_topk_idx % kNumTopk;

            // TMA load token from remote rank into shared memory 
            // ⚠️ 直接从远端src_rank pull token到tma buffer上
            if (cute::elect_one_sync()) {
                ptx::tma_load_1d(
                    pull_buffer.get_base_ptr(),
                    sym_buffer.map(input_token_buffer.get_data_buffer(src_token_idx).get_base_ptr(),
                                   current_rank_in_expert_idx),
                    pull_mbarrier, kHidden);
            }
            __syncwarp();

            // Load and store SF (overlaps with TMA token load)
            //
            // 【SF 存储布局说明】
            // SF (Scaling Factor) 在 L1 中的存储布局是 [kNumSFUint32][kNumPaddedSFPoolTokens]，
            // 即按 K 方向分组排列，而非按 token 连续排列：
            //   local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx]
            //   - 第 1 维 j: K 方向上每 128 个元素对应 1 个 uint32_t，共 kHidden/128 个
            //   - 第 2 维 sf_pool_token_idx: token 索引（经过 UTCCP 转置映射）
            //
            // 为什么不是 [token][kNumSFUint32] 连续排列？
            // 因为后续 GEMM 阶段 SFA 通过 UTCCP 指令从 L1 加载到 Tensor Memory，
            // UTCCP 要求源数据在 shared memory 中已按 4×32 转置格式排列，
            // 一条 UTCCP 指令可搬运 128 个 SF 值到 tmem，效率极高。
            // 所以 dispatch 阶段写入时就要提前做好转置，物理布局必须适配硬件 UTCCP 格式。
            //
            // 整个链路：
            //   1. Dispatch 阶段（此处）：从远端 GPU 拉取 SF，通过 transform_sf_token_idx 计算转置后地址，写入 L1 SF 池
            //   2. GEMM 阶段：MMA warp 通过 UTCCP 指令从 L1 SF 池直接加载到 tmem
            //   3. Epilogue 阶段：从 tmem 读取 SFA，对 UMMA 累加结果做反量化
            //
            // 【sf_pool_token_idx 的计算】
            //   sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)
            //   - expert_pool_block_offset * SF_BLOCK_M: 当前 expert 在池中的起始偏移（按 SF 块对齐后的偏移）
            //   - transform_sf_token_idx(): UTCCP 4×32 转置索引映射，将逻辑连续 token 索引映射为 UTCCP 所需的物理位置
            //
            // 【transform_sf_token_idx 转置映射详解】(定义在本文件第 253 行)
            //   idx = token_idx_in_expert % BLOCK_M
            //   结果 = token_idx_in_expert / BLOCK_M * SF_BLOCK_M + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)
            //
            //   SF_BLOCK_M = align(BLOCK_M, 128)，将 BLOCK_M 向上对齐到 128 的倍数（UTCCP 以 128 为单位搬运）
            //
            //   在每 128 个元素的组内做 4×32 矩阵转置：
            //     128 个元素被看作 4×32 的矩阵（4列 × 32行）
            //     - (idx & 31u): 行号（低 5 位，0~31）
            //     - ((idx >> 5) & 3u): 列号（接下来的 2 位，0~3）
            //     - 转置后地址 = 行 * 4 + 列 = (idx & 31u) * 4 + ((idx >> 5) & 3u)
            //
            //   例子（BLOCK_M=128, idx_in_group 0~127）：
            //     idx=0 → 转置后位置 0*4+0 = 0
            //     idx=1 → 转置后位置 1*4+0 = 4
            //     idx=4 → 转置后位置 0*4+1 = 1
            //     idx=5 → 转置后位置 1*4+1 = 5
            constexpr uint32_t kNumSFUint32 = kHidden / 128; 
            DG_STATIC_ASSERT(kNumSFUint32 > 0 and kHidden % 128 == 0, "Invalid SF");
            const auto remote_sf_ptr = sym_buffer.map(
                input_sf_buffer.get_data_buffer(src_token_idx).get_base_ptr<uint32_t>(),
                current_rank_in_expert_idx);
            // 直接存到 L1 gmem，不通过 smem 中转
            const auto local_sf_ptr = l1_sf_buffer.get_base_ptr<uint32_t>();
            // 计算 SF 在 L1 池中的索引：expert 偏移 + UTCCP 转置映射后的 token 位置
            const auto sf_pool_token_idx = expert_pool_block_offset * SF_BLOCK_M +
                transform_sf_token_idx(token_idx_in_expert);
            // 每个 lane 负责写入若干个 K 分组的 SF 值，warp 内 32 个 lane 协作覆盖所有 kNumSFUint32
            #pragma unroll
            for (uint32_t i = 0; i < math::constexpr_ceil_div(kNumSFUint32, 32u); ++ i) {
                const uint32_t j = i * 32 + lane_idx;
                if (j < kNumSFUint32)
                    local_sf_ptr[j * kNumPaddedSFPoolTokens + sf_pool_token_idx] = remote_sf_ptr[j];
            }
            __syncwarp();

            // Store weights and token data
            const uint32_t pool_token_idx = expert_pool_block_offset * BLOCK_M + token_idx_in_expert;
            if (cute::elect_one_sync()) {
                // Load weights
                const auto weight = *sym_buffer.map(
                    input_topk_weights_buffer.get_base_ptr<float>() + src_token_topk_idx,
                    current_rank_in_expert_idx);
                *l1_topk_weights_buffer.get_data_buffer(pool_token_idx).get_base_ptr<float>() = weight;

                // Wait for TMA token load to complete
                ptx::mbarrier_arrive_and_set_tx(pull_mbarrier, kHidden);
                ptx::mbarrier_wait_and_flip_phase(pull_mbarrier, pull_mbarrier_phase);

                // Store token to local L1 buffer via TMA
                ptx::tma_store_1d(
                    l1_token_buffer.get_data_buffer(pool_token_idx).get_base_ptr(),
                    pull_buffer.get_base_ptr(), pull_buffer.get_num_bytes());

                // Write source metadata for combine write-back
                *workspace.get_token_src_metadata_ptr(pool_token_idx) =
                    {current_rank_in_expert_idx, src_token_idx, src_topk_idx};

                // Wait for token TMA store to complete
                cute::tma_store_arrive();
                ptx::tma_store_wait<0>();

                // 生产者 (Dispatch warp, 第 1015 行):
                ptx::red_add_rel( //带 release 语义的原子加，确保 token 数据写入对 GEMM 可见
                    workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + token_idx_in_expert / BLOCK_M), 1);
            }
            __syncwarp();
        }

        //
        // 5. 清理阶段: 清理workspace，为下一轮计算做准备
        //    - 清除expert发送计数
        //    - 清除token计数
        //    - 清除L1/L2到达标记
        //
        // 6. 同步阶段: 等待所有rank完成清理
        //
        // Clean workspace for the next usage, and also do cumulative stats
        // NOTES: it is overlapped with combine reduction epilogue
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        DG_STATIC_ASSERT(kNumSMs > 1, "Invalid SM count");
        if (sm_idx == 0) {
            // SM 0: clear expert send count
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads)
                *workspace.get_expert_send_count_ptr(i) = 0;
        } else {
            // Other SMs: clean blocks
            for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
                // Read expert token count before clearing
                const auto num_recv_tokens = static_cast<uint32_t>(
                    *workspace.get_expert_recv_count_sum_ptr(i));
                const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);

                // Compute expert pool block offset
                expert_pool_block_offset = scheduler.get_pool_block_offset(i);

                // Wait read count ready
                ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx);

                // Clean expert token count, and add cumulative results
                DG_STATIC_ASSERT(kNumDispatchWarps >= 2, "Not enough dispatch warps");
                if (warp_idx == 0) {
                    *workspace.get_expert_recv_count_sum_ptr(i) = 0;
                } else if (warp_idx == 1) {
                    if (cute::elect_one_sync() and cumulative_local_expert_recv_stats != nullptr)
                        ptx::red_add(cumulative_local_expert_recv_stats + i, static_cast<int>(num_recv_tokens));
                    __syncwarp();
                }

                // Clean per-rank token count
                for (uint32_t j = thread_idx; j < kNumRanks; j += kNumDispatchThreads)
                    *workspace.get_expert_recv_count_ptr(j, i) = 0;
                __syncwarp();

                // Clean L1 and L2 arrival stuffs
                for (uint32_t j = thread_idx; j < num_recv_m_blocks; j += kNumDispatchThreads) {
                    *workspace.get_l1_arrival_count_ptr(expert_pool_block_offset + j) = 0;
                    *workspace.get_l2_arrival_mask_ptr(expert_pool_block_offset + j) = 0;
                }
                __syncwarp();
            }
        }

        // Wait for all ranks to finish cleaning
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumDispatchThreads,
                             kDispatchGridSyncIndex, kAfterWorkspaceCleanBarrierTag>(
            workspace, sym_buffer, sm_idx, thread_idx,
            [=]() { ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx); },
            /* Before the NVLink barrier, there is a grid sync */ true,
            /* At the end of kernel does not need to sync */ false
        );

        
    } else if (warp_idx == kNumDispatchWarps) {
        // ============================================================
        // 【MMA Load Warp A - 加载Tokens和SFA】
        //
        //  ┌──────────────────────────────────────────────────────────────────────┐
        //  │ Global Memory ──TMA──> Shared Memory A (tokens) ──TMA──> TMEM          │
        //  │                     Shared Memory SFA (scaling factors)               │
        //  └──────────────────────────────────────────────────────────────────────┘
        //
        // 职责:
        //   - 通过调度器获取当前需要处理的block信息
        //   - 使用TMA从全局内存加载tokens到共享内存
        //   - 加载对应的scaling factors (SFA)
        //   - 支持L1和L2两个线性层的切换
        //
        // 调度流程:
        //   for_each_block():
        //     ┌─────────────────────────────────────────────────────────┐
        //     │ Block #0 (expert 0, m_block 0):                        │
        //     │   Linear1: [tokens] @ [weights_L1] → [intermediate]    │
        //     │   Linear2: [intermediate] @ [weights_L2] → [output]    │
        //     │                                                      │
        //     │ Block #1 (expert 0, m_block 1):                       │
        //     │   ...                                                 │
        //     │                                                      │
        //     │ Block #2 (expert 1, m_block 0):                       │
        //     │   ...                                                 │
        //     └─────────────────────────────────────────────────────────┘
        //
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for tokens with SFA
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_a_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts : &tensor_map_l1_acts;
            const auto tensor_map_sfa_ptr = block_phase == sched::BlockPhase::Linear2
                ? &tensor_map_l2_acts_sf : &tensor_map_l1_acts_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_sfa_k = math::ceil_div(shape_k, kGranK * 4u);

            // Compute pool block offset for this expert
            // ⚠️ 全局m block idx
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;

            // Wait the entire token arrival for linear 1
            if (block_phase == sched::BlockPhase::Linear1) {
                // 消费者 (GEMM warp, 第 1136-1139 行):
                const auto ptr = workspace.get_l1_arrival_count_ptr(pool_block_idx);
                const auto expected = scheduler.template get_valid_m<false>();// 当前 block 期望的 token 数
                while (ptx::ld_acq(ptr) != expected);// 自旋等待，带 acquire 语义的加载，确保能看到 dispatch 写入的数据

            } else {
                // The L1 output's block N is halved into `BLOCK_K / 2`, so we have to wait 2x L1 blocks' arrival
                // NOTES: Originally we wait blocks on-demand to overlap L1 calculation
                // with L2, but this optimization is negative when `num_experts_per_wave`
                // guarantees L1's completion when L2 starts. So we remove it.
                // In the future, if `num_experts_per_wave` is not large enough
                // due to small `num_experts_per_rank`, we may need to add it back or add a switch
                DG_STATIC_ASSERT(BLOCK_K == BLOCK_N, "Invalid block sizes");
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                
                // NOTES: Equivalent to `(1ull << (2 * num_k_blocks)) - 1`, but split into two shifts
                // to avoid undefined behavior when `num_k_blocks == 32`

                // 低 2*num_k_blocks 位全 1
                // L2 的输入是 L1 的输出（中间激活），L1 按 K 维度分块写出，每个 K 块对应一个 bit。
                // 因为 BLOCK_K == BLOCK_N，L1 输出的 block N 被拆成 BLOCK_K/2 大小，所以 L1 每写完一个 K block，L2 需要等 2 个子块到位，即 2 * num_k_blocks 个 bit 全部置 1。
                const uint64_t expected = ((1ull << num_k_blocks) << num_k_blocks) - 1;
                while (ptx::ld_acq_gpu(ptr) != expected);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // ⚠️ Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute token offset from pool block index
                uint32_t m_idx = pool_block_idx * BLOCK_M;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t sfa_m_idx = pool_block_idx * SF_BLOCK_M;
                uint32_t sfa_k_idx = k_block_idx;

                // Add 2 CTA offsets for non-leader CTA
                if (not is_leader_cta)
                    m_idx += scheduler.template get_valid_m<true>() / 2;

                // TMA copy tokens and SFA, then arrive at full barrier
                if (cute::elect_one_sync()) {
                    // BLOCK_K (=128)	内维（K方向）的块大小
                    // LOAD_BLOCK_M	外维（M方向）的块大小，即一次加载的 token 行数
                    // kSwizzleAMode	smem 的 swizzle 模式，用于避免 bank conflict
                    // a_dtype_t	数据类型（FP8 e4m3/e5m2）
                    // tensor_map_a_ptr	TMA tensor map 描述符（指向 HBM 中 A 矩阵的布局信息）
                    // k_idx = k_block_idx * BLOCK_K	内维起始偏移（从第几个 K 元素开始读）
                    // m_idx = pool_block_idx * BLOCK_M	外维起始偏移（从第几个 token 行开始读）
                    // ⚠️num_tma_multicast	2	2-CTA multicast：2 个 CTA（leader + non-leader）共享同一笔 TMA 传输，数据只需从 HBM 读一次，写入两个 CTA 的 smem
                    //                          这是 SM100 的 2SM TMA Load 特性，同一个 128×LOAD_BLOCK_M 的 tile 只需从 HBM 搬一次，两个 CTA 同时收到，节省带宽。
                    // ⚠️LOAD_BLOCK_M = BLOCK_M / 2，因为 2-CTA multicast 模式下，leader CTA 和 non-leader CTA 各负责 BLOCK_M 的一半（第 1176-1177 行 non-leader 会加偏移 get_valid_m<true>() / 2），所以每个 CTA 只需从 HBM 加载半行，两份拼起来就是完整的 BLOCK_M 行。
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(
                        tensor_map_a_ptr, full_barriers[stage_idx], smem_a[stage_idx], k_idx, m_idx, 2);

                    // K 方向，总共有 kHidden/128 行（每 128 个元素 = 4 个 SF 组合为 1 个 uint32_t）
                    // sfa_m_idx = pool_block_idx * SF_BLOCK_M：M 方向偏移（token 行）
                    // sfa_k_idx = k_block_idx：K 方向偏移（哪一行 SF）
                    // 所以 SFA 的 TMA 是一次加载 SF_BLOCK_M 个连续 token 在同一个 K 位置的 SF 值，正好对应 UTCCP 一次搬运 128 个 SF 的需求。
                    tma::copy<SF_BLOCK_M, 1, 0>(
                        tensor_map_sfa_ptr, full_barriers[stage_idx], smem_sfa[stage_idx], sfa_m_idx, sfa_k_idx, 2);

                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE * 2 + SF_BLOCK_M * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 1) {
        // ============================================================
        // 【MMA Load Warp B - 加载Weights和SFB】
        //
        //  ┌──────────────────────────────────────────────────────────────────────┐
        //  │ Global Memory ──TMA──> Shared Memory B (weights, FP4)              │
        //  │                     Shared Memory SFB (scaling factors)            │
        //  └──────────────────────────────────────────────────────────────────────┘
        //
        // 职责:
        //   - 加载FP4格式的权重数据到共享内存
        //   - 加载对应的scaling factors (SFB)
        //   - 权重按expert分区，每个expert有独立的权重块
        //
        // 权重布局 (以L1为例):
        //   ┌────────────────────────────────────────────────────────────┐
        //   │ expert[0] weights:  [N=intermediate*2, K=hidden]           │
        //   │ expert[1] weights:  [N=intermediate*2, K=hidden]           │
        //   │ ...                                                        │
        //   │ expert[n] weights:  [N=intermediate*2, K=hidden]           │
        //   └────────────────────────────────────────────────────────────┘
        //
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM TMA load warp for weights with SF
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            const auto tensor_map_b_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights : &tensor_map_l1_weights;
            const auto tensor_map_sfb_ptr =
                block_phase == sched::BlockPhase::Linear2 ? &tensor_map_l2_weights_sf : &tensor_map_l1_weights_sf;

            const auto shape_k = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_K : L1_SHAPE_K;
            const auto shape_n = block_phase == sched::BlockPhase::Linear2 ? L2_SHAPE_N : L1_SHAPE_N;

            // ⚠️ 同理 A 和 B 的 SF 量化粒度是一样的：1 SF / 32 元素，4 个 SF 打包为 1 个 uint32。
            // ⚠️ A 和 B 的 shape_sf*_k 含义完全相同——都是 K 方向的 SF uint32 行数 = ceil(shape_k / 128)。
            const auto shape_sfb_k = math::ceil_div(shape_k, kGranK * 4u);

            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                // ⚠️ Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute weight offset
                uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;
                uint32_t k_idx = k_block_idx * BLOCK_K;
                // ⚠️ 权重数据在 HBM 中是 [num_experts, shape_n, shape_k] 布局，第 local_expert_idx 个 expert 的权重从 local_expert_idx * shape_n 开始。
                // ⚠️ SFB（权重的 SF）同理，在 HBM 中也是按 expert 分区：[num_experts, shape_sfb_k, shape_sfb_n]。
                // 第 local_expert_idx 个 expert 的 SF 从 local_expert_idx * shape_sfb_k 行开始，再加上当前 K 块偏移 k_block_idx。
                uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                // BLOCK_K = 128 对应4个ScaleingFactors，包装成一个uint32
                uint32_t sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx;

                // TMA copy weights with SF
                if (cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(
                        tensor_map_b_ptr, full_barriers[stage_idx], smem_b[stage_idx], k_idx, n_idx, 2);
                    tma::copy<BLOCK_N, 1, 0>(
                        tensor_map_sfb_ptr, full_barriers[stage_idx], smem_sfb[stage_idx], sfb_n_idx, sfb_k_idx, 2);
                    if (is_leader_cta) {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE + BLOCK_N * sizeof(uint32_t) * 2);
                    } else {
                        full_barriers[stage_idx]->arrive(0u);
                    }
                }
                __syncwarp();
            }
        });
    } else if (warp_idx == kNumDispatchWarps + 2) {
        // ============================================================
        // 【MMA Issue Warp - 执行矩阵乘法】
        //
        //  ┌──────────────────────────────────────────────────────────────────────┐
        //  │                                                             │
        //  │    SMEM A (tokens)      SMEM B (weights)                     │
        //  │         │                       │                            │
        //  │         ▼                       ▼                            │
        //  │    ┌─────────────────────────────────────┐                 │
        //  │    │           UMMA FMA Unit             │                 │
        //  │    │    (Tensor Core on SM100)            │                 │
        //  │    │                                     │                 │
        //  │    │  D[M,N] += A[M,K] * B[N,K]         │                 │
        //  │    │                                     │                 │
        //  │    └─────────────────────────────────────┘                 │
        //  │                   │                                         │
        //  │                   ▼                                         │
        //  │              TMEM (Accumulator)                             │
        //  └──────────────────────────────────────────────────────────────────────┘
        //
        // SM100 UMMA特性:
        //   - 2x1 SM模式: 跨2个SM协同执行
        //   - Block-scaled: 使用UE8M0格式的scaling factors
        //   - 指令级并行: 每个warp同时执行多个MMA指令
        //
        // Pipeline流程:
        //   Stage 0: Load A,B → Compute → Store to TMEM
        //   Stage 1: Load A,B → Compute → Store to TMEM
        //   Stage 2: Load A,B → Compute → Store to TMEM
        //   ...
        //
        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

        // GEMM MMA issue warp (only the leader CTA will run)
        if (is_leader_cta) {
            // Make instruction descriptor with block scaling
            // NOTES: always swap A/B
            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                UMMA_M, UMMA_N,
                cute::UMMA::Major::K, cute::UMMA::Major::K
            >();
            auto sf_desc = mma::sm100::make_sf_desc(nullptr);

            DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem_a[0], 0, 0);
            auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem_b[0], 0, 0);
            uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
            uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

            // Checks for MMA instructions
            DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                             (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                             "Invalid MMA instruction shape");

            // Persistently schedule over blocks
            uint32_t current_iter_idx = 0;
            scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                         const uint32_t& local_expert_idx,
                                         const uint32_t& num_k_blocks,
                                         const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
                // Dynamic update of UMMA N based on effective M
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, scheduler.template get_valid_m<true>());

                // Wait tensor memory empty barrier arrival
                const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
                const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
                tmem_empty_barriers[accum_stage_idx]->wait(accum_phase ^ 1);
                ptx::tcgen05_after_thread_sync();

                // Empty barrier arrival
                auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                    auto umma_arrive = [](const uint64_t* barrier) {
                        constexpr uint16_t kCTAMask = (1 << 2) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    };
                    umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                    // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                    if (do_tmem_full_arrive)
                        umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                    __syncwarp();
                };

                // Launch MMAs
                #pragma unroll 2
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
                    // Wait TMA load completion
                    full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();

                    const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                    const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);
                    if (cute::elect_one_sync()) {
                        // UTCCP copy SFA and SFB to TMEM
                        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }

                        // Issue UMMA
                        #pragma unroll
                        for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++ k) {
                            const auto runtime_instr_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
                            a_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, k * UMMA_K);
                            b_desc.lo = mma::sm100::advance_umma_desc_lo<
                                cute::UMMA::Major::K, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
                            ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
                                b_desc, a_desc, accum_stage_idx * UMMA_N,
                                k_block_idx > 0 or k > 0, runtime_instr_desc,
                                kTmemStartColOfSFB, kTmemStartColOfSFA);
                        }
                    }
                    __syncwarp();

                    // Commit to the mbarrier object
                    // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
                }
            });

            // To safely deconstruct barriers, we need another round of waits
            if (current_iter_idx > 0) {
                const auto accum_phase_idx = ((current_iter_idx - 1) / kNumEpilogueStages) & 1;
                tmem_empty_barriers[(current_iter_idx - 1) % kNumEpilogueStages]->wait(accum_phase_idx);
            }
        }
    } else if (warp_idx == kNumDispatchWarps + 3) {
        // 【预留Warp】 - 暂时为空，保持线程同步

        // Adjust registers
        cutlass::arch::warpgroup_reg_dealloc<kNumNonEpilogueRegisters>();

    } else if (warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps) {
        // ============================================================
        // 4.5 Epilogue Warp — 覆盖 L1 Epilogue / L2 Epilogue / Combine 三阶段
        //
        // 8 个 epilogue warp (256 线程, reg=208) 分为 2 个 WarpGroup (WG):
        //
        // 数据切分 (以 BLOCK_M=128, BLOCK_N=128, STORE_BLOCK_M=32 为例):
        //
        //   M=0                                M=127
        //   ┌──────────────────────────────────┐
        //   │           WG 0                   │  WG_BLOCK_M = 64
        //   │  ┌────────────┐ ┌────────────┐   │
        //   │  │ STORE 0    │ │ STORE 1    │   │  STORE_BLOCK_M = 32
        //   │  │┌──┐┌──┐┌──┐│ │            │   │
        //   │  ││A0││A1││A2││ │  ...       │   │  ATOM_M = 8
        //   │  ││A3││A4││A5││ │            │   │
        //   │  │└──┘└──┘└──┘│ │            │   │
        //   │  └────────────┘ └────────────┘   │
        //   ├──────────────────────────────────┤
        //   │           WG 1                   │  WG_BLOCK_M = 64
        //   │  ┌────────────┐ ┌────────────┐   │
        //   │  │ STORE 2    │ │ STORE 3    │   │
        //   │  └────────────┘ └────────────┘   │
        //   └──────────────────────────────────┘
        //
        //   WG 内 N 维平铺 (BLOCK_N=128 → 每 warp 32 列):
        //           warp0    warp1    warp2    warp3
        //     N=0 ┌──────┬────────┬────────┬──────┐ N=127
        //         │ 32列 │  32列  │  32列  │ 32列 │
        //         └──────┴────────┴────────┴──────┘
        //
        //   每个 Atom = ATOM_M(8行) × (BLOCK_N/4=32列), 由 1 个 warp 处理
        //   每个 warp 内 32 lane: 每 4 lane 处理同一 token 的 N 维连续元素
        //
        // ┌──────────────────────────────────────────────────────────────────┐
        // │  L1 Epilogue: TMEM→SwiGLU(silu(gate)*up*weight)→amax→FP8量化  │
        // │               →TMA store l2_token_buffer→red_or(l2_arrival_mask)│
        // │  L2 Epilogue: TMEM→BF16→STSM smem→NVLink 远端写 combine_buffer   │
        // │  Combine:     双缓冲 topk BF16→float累加→cast BF16→TMA store y  │
        // └──────────────────────────────────────────────────────────────────┘
        //
        // Adjust registers: epilogue 需要 208 寄存器 (SwiGLU/amax/combine 开销大)
        cutlass::arch::warpgroup_reg_alloc<kNumEpilogueRegisters>();

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // 4.5.1 初始化阶段 — 多级 ID 分解 (详见上方全景图)
        //
        // 数据切分层级: BLOCK_M → WG_BLOCK_M → STORE_BLOCK_M → ATOM_M(8)
        //           ──▶ STORE_BLOCK_M (TMA store 粒度)
        //           ──▶ ATOM_M = 8 (最小 TMEM_LOAD/SwiGLU/amax 粒度)
        //
        // 每个 warp (32 lane) 负责 WG_BLOCK_M × (BLOCK_N/4) 的子块
        const auto epilogue_warp_idx = warp_idx - (kNumDispatchWarps + kNumMMANonEpilogueWarps); // 0..7
        const auto epilogue_wg_idx = epilogue_warp_idx / 4;               // WG 编号: 0 或 1
        const auto epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx; // WG 内线程号
        const auto warp_idx_in_wg = epilogue_warp_idx % 4;                // WG 内 warp 编号: 0..3
        DG_STATIC_ASSERT((kNumDispatchWarps + kNumMMANonEpilogueWarps) % 4 == 0 and
                         kNumEpilogueWarps % 4 == 0, "Invalid epilogue warps");

        // TODO: support effective block M
        // NOTES:
        //  - 2 warpgroups divide the whole BM into BM / 2
        //  - 4 warps divide the whole BN into BN / 4
        //  - BM / 2 is further divided into stored blocks, i.e. with `STORE_BLOCK_M` size
        //  - `STORE_BLOCK_M` in further divided into `ATOM_M`
        constexpr uint32_t WG_BLOCK_M = BLOCK_M / kNumEpilogueWarpgroups;
        constexpr uint32_t ATOM_M = 8;
        constexpr uint32_t kNumBankGroupBytes = 16u;
        constexpr uint32_t kNumAtomsPerStore = STORE_BLOCK_M / ATOM_M;
        DG_STATIC_ASSERT(BLOCK_M % kNumEpilogueWarpgroups == 0, "Invalid block M");
        DG_STATIC_ASSERT(WG_BLOCK_M % STORE_BLOCK_M == 0, "Invalid warpgroup block M");
        DG_STATIC_ASSERT(STORE_BLOCK_M % ATOM_M == 0, "Invalid store block M");
        DG_STATIC_ASSERT(BLOCK_N == 128, "Invalid block N");

        // 4.5.2 与 Dispatch Warp 同步
        //
        // 两次同步, 解决 SMEM 和 workspace 所有权切换:
        //
        // 第 1 次 (此处):
        //   dispatch 的 smem_send_buffers (pull TMA 1D 落地区)
        //   与 epilogue combine 阶段的 combine_load/combine_store 复用 smem
        //   同步后: dispatch pull 阶段和 epilogue 阶段互不干扰
        //
        // 第 2 次 (Dispatch 清理 workspace 前):
        //   epilogue combine 需要复用 smem 做 chunk 缓冲
        //   dispatch 需要等 combine 启动后才能安全清理 workspace
        //
        // 使用 sync_unaligned: dispatch (128 线程) 和 epilogue (256 线程)
        //   线程数不同且 warp 边界不对齐, 只能用 bar.sync unaligned 变体
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // 4.5.3 Block 循环 — 与 TMA/MMA warp 共享同一 scheduler.for_each_block
        //
        // 循环内按 Phase 分支: Linear1 → L1 Epilogue (SwiGLU+FP8量化)
        //                   Linear2 → L2 Epilogue (BF16+NVLink远端写)
        // 循环结束后 → Combine 阶段
        uint32_t current_iter_idx = 0;
        scheduler.for_each_block([&](const sched::BlockPhase& block_phase,
                                     const uint32_t& local_expert_idx,
                                     const uint32_t& num_k_blocks,
                                     const uint32_t& m_block_idx, const uint32_t& n_block_idx) {
            // 等待 UMMA arrival: tmem_full_barriers→init(1), 仅 leader CTA arrive 一次
            // 2-CTA UMMA 在两个 CTA 的 TMEM 上都写了结果, leader arrive 后硬件保证两侧就绪
            const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
            const auto accum_phase = (current_iter_idx ++ / kNumEpilogueStages) & 1;
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase);
            ptx::tcgen05_after_thread_sync(); // 普通线程同步对 tcgen05 可见

            // 计算 block 在全局坐标系的起点
            // ptx::exchange(..., 0): 从 lane 0 广播 valid_m, 显式告诉编译器全 warp 一致
            const uint32_t valid_m = ptx::exchange(scheduler.template get_valid_m<false>(), 0);
            const uint32_t pool_block_idx = scheduler.get_current_pool_block_offset() + m_block_idx;
            uint32_t m_idx = pool_block_idx * BLOCK_M;
            uint32_t n_idx = n_block_idx * BLOCK_N;

            if (block_phase == sched::BlockPhase::Linear1) {
                // ============================================================
                // 【L1 Epilogue - SwiGLU激活】
                //
                // SwiGLU流程:
                //   ┌─────────────────────────────────────────────────┐
                //   │  L1_output[:, :] (FP8)                           │
                //   │         │                                        │
                //   │         ▼ split                                 │
                //   │  ┌─────────────────┐  ┌─────────────────┐       │
                //   │  │ gate (N/2 列)   │  │   up (N/2 列)   │       │
                //   │  └─────────────────┘  └─────────────────┘       │
                //   │         │                    │                 │
                //   │         ▼                    ▼                 │
                //   │    SiLU(gate)          (identity)            │
                //   │         │                    │                 │
                //   │         └─────────── * ──────┘                 │
                //   │                    │                          │
                //   │                    ▼                          │
                //   │         output (FP8, N/2 列)                  │
                //   └─────────────────────────────────────────────────┘
                //
                // Unified L1 epilogue: SwiGLU in-place using granularity 8 interleaved weights
                // With `SM100_TMEM_LOAD_16dp256b1x`, gate/up pairs are:
                //   (values[0], values[2]), (values[1], values[3]),
                //   (values[4], values[6]), (values[5], values[7])
                float stored_cached_weight = 0;

                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    // Iterate all atoms in the store block
                    //
                    // 单个 ATOM Tile = ATOM_M(8) × 32 个元素, 32 个 thread 各处理 1 行 × 8 列
                    //
                    //  列: g=gate, u=up (interleaved)       token 分组 (stride=4, 每 8 lane 1 token)
                    //   col 0  1  2  3  4  5  6  7           T=0  T=1  T=2  T=3
                    //   ┌──┬──┬──┬──┬──┬──┬──┬──┐
                    // 0 │g0│u0│g1│u1│g2│u2│g3│u3│ lane  0  L0
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // 1 │g0│u0│g1│u1│g2│u2│g3│u3│ lane  1  │L1
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // 2 │g0│u0│g1│u1│g2│u2│g3│u3│ lane  2  │ │L2
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // 3 │g0│u0│g1│u1│g2│u2│g3│u3│ lane  3  │ │ │L3
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // 4 │g0│u0│g1│u1│g2│u2│g3│u3│ lane 4   L4 │ │ │
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤         ↑stride=4, 同一 token
                    // 5 │g0│u0│g1│u1│g2│u2│g3│u3│ lane 5
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // ...                                 Token 0 lanes: {0,4,8,12,16,20,24,28}
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤         Token 1 lanes: {1,5,9,13,17,21,25,29}
                    // 28│g0│u0│g1│u1│g2│u2│g3│u3│ lane28  Token 2 lanes: {2,6,10,14,18,22,26,30}
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤         Token 3 lanes: {3,7,11,15,19,23,27,31}
                    // 29│g0│u0│g1│u1│g2│u2│g3│u3│ lane29
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤
                    // 30│g0│u0│g1│u1│g2│u2│g3│u3│ lane30  每 thread 的 8 个 values[]:
                    //   ├──┼──┼──┼──┼──┼──┼──┼──┤       TMEM_LOAD 1: values[0..3] (行 0..15)
                    // 31│g0│u0│g1│u1│g2│u2│g3│u3│ lane31  TMEM_LOAD 2: values[4..7] (行 16..31)
                    //   └──┴──┴──┴──┴──┴──┴──┴──┘
                    //   每行 8 值 → SwiGLU: silu(g0)u0, silu(g1)u1, silu(g2)u2, silu(g3)u3 → ×weight
                    float2 swiglu_values[kNumAtomsPerStore * 2];
                    float2 amax_values[kNumAtomsPerStore];
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        const uint32_t j = s * kNumAtomsPerStore + i;

                        // 4.5.4.2 加载 topk_weight: 每 4 atom (32 行) 做 1 次 GMEM 加载
                        // stored_cached_weight 缓存接下来 32 个 token 的权重
                        // 中间 3 atom 通过 ptx::exchange shuffle 直接取
                        DG_STATIC_ASSERT(32 % ATOM_M == 0, "Invalid block size");
                        if ((j * ATOM_M) % 32 == 0 and (WG_BLOCK_M % 32 == 0 or j * ATOM_M + lane_idx < WG_BLOCK_M)) {
                            stored_cached_weight = *l1_topk_weights_buffer
                                .get_data_buffer(m_idx + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M + lane_idx)
                                .get_base_ptr<float>();
                        }

                        // 从寄存器缓存读 weight: exchange(src_lane) 取对应 lane 的 weight
                        const float2 weights = {
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 0),
                            ptx::exchange(stored_cached_weight, (j * ATOM_M) % 32 + (lane_idx % 4) * 2 + 1)
                        };

                        // 4.5.4.3 TMEM 加载: 两次 SM100_TMEM_LOAD_16dp256b1x 凑满 32 行
                        // - accum_stage_idx * UMMA_N: 选 accumulator 槽 (AB-swap 后 UMMA_N 对应 M 方向)
                        // - epilogue_wg_idx * WG_BLOCK_M: warpgroup 半区偏移
                        // - j * ATOM_M: atom 内 M 偏移
                        // - | 0x00100000: 跳到下半 16 行 (256bit×16行 = ATOM_M×16, 再读 16 行凑 32)
                        //   gate/up 在行间交替, Thread 0 的 v[0],v[2] 构成一对 (gate,up)
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + j * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load(); // TMEM load 异步, 等待寄存器可见

                        // Signal tensor memory consumed on the last atom
                        if (j == WG_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // 4.5.4.5 SwiGLU: silu(gate) × up × weight
                        // FP32→BF16 截断降低精度需求, gate/up 对: (0,2), (1,3), (4,6), (5,7)
                        // k 循环 2 次: 每 atom 两对 (gate,up), 8 元素→4 个 float2 结果
                        auto fp32_values = reinterpret_cast<float*>(values);
                        #pragma unroll
                        for (uint32_t k = 0; k < 2; ++ k) {
                            auto bf16_gate = __float22bfloat162_rn(make_float2(fp32_values[k * 4], fp32_values[k * 4 + 1]));
                            auto bf16_up = __float22bfloat162_rn(make_float2(fp32_values[k * 4 + 2], fp32_values[k * 4 + 3]));

                            // Clamp: gate ≤ kActivationClamp, up ∈ [-kActivationClamp, kActivationClamp]
                            if constexpr (kActivationClamp != cute::numeric_limits<float>::infinity()) {
                                bf16_gate = __hmin2(bf16_gate, {kActivationClamp, kActivationClamp});
                                bf16_up = __hmax2(bf16_up, {-kActivationClamp, -kActivationClamp});
                                bf16_up = __hmin2(bf16_up, {kActivationClamp, kActivationClamp});
                            }

                            // SiLU: gate / (1 + exp(-gate))
                            // fastMath 用 __expf + fast_rcp 近似, 避免 SFU 算力瓶颈
                            auto gate = __bfloat1622float2(bf16_gate);
                            auto neg_gate_exp = make_float2(
                                kFastMath ? __expf(-gate.x) : expf(-gate.x),
                                kFastMath ? __expf(-gate.y) : expf(-gate.y));
                            const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
                            if constexpr (kFastMath) {
                                gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});
                            } else {
                                gate = {gate.x / denom.x, gate.y / denom.y};
                            }
                            const auto up = __bfloat1622float2(bf16_up);
                            swiglu_values[i * 2 + k] = __fmul2_rn(__fmul2_rn(gate, up), weights); // silu(gate)*up*weight
                        }

                        // 4.5.4.6 Amax 归约 — 第一层: warp 内 4-lane 规约
                        // 同一 token 在 N 维跳步为 4 (T0,T4,T8,...T28), warp_reduce<4,true> 求 max
                        amax_values[i].x = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].x), cute::abs(swiglu_values[i * 2 + 1].x)),
                            math::ReduceMax<float>());
                        amax_values[i].y = math::warp_reduce<4, true>(
                            cute::max(cute::abs(swiglu_values[i * 2 + 0].y), cute::abs(swiglu_values[i * 2 + 1].y)),
                            math::ReduceMax<float>());

                        // 第二层: 写 smem_amax_reduction, 每个 warp 写自己的槽位
                        // 前 4 个 lane 已在 broadcast 中得到 max 值, 只由它们写入
                        if (lane_idx < 4)
                            smem_amax_reduction[epilogue_warp_idx * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx] = amax_values[i];
                        __syncwarp();
                    }

                    // Wait shared memory release from previous TMA store
                    // And fence `smem_amax_reduction`
                    const uint32_t tma_stage_idx = s % kNumTMAStoreStages;
                    ptx::tma_store_wait<kNumTMAStoreStages - 1>();
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // Cast to FP8 E4M3 and store into shared memory
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumAtomsPerStore; ++ i) {
                        // Reduce amax
                        const float2 wp_amax =
                            smem_amax_reduction[(epilogue_warp_idx ^ 1) * (STORE_BLOCK_M / 2) + i * (ATOM_M / 2) + lane_idx % 4];
                        amax_values[i].x = cute::max(amax_values[i].x, wp_amax.x);
                        amax_values[i].y = cute::max(amax_values[i].y, wp_amax.y);

                        // Calculate SF
                        float2 sf, sf_inv;
                        math::get_e4m3_sf_and_sf_inv(amax_values[i], sf, sf_inv);

                        // 4.5.4.7 FP8 量化: amax→UE8M0 SF, sf_inv 缩放后 cast E4M3
                        // __nv_fp8x4_e4m3: 一次把 4 个 float 打包成 32-bit (4×E4M3)
                        const float2 upper = __fmul2_rn(swiglu_values[i * 2 + 0], sf_inv);
                        const float2 lower = __fmul2_rn(swiglu_values[i * 2 + 1], sf_inv);
                        const auto fp8x4_values = __nv_fp8x4_e4m3(make_float4(upper.x, upper.y, lower.x, lower.y));

                        // STSM: stmatrix.sync.aligned.m16n8.x1.trans, 16行×8列 FP8 → smem
                        // col ^ (row/2): XOR swizzle 消 bank conflict (与 BF16 epilogue 不同)
                        uint32_t row = lane_idx;       // 0..31, 同一 warp 32 行
                        uint32_t col = warp_idx_in_wg; // 0..3, WG 内 4 个 warp 在 N 维的列段
                        const auto smem_ptr = smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N
                                                                     + i * ATOM_M * L1_OUT_BLOCK_N
                                                                     + row * L1_OUT_BLOCK_N              // 每行 L1_OUT_BLOCK_N 字节
                                                                     + (col ^ (row / 2)) * kNumBankGroupBytes; // XOR swizzle 解交织
                        ptx::SM100_U8x4_STSM_T<__nv_fp8x4_e4m3>::copy(fp8x4_values, smem_ptr);

                    // 4.5.4.7 FP8 量化: cast E4M3 + STSM swizzle 写 smem
                    // 
                    // 4.5.4.8 写入 L2 input SF (UE8M0): 只有偶数 warp 写
                    // 相邻 warp pair (warp0↔warp1) 共享同一 SF, 每 4 lane 写 2 行 (sf.x, sf.y)
                    if (warp_idx_in_wg % 2 == 0 and lane_idx < 4) {
                            const uint32_t k_idx = n_block_idx * 2 + warp_idx_in_wg / 2; 
                            const uint32_t k_uint_idx = k_idx / 4, byte_idx = k_idx % 4;
                            const uint32_t mn_stride = kNumPaddedSFPoolTokens * sizeof(uint32_t);
                            const auto sf_base_ptr = l2_sf_buffer.get_base_ptr<uint8_t>();
                            // NOTES: consecutive tokens (t, t + 1) are in the same 32-group, so `sf_idx` differs by 4
                            // NOTES: originally there was:
                            //   - `const uint32_t token_idx_in_expert = m_block_idx * BLOCK_M + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2
                            //   - `scheduler.get_current_pool_block_offset() * SF_BLOCK_M + transform_sf_token_idx(token_idx_in_expert)`
                            // We find out that
                            //   1. `m_block_idx * BLOCK_M` mod `BLOCK_M` is 0, and `epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M + lane_idx * 2` is always < `BLOCK_M`, so we can put `m_block_idx * BLOCK_M` outside
                            //   2. `lane_idx * 2` controls the lowest 3 bit of `token_idx_in_expert`, and `transform_sf_token_idx` is a bitwise-independent transformation if the input is less than `BLOCK_M`, so we can put `lane_idx * 2` outside
                            // This reduce the number of computation instructions.
                            const uint32_t token_base_idx = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                            __builtin_assume(token_base_idx < BLOCK_M); 
                            const auto sf_pool_token_idx = scheduler.get_current_pool_block_offset() * SF_BLOCK_M
                                + m_block_idx * SF_BLOCK_M + transform_sf_token_idx(token_base_idx) + (lane_idx * 2) * 4;
                            const auto sf_addr = k_uint_idx * mn_stride + sf_pool_token_idx * static_cast<uint32_t>(sizeof(uint32_t)) + byte_idx;
                            sf_base_ptr[sf_addr] =
                                (*reinterpret_cast<const uint32_t*>(&sf.x) >> 23);
                            sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))] =
                                (*reinterpret_cast<const uint32_t*>(&sf.y) >> 23);
                        }
                        __syncwarp();
                    }
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // 4.5.4.9 TMA store L1 output → l2_token_buffer (供 L2 TMA-A 加载)
                    // SwiGLU 后 N 减半: L1_OUT_BLOCK_N = BLOCK_N/2
                    if (warp_idx_in_wg == 0 and cute::elect_one_sync()) {
                        uint32_t out_n_idx = n_block_idx * L1_OUT_BLOCK_N;
                        cute::tma_store_fence();
                        cute::SM90_TMA_STORE_2D::copy(
                            &tensor_map_l1_output,
                            smem_cd[tma_stage_idx] + epilogue_wg_idx * STORE_BLOCK_M * L1_OUT_BLOCK_N,
                            out_n_idx,
                            m_idx + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M);
                        cute::tma_store_arrive();
                    }
                    __syncwarp();
                }

                // 通知 L2: red_or_rel_gpu(l2_arrival_mask) 
                // l2_arrival_mask 是 bitmap, 每 bit 对应一个 N-block
                // TMA-A warp L2 阶段自旋 ld_acq_gpu 等待全 bit 就绪
                // TODO: less epilogue sync scope
                ptx::tma_store_wait<0>();
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                    DG_STATIC_ASSERT(L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N, "L2 shape K is too large");
                    ptx::red_or_rel_gpu(
                        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
                        1ull << n_block_idx
                    );
                }
                __syncwarp();
                
            } else {
                // ============================================================
                // 【L2 Epilogue - 结果写回与Combine】
                //
                // L2层的输出需要通过NVLink发送回原始token位置
                // 与L1不同，L2输出是BF16格式，直接存储到combine buffer
                //
                //  Combine流程:
                //    ┌────────────────────────────────────────────────────────┐
                //    │  L2_output (BF16)                                      │
                //    │       │                                               │
                //    │       │ NVLink send                                   │
                //    │       ▼                                               │
                //    │  ┌─────────────────────────────────────────────┐    │
                //    │  │  Combine Buffer (按token分组)                  │    │
                //    │  │  token[0] <- from expert[5] (topk 1)          │    │
                //    │  │  token[0] <- from expert[2] (topk 2)          │    │
                //    │  │  token[0] <- from expert[8] (topk 3)          │    │
                //    │  │  ...                                          │    │
                //    │  │  token[1] <- from expert[5] (topk 1)          │    │
                //    │  │  token[1] <- from expert[8] (topk 2)          │    │
                //    │  └─────────────────────────────────────────────┘    │
                //    │                     │                               │
                //    │                     ▼                               │
                //    │  ┌─────────────────────────────────────────────┐ │
                //    │  │  Weighted Sum (topk_weights * L2_output)        │ │
                //    │  │  y = w1*out1 + w2*out2 + ...                  │ │
                //    │  └─────────────────────────────────────────────┘ │
                //    └────────────────────────────────────────────────────────┘
                //
                DG_STATIC_ASSERT(STORE_BLOCK_M % 8 == 0, "Invalid store M");
                constexpr uint32_t kNumRowsPerWarp = STORE_BLOCK_M / 8;

                // L2 BF16 epilogue: write GEMM output to remote combine buffer via NVLink
                #pragma unroll
                for (uint32_t s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++ s) {
                    // Early break if the entire store block is beyond the valid token range
                    // TODO: check performance
                    if (epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M >= valid_m) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        break;
                    }

                    #pragma unroll
                    for (uint32_t i = 0; i < STORE_BLOCK_M / ATOM_M; ++ i) {
                        // Load from TMEM using .16x256b shape to satisfy STSM layout requirements
                        // Start from lane index 0 and 16
                        uint32_t tmem_addr = accum_stage_idx * UMMA_N + epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + i * ATOM_M;
                        uint32_t values[ATOM_M];
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                               values[0], values[1], values[2], values[3]);
                        cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                               values[4], values[5], values[6], values[7]);
                        cutlass::arch::fence_view_async_tmem_load();

                        // Wait shared memory release from previous NVLink store
                        // NOTES: skip for the first store block since the prior full barrier already ensures completion
                        if (i == 0 and s > 0)
                            ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                        // Signal tensor memory consumed
                        if (s == WG_BLOCK_M / STORE_BLOCK_M - 1 and i == STORE_BLOCK_M / ATOM_M - 1) {
                            ptx::tcgen05_before_thread_sync();
                            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                        }

                        // 4.5.5.2 转换为 BF16 并保存到 SMEM
                        // 2 个 warp 共享一个 BF16 swizzle atom (128B)
                        // col ^ row: BF16 版本的 XOR swizzle 消 bank conflict
                        uint32_t row = lane_idx % 8;
                        uint32_t col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (warp_idx_in_wg / 2) * STORE_BLOCK_M * kSwizzleCDMode +
                            i * ATOM_M * kSwizzleCDMode +
                            row * (kNumBankGroupBytes * 8) +
                            (col ^ row) * kNumBankGroupBytes;
                        ptx::SM90_U32x4_STSM_T<uint32_t>::copy(
                            math::cast_into_bf16_and_pack(values[0], values[1]),
                            math::cast_into_bf16_and_pack(values[2], values[3]),
                            math::cast_into_bf16_and_pack(values[4], values[5]),
                            math::cast_into_bf16_and_pack(values[6], values[7]),
                            smem_ptr
                        );
                    }

                    // Wait shared memory ready
                    ptx::sync_aligned(128, kEpilogueWGBarrierStartIdx + epilogue_wg_idx);

                    // 4.5.5.3 NVLink 远端写: 每 warp 独占整行, 16 lane 分 16 个 float4
                    // 通过 sym_buffer.map 跨 rank 写入 combine_token_buffer[dst_topk][dst_token]
                    // 读取 dispatch 阶段写入的 TokenSrcMetadata 确定目标路由
                    const uint32_t row_in_atom = (warp_idx_in_wg * 2 + lane_idx / 16) % ATOM_M;
                    const uint32_t bank_group_idx = lane_idx % 8;

                    #pragma unroll
                    for (uint32_t j = 0; j < kNumRowsPerWarp; ++ j) {
                        const uint32_t row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx / 16;
                        const uint32_t m_idx_in_block = epilogue_wg_idx * WG_BLOCK_M + s * STORE_BLOCK_M + row_in_store;

                        // Skip padding rows beyond the actual token count for this expert
                        if (m_idx_in_block >= valid_m)
                            break;

                        const auto src_metadata = *workspace.get_token_src_metadata_ptr(m_idx + m_idx_in_block);
                        const uint32_t dst_rank_idx = src_metadata.rank_idx;
                        const uint32_t dst_token_idx = src_metadata.token_idx;
                        const uint32_t dst_topk_idx = src_metadata.topk_idx;

                        // Read from shared memory
                        const auto smem_ptr = smem_cd_l2 +
                            epilogue_wg_idx * STORE_BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(nv_bfloat16)) +
                            (lane_idx % 16 / 8) * STORE_BLOCK_M * kSwizzleCDMode +
                            row_in_store * kSwizzleCDMode +
                            (bank_group_idx ^ row_in_atom) * kNumBankGroupBytes;
                        const auto packed = ptx::ld_shared(reinterpret_cast<float4*>(smem_ptr));

                        // Write into remote
                        const auto dst_token = combine_token_buffer.get_rank_buffer(dst_topk_idx)
                                               .get_data_buffer(dst_token_idx);
                        const auto dst_ptr = math::advance_ptr<float4>(
                            dst_token.get_base_ptr(),
                            n_idx * static_cast<uint32_t>(sizeof(nv_bfloat16)) + (lane_idx % 16) * static_cast<uint32_t>(sizeof(float4)));
                        *sym_buffer.map(dst_ptr, dst_rank_idx) = packed;
                    }
                }

                // Ensure the next epilogue safe to use shared memory
                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        });

        // 4.5.6 Combine 阶段 — top-k BF16 归约, 写出最终 y[num_tokens, hidden]
        //
        // 把各 rank 经 NVLink 写入 combine_token_buffer 的带权 BF16 部分和
        // 按 token 做 top-k reduction → BF16 → TMA store 最终输出
        //
        // Deallocate tensor memory: 两个 CTA 必须由相同的逻辑 warp ID 调用
        if (epilogue_warp_idx == 0)
            Allocator().free(0, kNumTmemCols);

        // NVLink barrier: 等待所有 rank 的 L2 输出已写入各 combine buffer (~4us)
        // grid_sync → 跨 rank NVLink barrier → 再次 grid_sync
        comm::nvlink_barrier<kNumRanks, kNumSMs, kNumEpilogueThreads,
                             kEpilogueGridSyncIndex, kBeforeCombineReduceBarrierTag>(
            workspace, sym_buffer, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); }
        );

        // 与 dispatch warp 再次同步: combine 需要占用 SMEM (smem_buffer→barrier 区间)
        // dispatch 此刻可以安全清理 workspace, 两者 overlap
        ptx::sync_unaligned(kNumDispatchThreads + kNumEpilogueThreads, kDispatchWithEpilogueBarrierIdx);

        // 4.5.6.2/4.5.6.3 工作切分与内存布局
        // 每个 warp 处理 1 个 token, hidden 维切 1 或 2 个 chunk (受 SMEM 约束)
        // 3 个 chunk 槽: 2 个 TMA load 双缓冲 + 1 个 store
        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
        constexpr uint32_t kNumElemsPerUint4 = sizeof(uint4) / sizeof(nv_bfloat162);
        constexpr uint32_t kNumChunkSlots = 3;
        constexpr uint32_t kNumMaxRegistersForBuffer = 128;
        // kNumChunks: 1 或 2, 受 SMEM + 寄存器双重约束
        constexpr uint32_t kNumChunks =
            kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes <= SMEM_BEFORE_BARRIER_SIZE and kHidden <= 32 * kNumMaxRegistersForBuffer ? 1 : 2;
        constexpr uint32_t kNumChunkBytes = kNumHiddenBytes / kNumChunks;
        constexpr uint32_t kNumChunkUint4 = kNumChunkBytes / sizeof(uint4);
        constexpr uint32_t kNumUint4PerLane = kNumChunkUint4 / 32; // 每个 lane 的 uint4 数
        DG_STATIC_ASSERT(kHidden % kNumChunks == 0, "Hidden must be divisible by number of chunks");
        DG_STATIC_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumHiddenBytes / kNumChunks <= SMEM_BEFORE_BARRIER_SIZE, "Hidden is too large");
        DG_STATIC_ASSERT(kNumChunkBytes % 16 == 0, "Combine chunk must be TMA-aligned (16 bytes)");
        DG_STATIC_ASSERT(kNumChunkBytes % sizeof(uint4) == 0, "Combine chunk must be divisible by 16 bytes");
        DG_STATIC_ASSERT(kNumChunkUint4 % 32 == 0, "Combine chunk must be a multiple of 32 16-byte elements (one per lane)");
        DG_STATIC_ASSERT(kNumTopk <= 32, "Top-k must fit in a single warp");

        // Verify combined shared memory budget at runtime
        DG_DEVICE_ASSERT(kNumChunkSlots * kNumEpilogueWarps * kNumChunkBytes <= static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer));

        // 4.5.6.3 SMEM 分布: 外层 slot、内层 warp 的 stripe pattern
        // 每个 warp 3 个 chunk 槽: 槽 0/1 双缓冲 load, 槽 2 store
        const auto combine_load_buffer = utils::PatternVisitor([&](const uint32_t& i) {
            return math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + i * kNumEpilogueWarps) * kNumChunkBytes);
        });
        const auto combine_store_buffer  = math::advance_ptr<uint4>(smem_buffer, (epilogue_warp_idx + kNumEpilogueWarps * 2) * kNumChunkBytes);

        // 每个 warp 2 个 mbarrier (对应 load slot 0/1)
        auto combine_load_barriers = utils::PatternVisitor([&](const uint32_t& i) {
            return combine_barriers[i + epilogue_warp_idx * 2];
        });

        // 4.5.6.4 主循环: 按 warp 粒度遍历所有 token, 步长 = kNumSMs * kNumEpilogueWarps
        uint32_t combine_phase = 0;
        uint32_t load_stage_idx = 0;
        for (uint32_t token_idx = sm_idx * kNumEpilogueWarps + epilogue_warp_idx;
             token_idx < num_tokens;
             token_idx += kNumSMs * kNumEpilogueWarps) {
            // 每个 lane 读一个 topk 槽 (-1 表示未使用)
            DG_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of topk");
            const int stored_topk_slot_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + token_idx * kNumTopk + lane_idx)) : -1;
            const uint32_t total_mask = __ballot_sync(0xffffffff, stored_topk_slot_idx >= 0); // 32 lane valid 位编织成 mask

            // 4.5.6.5 Chunk 循环: 每个 chunk 执行三段 — 预取→累加→写出, ping-pong 流水
            for (uint32_t chunk = 0; chunk < kNumChunks; ++ chunk) {
                const uint32_t chunk_byte_offset = chunk * kNumChunkBytes;

                // 4.5.6.6 move_mask_and_load: 用 __ffs(mask)-1 按位遍历 valid top-k 槽
                // mask 中逐 bit 消费, elect_one 选 1 个 lane 发 TMA 1D load
                uint32_t mask = total_mask;
                const auto move_mask_and_load = [&](const uint32_t& i) {
                    if (mask) {
                        const uint32_t slot_idx = __ffs(mask) - 1;   // 取最低 bit 对应 rank
                        mask ^= 1 << slot_idx;                       // 从 mask 移除
                        if (cute::elect_one_sync()) {                // 选 1 个 lane 发起 TMA
                            const auto src_ptr = math::advance_ptr<uint8_t>(
                                combine_token_buffer.get_rank_buffer(slot_idx)
                                                    .get_data_buffer(token_idx).get_base_ptr(),
                                chunk_byte_offset);
                            ptx::tma_load_1d(combine_load_buffer[i], src_ptr, combine_load_barriers[i], kNumChunkBytes);
                            ptx::mbarrier_arrive_and_set_tx(combine_load_barriers[i], kNumChunkBytes);
                        }
                        __syncwarp();
                        return true;
                    }
                    return false;
                };

                // 启动第一次 TMA load
                bool do_reduce = move_mask_and_load(load_stage_idx);

                // 4.5.6.7 累加循环: Ping-Pong 预取 pattern
                // 当前 stage 累加时, 先把下一 stage 的 TMA 发出去
                // ptx::accumulate: BF16→FP32 累加, 避免精度损失
                float2 reduced[kNumUint4PerLane * kNumElemsPerUint4] = {};
                while (do_reduce) {
                    // 预取下一个 top-k 到另一个 stage (while 当前 stage 正在累加)
                    do_reduce = move_mask_and_load(load_stage_idx ^ 1);

                    // 等待当前 stage TMA 完成 → BF16→FP32 累加
                    combine_load_barriers[load_stage_idx]->wait(combine_phase);
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                        const auto uint4_values = combine_load_buffer[load_stage_idx][j * 32 + lane_idx];
                        const auto bf16_values = reinterpret_cast<const nv_bfloat162*>(&uint4_values);
                        #pragma unroll
                        for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                            ptx::accumulate(reduced[j * kNumElemsPerUint4 + l], bf16_values[l]);
                    }
                    combine_phase ^= load_stage_idx; // stage 翻一整圈才翻 phase
                    load_stage_idx ^= 1;
                }

                // 4.5.6.8 Cast BF16 + Store: FP32 累加结果→BF16→写 SMEM→TMA store
                #pragma unroll
                for (uint32_t j = 0; j < kNumUint4PerLane; ++ j) {
                    uint4 casted;
                    auto casted_bf16 = reinterpret_cast<nv_bfloat162*>(&casted);
                    #pragma unroll
                    for (uint32_t l = 0; l < kNumElemsPerUint4; ++ l)
                        casted_bf16[l] = __float22bfloat162_rn(reduced[j * kNumElemsPerUint4 + l]);

                    // 等待上一轮 TMA store 完成, 写 SMEM combine_store_buffer
                    if (j == 0) {
                        ptx::tma_store_wait<0>();
                        __syncwarp();
                    }
                    ptx::st_shared(combine_store_buffer + j * 32 + lane_idx,
                                   casted.x, casted.y, casted.z, casted.w);
                }
                __syncwarp();

                // TMA 1D store: 将本 chunk 写入 y[token_idx][chunk_byte_offset:]
                if (cute::elect_one_sync()) {
                    cute::tma_store_fence();
                    ptx::tma_store_1d(
                        math::advance_ptr(y, static_cast<uint64_t>(token_idx) * kNumHiddenBytes + chunk_byte_offset),
                        combine_store_buffer, kNumChunkBytes);
                    cute::tma_store_arrive();
                }
                __syncwarp();
            }
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

} // namespace deep_gemm
