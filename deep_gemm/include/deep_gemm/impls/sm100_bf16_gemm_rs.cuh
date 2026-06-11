#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

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
//  sm100_bf16_gemm_rs_impl —— BF16 GEMM + Reduce-Scatter (Push-based, Pipelined)
// ============================================================================================
//
//  【设计思想 — Optimized Warp 编排 (iter 15)】
//
//  Blackwell (SM100) Warp 编排 (优化版):
//
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │  Comm Warp (W0, 32T, 48 regs/thread):                                │
//  │    Push-based Reduce-Scatter:                                         │
//  │    - Poll LOCAL ready flags (all ranks' data pushed into our buffer) │
//  │    - Vectorized LOCAL HBM loads + FP32 reduce + write output         │
//  │    - 1 warp sufficient: memory-bound, not thread-bound               │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Load Warp (W1, 32T, 40 regs):                                       │
//  │    TMA multicast load A+B tiles → smem (2-CTA 共享)                  │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Reserved (W2, 32T, 40 regs):                                        │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  MMA Issue Warp (W3, 32T, 40 regs):                                  │
//  │    单 warp 发射 UMMA FMA (Blackwell 架构: 1 warp 驱动 Tensor Core)   │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Reserved (W4, 32T, 40 regs):                                        │
//  │    TMEM Allocator                                                    │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Epilogue Warps (W5~W8, 128T, 208 regs/thread):                      │
//  │    TMEM → smem → NVLink push to remote partial buffer                │
//  │    + per-tile ready flag signaling (st_rel_sys to remote)            │
//  └──────────────────────────────────────────────────────────────────────┘
//
//  寄存器预算 (SM100 Max = 64512):
//    48 × 32 (comm) + 40 × 128 (non-epi) + 208 × 128 (epilogue)
//    = 1536 + 5120 + 26624 = 33280  ← 充裕!
//
//  【关键改进 vs 之前版本 (384T)】
//
//  1. Comm warps 从 128T→32T: 节省 3 warp slots
//  2. 总线程 384T→288T (12 warps→9 warps): 减少 warp scheduler 压力
//  3. launch_bounds(288) → 编译器初始 reg/thread = 224 (vs 168 for 384T)
//  4. 更高的寄存器预算 → 减少 register spilling → GEMM pipeline 更高效
//
//  【TMA Multicast = 2 (2-CTA Cluster)】
//
//  A 矩阵从 HBM 只读一次，multicast 到 2 个 SM 的 smem。
//  等效 HBM 读带宽翻倍（对 compute-bound tiles 尤其有效）。
//
// ============================================================================================

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumStages,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          bool kSwapAB, bool kWithAccumulation,
          uint32_t kNumCommThreads,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumSMs, uint32_t kNumRanks,
          typename cd_dtype_t,
          typename comm_dtype_t = cd_dtype_t>
__global__ void __launch_bounds__(kNumCommThreads + kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
sm100_bf16_gemm_rs_impl(const uint32_t shape_m_per_rank,
                           const uint32_t runtime_m_per_rank,
                           const uint32_t shape_n,
                           const uint32_t shape_k,
                           cd_dtype_t* __restrict__ output,
                           const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                           const __grid_constant__ cute::TmaDescriptor tensor_map_b) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 2, cute::TMEM::Allocator2Sm, cute::TMEM::Allocator1Sm>;
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
    constexpr uint32_t kNumThreads = kNumCommThreads + kNumNonEpilogueThreads + kNumEpilogueThreads;
    constexpr uint32_t kNumEpilogueStages = 2;

    // Warp layout:
    //   W0..W3: Comm (128T) — poll flags + vectorized reduce + write output
    //   W4: TMA Load (A+B unified), W5: Reserved, W6: MMA Issue, W7: Reserved/TMEM Alloc
    //   W8-W11: Epilogue (128T = 4 warps)
    constexpr uint32_t kNumCommWarps = kNumCommThreads / 32;   // 4
    constexpr uint32_t kNumNonEpiWarps = kNumNonEpilogueThreads / 32;  // 4 warps
    constexpr uint32_t kNumEpiWarps = kNumEpilogueThreads / 32;        // 4 warps
    constexpr uint32_t kLoadWarpIdx = kNumCommWarps;         // W4: unified TMA load (A+B)
    constexpr uint32_t kMMAWarpIdx = kNumCommWarps + 2;      // W6
    constexpr uint32_t kReservedWarpIdx = kNumCommWarps + 3; // W7
    constexpr uint32_t kEpilogueWarpStart = kNumCommWarps + kNumNonEpiWarps;  // W8

    // Comm warp pipeline stages for TMA fetch
    constexpr uint32_t kNumCommFetchStages = 2;

    DG_STATIC_ASSERT(BLOCK_K == 64, "Invalid block K for BF16");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT(kNumNonEpilogueThreads == 128, "Non-epilogue must be 128 threads (4 warps)");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(comm_dtype_t);
    // Use all epilogue threads for barrier sync (safe even when STORE_BLOCK_M < kNumEpilogueThreads)
    constexpr uint32_t kNumUMMAStoreThreads = kNumEpilogueThreads;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // smem CD stage sized for comm_dtype_t (what gets written to local output buffer)
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(ab_dtype_t);
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols>();

    // Comm warp smem: currently unused (comm warps do direct P2P global loads)
    // Kept as zero to maximize pipeline stages for TMA A/B loads.
    constexpr uint32_t SMEM_COMM_SIZE_PER_STAGE = 0;
    constexpr uint32_t SMEM_COMM_SIZE = 0;

    // Register budget validation (MegaMoE style)
    // 48×128 + 40×128 + 208×128 = 6144 + 5120 + 26624 = 37888 (< 64512)
    constexpr uint32_t kNumCommRegisters = 48;
    constexpr uint32_t kNumNonEpiRegisters = 40;
    constexpr uint32_t kNumEpiRegisters = 208;
    DG_STATIC_ASSERT(kNumCommRegisters * kNumCommThreads +
                     kNumNonEpiRegisters * kNumNonEpilogueThreads +
                     kNumEpiRegisters * kNumEpilogueThreads <= 64512,
                     "Too many registers");

    // ── 运行时变量 ──
    const uint32_t shape_m = runtime_m_per_rank * kNumRanks;
    const uint32_t num_m_blocks_per_rank = ceil_div(runtime_m_per_rank, BLOCK_M);
    const uint32_t num_m_blocks = num_m_blocks_per_rank * kNumRanks;
    const uint32_t num_n_blocks = ceil_div(shape_n, BLOCK_N);
    const uint32_t num_n_slices = BLOCK_N / STORE_BLOCK_N;
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const uint32_t cta_rank = cute::block_rank_in_cluster();  // 0 or 1 within cluster
    // For multicast=2 (2-CTA cluster): each cluster processes TWO adjacent M-tiles.
    // The scheduler assigns pairs of M-tiles to clusters. Each CTA in the cluster
    // handles one M-tile (CTA0 → even, CTA1 → odd) but they share the same N-tile.
    // This matches standard GEMM's 2-CTA behavior where each CTA loads different A rows
    // and the 2SM UMMA computes UMMA_M=256 (128 rows per CTA).
    // For multicast=1: cluster_idx == blockIdx.x, kNumClusters == kNumSMs.
    constexpr uint32_t kNumClusters = kNumSMs / kNumMulticast;
    const uint32_t cluster_idx = blockIdx.x / kNumMulticast;
    const uint32_t sm_idx = cluster_idx;  // Used for persistent scheduling stride
    const uint32_t thread_idx = threadIdx.x;
    const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
    const uint32_t lane_idx = ptx::get_lane_idx();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto workspace = layout::GemmRSWorkspace(
        sym_buffer.get_base_ptr(), kNumRanks, shape_m_per_rank, shape_n, sizeof(comm_dtype_t), BLOCK_M, BLOCK_N);

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? cute::cluster_sync() : void();

    // ── Prefetch TMA descriptors ──
    if (warp_idx == kLoadWarpIdx) {
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
    //   [+barriers, +SMEM_COMM_SIZE)                         : Comm warp TMA fetch buffer
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

    // Comm warp TMA fetch buffer (after barriers + tmem_ptr)
    auto smem_comm_base = smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE)
                          + (kNumStages * 2 + kNumEpilogueStages * 2) * sizeof(Barrier) + sizeof(uint32_t);
    // Align to 128 bytes for TMA
    smem_comm_base = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(smem_comm_base) + 127) & ~127ull);

    auto smem_comm = utils::PatternVisitor([=](const uint32_t& i) {
        return reinterpret_cast<comm_dtype_t*>(smem_comm_base + i * SMEM_COMM_SIZE_PER_STAGE);
    });

    // ── Initialize barriers ──
    if (warp_idx == kMMAWarpIdx and cute::elect_one_sync()) {
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
    } else if (warp_idx == kReservedWarpIdx) {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    // ── Initial NVLink barrier: ensure previous iteration's data is consumed ──
    // Note: nvlink_barrier / grid_sync need unique per-block identity (blockIdx.x),
    // NOT cluster_idx, since all kNumSMs blocks must independently participate.
    constexpr uint32_t kInitBarrierTag = 41;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kInitBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

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
    //    chunk for rank i (self) last  → self doesn't need remote pull for own contribution
    //
    //  For multicast=2 (2-CTA cluster, kIsMulticastOnA=false, cluster_m=2):
    //    The scheduler assigns M-tile PAIRS to clusters. Within each cluster:
    //    - CTA0 (block_rank=0): gets the base m_block_idx (even tile in the pair)
    //    - CTA1 (block_rank=1): gets m_block_idx + 1 (odd tile in the pair)
    //    Both CTAs share the same n_block_idx.
    //    This ensures each CTA loads DIFFERENT A rows, matching 2SM UMMA requirements.
    //    The total number of M-tile pairs = num_m_blocks_per_rank / 2 (must be even).
    //
    //  For multicast=1: each CTA handles one tile independently.
    //
    auto get_next_block = [&](uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx, uint32_t& iter_idx) {
        // In multicast=2 mode, each cluster handles 2 adjacent M-tiles as a pair.
        // The block_idx enumerates "cluster-level work units" (pairs for mc=2, singles for mc=1).
        const uint32_t m_blocks_per_cluster = kNumMulticast;  // 2 for mc=2, 1 for mc=1
        const uint32_t num_m_pairs_per_rank = num_m_blocks_per_rank / m_blocks_per_cluster;
        const uint32_t tiles_per_rank = num_m_pairs_per_rank * num_n_blocks;
        const uint32_t total_cluster_tiles = tiles_per_rank * kNumRanks;

        if (block_idx >= total_cluster_tiles)
            return false;

        // ── Round-robin interleaved scheduling ──
        // Instead of computing ALL tiles for rank (i+1) then ALL for rank (i+2)...
        // we interleave: tile 0→rank(i+1), tile 1→rank(i+2), tile 2→rank(i+3), ...
        // This ensures all remote ranks get their first tiles quickly,
        // reducing Comm warp tail latency (no rank starves waiting for pushes).
        //
        // block_idx advances by kNumClusters (persistent stride).
        // For each block_idx, compute which dst_rank and which local tile within that rank.
        const uint32_t local_tile_idx = block_idx / kNumRanks;
        const uint32_t rank_offset = block_idx % kNumRanks;

        // dst_rank: round-robin starting from rank (i+1), with self last
        const uint32_t dst_rank = (rank_offset + 1 < kNumRanks) ?
            (rank_idx + rank_offset + 1) % kNumRanks : rank_idx;

        const uint32_t local_m_pair_idx = local_tile_idx / num_n_blocks;
        n_block_idx = local_tile_idx - local_m_pair_idx * num_n_blocks;

        // Each CTA in the cluster handles a different M-tile within the pair
        const uint32_t local_m_block_idx = local_m_pair_idx * m_blocks_per_cluster + cta_rank;
        m_block_idx = dst_rank * num_m_blocks_per_rank + local_m_block_idx;

        // Stride by kNumClusters
        block_idx += kNumClusters;
        ++ iter_idx;
        return true;
    };

    // ════════════════════════════════════════════════════════════════
    //  Warp 0~3 (Comm Warps, 128T): Push-based All-Ranks-Ready Reduce
    // ════════════════════════════════════════════════════════════════
    //
    //  Design:
    //    - Phase 1: Wait ALL ranks' ready flags (distributed polling across warps)
    //    - Phase 2: All 128 threads load from ALL ranks + FP32 accumulate in regs
    //              + write output ONCE (no intermediate HBM storage)
    //
    if (warp_idx < kNumCommWarps) {
        // Adjust registers (48 regs for comm warps)
        cutlass::arch::warpgroup_reg_dealloc<kNumCommRegisters>();

        const uint32_t comm_warp_local_idx = warp_idx;
        const uint32_t comm_thread_local_idx = comm_warp_local_idx * 32 + lane_idx;

        // My chunk: tiles belonging to rank_idx
        const uint32_t total_my_tiles = num_m_blocks_per_rank * num_n_blocks;

        // Vectorized operations: 16 bytes at a time
        constexpr uint32_t kVecBytes = 16;
        constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // 8 for bf16, 4 for fp32

        // Elements per tile
        const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;
        const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

        // For multicast=2, use both CTAs' comm warps for better parallelism
        const uint32_t comm_cta_offset = kNumMulticast > 1 ? cute::block_rank_in_cluster() : 0;
        const uint32_t comm_cta_stride = kNumMulticast > 1 ? (kNumClusters * kNumMulticast) : kNumClusters;
        for (uint32_t tile_idx = sm_idx * kNumMulticast + comm_cta_offset; tile_idx < total_my_tiles; tile_idx += comm_cta_stride) {
            const uint32_t my_m_block = tile_idx / num_n_blocks;
            const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;

            const uint32_t base_row = my_m_block * BLOCK_M;
            const uint32_t base_col = my_n_block * BLOCK_N;

            // ── Phase 1: Wait for ALL ranks' ready flags ──
            // Including self-rank (ensures Epilogue finished writing output before Comm reads it)
            if (lane_idx == 0) {
                for (uint32_t rank_iter = comm_warp_local_idx; rank_iter < kNumRanks; rank_iter += kNumCommWarps) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;
                    auto* poll_ptr = workspace.get_ready_ptr(src_rank, my_m_block, my_n_block);

                    constexpr int64_t kTimeoutCycles = 30ll * 2000000000ll;
                    const auto start_clock = clock64();
                    while (ptx::ld_acq_sys(poll_ptr) == 0u) {
                        if (clock64() - start_clock >= kTimeoutCycles) {
                            printf("GEMM-RS comm timeout: rank=%d, src=%d, tile=(%d,%d)\n",
                                   rank_idx, src_rank, my_m_block, my_n_block);
                            DG_DEVICE_ASSERT(false and "Comm warp ready flag timeout");
                        }
                    }
                }
            }
            // All comm threads sync after polling completes
            cutlass::arch::NamedBarrier::sync(kNumCommThreads, 2);

            // ── Phase 2: Reduce N-1 remote ranks + existing output (self-rank) ──
            // Output already contains self-rank's contribution (written by Epilogue).
            // Load output value + accumulate N-1 remote partials → write back.
            for (uint32_t vec_offset = comm_thread_local_idx; vec_offset < vecs_per_tile; vec_offset += kNumCommThreads) {
                const uint32_t elem_offset = vec_offset * kVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;

                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;

                if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                    continue;

                // Load self-rank's contribution from output (already there from Epilogue)
                auto* out_ptr = output + global_row * shape_n + global_col;
                uint4 self_data = *reinterpret_cast<const uint4*>(out_ptr);

                float acc[kVecSize];
                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    const auto* self_bf16 = reinterpret_cast<const cd_dtype_t*>(&self_data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] = static_cast<float>(self_bf16[i]);
                } else {
                    const auto* self_f32 = reinterpret_cast<const float*>(&self_data);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] = self_f32[i];
                }

                // Accumulate N-1 remote ranks' contributions from local partial buffer
                #pragma unroll 1
                for (uint32_t rank_iter = 0; rank_iter < kNumRanks - 1; ++ rank_iter) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;

                    const comm_dtype_t* partial_ptr =
                        workspace.get_partial_ptr<comm_dtype_t>(src_rank, global_row, global_col);

                    uint4 data = *reinterpret_cast<const uint4*>(partial_ptr);
                    const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);

                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += static_cast<float>(comm_data[i]);
                }

                // Write final reduced result to output
                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    uint4 result;
                    auto* out_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        out_bf16[i] = cd_dtype_t(acc[i]);
                    *reinterpret_cast<uint4*>(out_ptr) = result;
                } else {
                    uint4 result;
                    auto* out_f32 = reinterpret_cast<float*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        out_f32[i] = acc[i];
                    *reinterpret_cast<uint4*>(out_ptr) = result;
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 4 (TMA Load Warp): Load both A and B into smem
    //  Aligned with standard GEMM: single warp issues both TMA loads
    //  and does a single arrive_and_expect_tx per stage.
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == kLoadWarpIdx and cute::elect_one_sync()) {
        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
        while (get_next_block(block_idx, m_block_idx, n_block_idx, iter_idx)) {
            const uint32_t global_m = m_block_idx * BLOCK_M;
            const uint32_t n_idx = n_block_idx * BLOCK_N;
            const uint32_t num_total_k_blocks = ceil_div(shape_k, BLOCK_K);

            // Each CTA already has its own m_block_idx (from scheduler), so:
            // - For A: use global_m directly (each CTA loads its own 128 A rows)
            // - For B: split across CTAs with block_rank offset (multicast ensures both have full B)
            // When kIsMulticastOnA=false: A is NOT split, B is split (and multicast fills both)
            // When kIsMulticastOnA=true: A is split (and multicast fills both), B is NOT split
            uint32_t load_m_idx = global_m;
            uint32_t load_n_idx = n_idx;
            if constexpr (kNumMulticast > 1) {
                // No M offset needed: each CTA's m_block_idx already differs
                // B split: each CTA loads half of B columns
                load_n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);
                const uint32_t k_idx = k_block_idx * BLOCK_K;

                // Issue TMA load A
                tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, ab_dtype_t>(
                    &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_idx, load_m_idx, kNumMulticast);

                // Issue TMA load B
                tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, ab_dtype_t>(
                    &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_idx, load_n_idx, kNumMulticast);

                // Single arrive with total A+B byte count (same as standard GEMM)
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    }

    // Warp 5: Reserved (no-op, matching standard GEMM's warp layout)
    else if (warp_idx == (kNumCommWarps + 1)) {
        // Intentionally empty — this warp slot is reserved for future use
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 6 (MMA Issue Warp): Execute UMMA FMA → TMEM accumulator
    // ════════════════════════════════════════════════════════════════
    //
    //  Blackwell: single warp issues UMMA instructions.
    //  This is THE architectural design — 1 warp drives the entire Tensor Core.
    //
    else if (warp_idx == kMMAWarpIdx and is_leader_cta) {
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
            constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
            if constexpr (kNumMulticast == 1) {
                cutlass::arch::umma_arrive(barrier);
            } else {
                cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
            }
        };

        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
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
    //  Warp 7 (Reserved): keeps non-epilogue alignment
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == kReservedWarpIdx) {
        // Reserved warp — does nothing but TMEM allocation (done above)
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 8~11 (Epilogue Warps, 128T): TMEM → smem → NVLink push to remote
    // ════════════════════════════════════════════════════════════════
    //
    //  Push-based: Epilogue writes to REMOTE rank's partial buffer via NVLink,
    //  then sets ready flag in remote rank's flag array.
    //
    else if (warp_idx >= kEpilogueWarpStart) {
        // Adjust registers (MegaMoE style: 208 regs for epilogue)
        cutlass::arch::warpgroup_reg_alloc<kNumEpiRegisters>();

        const auto epilogue_warp_idx = warp_idx - kEpilogueWarpStart;
        const uint32_t epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx;

        constexpr uint32_t kElemsPerStore = 16 / sizeof(comm_dtype_t);
        constexpr uint32_t kRowBytesPerNSlice = STORE_BLOCK_N * sizeof(comm_dtype_t);
        constexpr uint32_t kStoresPerRow = STORE_BLOCK_N / kElemsPerStore;
        constexpr uint32_t kNumNSlices = BLOCK_N / STORE_BLOCK_N;

        uint32_t tma_stage_idx = 0;

        uint32_t block_idx = cluster_idx, iter_idx = 0, m_block_idx, n_block_idx;
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
            // In multicast=2: each CTA has its OWN m_block_idx (from scheduler) and reads
            // its own SM's TMEM (which holds that CTA's portion of the 2SM UMMA result).
            // Both CTAs independently write to their respective partial buffer slots.
            // No race condition because they write different m_block addresses.
            // Both CTAs must arrive at tmem_empty_barriers for MMA pipeline progress.
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumNSlices; ++ s) {
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

                    // Sync before reusing smem CD buffer (ensure prior global stores completed)
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Phase 1: TMEM → registers → smem
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

                    // Phase 2: smem → global store (vectorized, parallel)
                    // For self-rank tiles (dst_rank == rank_idx): write directly to OUTPUT
                    // For remote tiles: push to remote rank's partial buffer via NVLink
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    {
                        uint32_t base_row = local_m + w * STORE_BLOCK_M;
                        uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;
                        constexpr uint32_t kVecElems = 16 / sizeof(comm_dtype_t);
                        constexpr uint32_t kVecsPerRow = STORE_BLOCK_N / kVecElems;
                        constexpr uint32_t kTotalVecs = STORE_BLOCK_M * kVecsPerRow;

                        if (dst_rank == rank_idx) {
                            // Self-rank: write directly to output (no NVLink, no partial buffer!)
                            for (uint32_t vid = epilogue_thread_idx; vid < kTotalVecs; vid += kNumUMMAStoreThreads) {
                                const uint32_t row = vid / kVecsPerRow;
                                const uint32_t col_vec = vid - row * kVecsPerRow;

                                auto* src = reinterpret_cast<const uint4*>(
                                    smem_base_ptr + row * kRowBytesPerNSlice + col_vec * 16);
                                uint4 data = *src;

                                const uint32_t global_row = base_row + row;
                                const uint32_t global_col = base_col + col_vec * kVecElems;
                                if (global_row < runtime_m_per_rank and global_col < shape_n) {
                                    *reinterpret_cast<uint4*>(output + global_row * shape_n + global_col) = data;
                                }
                            }
                        } else {
                            // Remote rank: push to remote partial buffer via NVLink
                            for (uint32_t vid = epilogue_thread_idx; vid < kTotalVecs; vid += kNumUMMAStoreThreads) {
                                const uint32_t row = vid / kVecsPerRow;
                                const uint32_t col_vec = vid - row * kVecsPerRow;

                                auto* src = reinterpret_cast<const uint4*>(
                                    smem_base_ptr + row * kRowBytesPerNSlice + col_vec * 16);
                                uint4 data = *src;

                                auto* local_ptr = workspace.get_partial_ptr<comm_dtype_t>(
                                    rank_idx, base_row + row, base_col + col_vec * kVecElems);
                                auto* remote_ptr = sym_buffer.map(local_ptr, dst_rank);
                                *reinterpret_cast<uint4*>(remote_ptr) = data;
                            }
                        }
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

            // ── After all N-slices of this tile are stored, set per-tile ready flag ──
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if (dst_rank != rank_idx) {
                    // Remote tiles: set ready flag in REMOTE rank's flag array via NVLink
                    auto* local_flag_ptr = workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx);
                    auto* remote_flag_ptr = reinterpret_cast<uint32_t*>(sym_buffer.map(local_flag_ptr, dst_rank));
                    ptx::st_rel_sys(remote_flag_ptr, 1u);
                } else {
                    // Self-rank tiles: set LOCAL flag (Comm needs to know output is ready)
                    auto* local_flag_ptr = workspace.get_ready_ptr(rank_idx, local_m_block_idx, n_block_idx);
                    ptx::st_rel_sys(local_flag_ptr, 1u);
                }
            }
        }
    }

    // ── Final synchronization ──
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, static_cast<uint32_t>(blockIdx.x), thread_idx,
        [&]() { __syncthreads(); }, true, true);

    // Reset ready flags for next iteration (all dst_rank slots in our buffer)
    {
        const uint32_t flags_per_slot = num_m_blocks_per_rank * num_n_blocks;
        const uint32_t total_flags = kNumRanks * flags_per_slot;
        for (uint32_t flag_idx = thread_idx; flag_idx < total_flags; flag_idx += kNumThreads) {
            const uint32_t slot = flag_idx / flags_per_slot;
            const uint32_t local_idx = flag_idx - slot * flags_per_slot;
            const uint32_t mb = local_idx / num_n_blocks;
            const uint32_t nb = local_idx - mb * num_n_blocks;
            auto* ready_ptr = workspace.get_ready_ptr(slot, mb, nb);
            *ready_ptr = 0u;
        }
    }

    // Deallocate tensor memory
    if (warp_idx == kLoadWarpIdx)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f");
#endif
}

} // namespace deep_gemm

#pragma clang diagnostic pop
