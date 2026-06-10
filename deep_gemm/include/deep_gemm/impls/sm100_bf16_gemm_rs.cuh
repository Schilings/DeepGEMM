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
//  sm100_bf16_gemm_rs_impl —— BF16 GEMM + Reduce-Scatter (Pull-based, Pipelined per-Rank)
// ============================================================================================
//
//  【设计思想 — MegaMoE Warp 编排 + Flux RS 调度】
//
//  Blackwell (SM100) 原生 Warp 编排:
//
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │  Comm (Dispatch) Warps (W0~W3, 128T, 48 regs/thread):               │
//  │    Pull-based Reduce-Scatter 通信:                                    │
//  │    - 逐 rank 流水: 每个 tile 不等所有 rank，哪个 rank ready 就先 pull │
//  │    - TMA 异步 fetch 远端 tile → smem → FP32 reduce → 写最终输出       │
//  │    - 与 GEMM 完全 overlap: GEMM 算 tile_i 时 comm 处理 tile_{i-k}    │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Load Warp A (W4, 32T, 40 regs):                                     │
//  │    TMA multicast load A tiles → smem (2-CTA 共享)                    │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Load Warp B (W5, 32T, 40 regs):                                     │
//  │    TMA multicast load B tiles → smem (2-CTA 共享)                    │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  MMA Issue Warp (W6, 32T, 40 regs):                                  │
//  │    单 warp 发射 UMMA FMA (Blackwell 架构: 1 warp 驱动 Tensor Core)   │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Reserved Warp (W7, 32T, 40 regs):                                   │
//  │    预留 (保持 non-epilogue 线程对齐)                                  │
//  ├──────────────────────────────────────────────────────────────────────┤
//  │  Epilogue Warps (W8~W11, 128T, 208 regs/thread):                     │
//  │    TMEM → smem → local partial buffer (TMA bulk store)               │
//  │    + per-tile ready flag signaling (fence_system + st_rel)           │
//  └──────────────────────────────────────────────────────────────────────┘
//
//  寄存器预算 (SM100 Max = 64512):
//    48 × 128 (comm) + 40 × 128 (non-epi) + 208 × 128 (epilogue)
//    = 6144 + 5120 + 26624 = 37888  ← 充裕!
//
//  【RS 调度: 逐 Rank 流水 Reduce (学习 Flux)】
//
//  核心区别: 不等 ALL ranks ready! 而是:
//    for each tile in my_chunk:
//      acc = 0  (FP32)
//      for src_rank in ring_order(rank_idx):
//        wait(src_rank's ready flag for this tile)
//        TMA fetch src_rank's partial → comm_smem
//        reduce: acc += comm_smem data
//      store acc → final output
//
//  Ring order 让每个 rank 先 pull 最可能先 ready 的那个 rank (M-Swizzle 对应)
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

    // Warp layout (aligned with standard GEMM):
    //   W0..W3: Comm (Dispatch) = kNumCommThreads / 32 warps
    //   W4: TMA Load (A+B unified), W5: Reserved, W6: MMA Issue, W7: Reserved
    //   W8+: Epilogue
    constexpr uint32_t kNumCommWarps = kNumCommThreads / 32;
    constexpr uint32_t kNumNonEpiWarps = kNumNonEpilogueThreads / 32;  // 4 warps
    constexpr uint32_t kNumEpiWarps = kNumEpilogueThreads / 32;
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

    // Comm warp smem: buffer for TMA fetching remote tiles (per-rank pipeline)
    // One full tile per stage: BLOCK_M × STORE_BLOCK_N elements in comm_dtype_t
    constexpr uint32_t SMEM_COMM_SIZE_PER_STAGE = BLOCK_M * STORE_BLOCK_N * sizeof(comm_dtype_t);
    constexpr uint32_t SMEM_COMM_SIZE = SMEM_COMM_SIZE_PER_STAGE * kNumCommFetchStages;

    // Register budget validation (MegaMoE style)
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
    //    chunk for rank i (self) last  → self doesn't need remote pull for own contribution
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
    //  Warp 0~3 (Comm/Dispatch Warps): Pull-based All-Ranks-Ready Reduce
    // ════════════════════════════════════════════════════════════════
    //
    //  Optimized design:
    //    - Phase 1: Wait ALL ranks' ready flags (single polling loop)
    //    - Phase 2: All threads load from ALL ranks + FP32 accumulate in regs
    //              + write output ONCE (no intermediate HBM storage)
    //
    //  Benefits over per-rank sequential approach:
    //    - Only 1 barrier sync per tile (vs N syncs before)
    //    - Only 1 HBM write per vector (vs N writes before)
    //    - Maximum memory-level parallelism in Phase 2
    //    - Ring order polling: rank(i+1) polled first (most likely ready)
    //
    if (warp_idx < kNumCommWarps) {
        // Adjust registers (MegaMoE style: 48 regs for comm warps)
        cutlass::arch::warpgroup_reg_dealloc<kNumCommRegisters>();

        const uint32_t comm_warp_local_idx = warp_idx;
        const uint32_t comm_thread_local_idx = comm_warp_local_idx * 32 + lane_idx;

        // My chunk: tiles belonging to rank_idx
        const uint32_t total_my_tiles = num_m_blocks_per_rank * num_n_blocks;

        // ════════════════════════════════════════════════════════════════
        //  Optimized Comm Reduce: Per-Rank Pipelined + Write-Once
        //
        //  Key optimization: Eliminate N-1 HBM read+write round-trips.
        //
        //  Original flow (per tile, per rank):
        //    rank 0: load partial → write output
        //    rank 1: load partial + READ output → add → WRITE output
        //    rank 2: load partial + READ output → add → WRITE output  ...
        //    = N P2P loads + (N-1) HBM reads + N HBM writes
        //
        //  Optimized flow (per tile):
        //    for each rank: poll flag → all threads load+accumulate in regs
        //    write output once
        //    = N P2P loads + 0 HBM reads + 1 HBM write
        //
        //  Register-only accumulation:
        //    Each thread processes a FIXED set of vector positions across the tile.
        //    For each position, it accumulates all N ranks' data in FP32 registers.
        //    The barrier sync is per-rank (N syncs total), same as before.
        //    But each thread's work set is fixed — no intermediate HBM storage needed.
        //
        //  Key: We keep rank in the INNER loop per vector position.
        //  This means each thread holds FP32 accumulators for ONE vector (8 floats)
        //  and iterates over all ranks for that vector before moving to the next.
        //  The barrier sync happens BETWEEN vector chunks (not per-vector).
        //
        //  Actually, the cleanest approach: keep rank in OUTER loop (preserving
        //  per-rank pipelining), but write output only ONCE after all ranks.
        //  We need smem-free accumulation — store partial sums back to the
        //  SAME output location. But wait... that's what the original did!
        //
        //  The real optimization: The original reads output + writes output for
        //  EVERY rank after the first. We eliminate the READ by keeping the
        //  running sum in the output itself (write-only after first rank).
        //  Actually... re-reading my own write is fine (L2 cached).
        //
        //  *** TRUE OPTIMIZATION ***:
        //  The real bottleneck is NOT the HBM reads/writes (they're L2 cached
        //  since we just wrote there). The real bottleneck is:
        //  1. Sequential per-rank polling with barrier syncs
        //  2. Not enough in-flight memory operations (low MLP)
        //
        //  NEW DESIGN: Wait for ALL ranks at once, then process with maximum MLP.
        //  - Phase 1: Poll all N ranks' ready flags (warp 0 does all polls)
        //  - Phase 2: All 128 threads process tile with all-ranks interleaved loads
        //             maximizing memory-level parallelism (MLP)
        // ════════════════════════════════════════════════════════════════

        // Vectorized operations: 16 bytes at a time
        constexpr uint32_t kVecBytes = 16;
        constexpr uint32_t kVecSize = kVecBytes / sizeof(comm_dtype_t);  // 8 for bf16, 4 for fp32

        // Elements per tile
        const uint32_t elems_per_tile = BLOCK_M * BLOCK_N;
        const uint32_t vecs_per_tile = elems_per_tile / kVecSize;

        // Iterate over tiles assigned to this SM
        for (uint32_t tile_idx = sm_idx; tile_idx < total_my_tiles; tile_idx += kNumSMs) {
            const uint32_t my_m_block = tile_idx / num_n_blocks;
            const uint32_t my_n_block = tile_idx - my_m_block * num_n_blocks;

            const uint32_t base_row = my_m_block * BLOCK_M;
            const uint32_t base_col = my_n_block * BLOCK_N;

            // ── Phase 1: Wait for ALL ranks' ready flags ──
            // Warp 0, lane 0 polls all N ranks sequentially in ring order.
            // Once all are ready, we can process the entire tile without any
            // further synchronization, maximizing memory-level parallelism.
            if (comm_warp_local_idx == 0 and lane_idx == 0) {
                for (uint32_t rank_iter = 0; rank_iter < kNumRanks; ++ rank_iter) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;
                    auto* local_ready_ptr = workspace.get_ready_ptr(rank_idx, my_m_block, my_n_block);
                    const uint32_t* poll_ptr = (src_rank != rank_idx) ?
                        sym_buffer.map(local_ready_ptr, src_rank) : local_ready_ptr;

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
            // Single barrier sync — all comm threads wait until all flags confirmed
            cutlass::arch::NamedBarrier::sync(kNumCommThreads, 2);

            // ── Phase 2: Reduce all ranks and write output (NO more syncs!) ──
            // Each thread processes its assigned vector positions.
            // For each position: load from all N ranks → FP32 accumulate → write output.
            // No intermediate HBM storage, no barrier syncs within this phase.
            for (uint32_t vec_offset = comm_thread_local_idx; vec_offset < vecs_per_tile; vec_offset += kNumCommThreads) {
                const uint32_t elem_offset = vec_offset * kVecSize;
                const uint32_t tile_row = elem_offset / BLOCK_N;
                const uint32_t tile_col = elem_offset - tile_row * BLOCK_N;

                const uint32_t global_row = base_row + tile_row;
                const uint32_t global_col = base_col + tile_col;

                if (global_row >= runtime_m_per_rank or global_col >= shape_n)
                    continue;

                // FP32 accumulators in registers (8 floats for bf16, 4 for fp32)
                float acc[kVecSize];
                #pragma unroll
                for (uint32_t i = 0; i < kVecSize; ++ i)
                    acc[i] = 0.0f;

                // Accumulate ALL ranks' contributions
                #pragma unroll 1
                for (uint32_t rank_iter = 0; rank_iter < kNumRanks; ++ rank_iter) {
                    const uint32_t src_rank = (rank_idx + 1 + rank_iter) % kNumRanks;

                    // Get pointer to src_rank's partial data for this vector
                    const comm_dtype_t* partial_ptr;
                    if (src_rank == rank_idx) {
                        partial_ptr = workspace.get_partial_ptr<comm_dtype_t>(rank_idx, global_row, global_col);
                    } else {
                        auto* local_ptr = workspace.get_partial_ptr<comm_dtype_t>(rank_idx, global_row, global_col);
                        partial_ptr = sym_buffer.map(local_ptr, src_rank);
                    }

                    // Vectorized P2P load (16 bytes)
                    uint4 data = *reinterpret_cast<const uint4*>(partial_ptr);
                    const auto* comm_data = reinterpret_cast<const comm_dtype_t*>(&data);

                    // FP32 accumulate (all in registers!)
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        acc[i] += static_cast<float>(comm_data[i]);
                }

                // Write final result to output — ONCE per vector position
                if constexpr (cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>) {
                    uint4 result;
                    auto* out_bf16 = reinterpret_cast<cd_dtype_t*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        out_bf16[i] = cd_dtype_t(acc[i]);
                    *reinterpret_cast<uint4*>(output + global_row * shape_n + global_col) = result;
                } else {
                    uint4 result;
                    auto* out_f32 = reinterpret_cast<float*>(&result);
                    #pragma unroll
                    for (uint32_t i = 0; i < kVecSize; ++ i)
                        out_f32[i] = acc[i];
                    *reinterpret_cast<uint4*>(output + global_row * shape_n + global_col) = result;
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
    //  Warp 7 (Reserved): keeps non-epilogue alignment
    // ════════════════════════════════════════════════════════════════
    else if (warp_idx == kReservedWarpIdx) {
        // Reserved warp — does nothing but TMEM allocation (done above)
    }

    // ════════════════════════════════════════════════════════════════
    //  Warp 8~11 (Epilogue Warps): TMEM → smem → local partial buffer + set ready flag
    // ════════════════════════════════════════════════════════════════
    //
    //  Key: We write to LOCAL partial buffer only (our rank's slot),
    //  then set a per-tile ready flag. Comm warps on peer ranks will pull from us.
    //  No cross-rank NVLink writes here — only local memory operations.
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
            // IMPORTANT: Both CTAs in a 2-CTA cluster execute TMEM reads and arrive at
            // tmem_empty_barriers (required for MMA pipeline progress). Only leader CTA
            // performs the actual TMA store to partial buffer and sets the ready flag.
            #pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++ w) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumNSlices; ++ s) {
                    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

                    // Wait previous TMA stores (only leader CTA issues TMA stores)
                    if (is_leader_cta and epilogue_warp_idx == 0)
                        cute::tma_store_wait<kNumTMAStoreStages - 1>();
                    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                    // Phase 1: TMEM → registers → smem (both CTAs do this)
                    // Both CTAs share TMEM in 2x1SM mode, so both can read.
                    // This is required for tmem_empty_barriers to get enough arrivals.
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

                    // Release TMEM stage — BOTH CTAs must arrive to satisfy
                    // tmem_empty_barriers init count of kNumMulticast * kNumUMMAStoreThreads
                    if (w == kNumMWaves - 1 and s == kNumNSlices - 1) {
                        ptx::tcgen05_before_thread_sync();
                        tmem_empty_barriers[accum_stage_idx]->arrive(0u);
                    }

                    // Phase 2: Issue per-row TMA 1D bulk copies to LOCAL partial buffer
                    // IMPORTANT: Only leader CTA writes to avoid race condition with Comm warps.
                    // In 2-CTA mode, both CTAs read same TMEM data. If both write + set flag,
                    // CTA 0 may set flag before CTA 1 finishes writing → Comm reads partial data.
                    if (is_leader_cta) {
                        cute::tma_store_fence();
                        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

                        if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                            uint32_t base_row = local_m + w * STORE_BLOCK_M;
                            uint32_t base_col = n_block_idx * BLOCK_N + s * STORE_BLOCK_N;

                            #pragma unroll 1
                            for (uint32_t row = 0; row < STORE_BLOCK_M; ++ row) {
                                // Write to dst_rank's slot in our partial buffer
                                comm_dtype_t* dst_ptr = workspace.get_partial_ptr<comm_dtype_t>(
                                    dst_rank, base_row + row, base_col);

                                auto* src_ptr = smem_base_ptr + row * kRowBytesPerNSlice;
                                ptx::tma_store_1d(dst_ptr, src_ptr, kRowBytesPerNSlice);
                            }
                            cute::tma_store_arrive();
                        }
                    }

                    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
                }
            }

            // ── After all N-slices of this tile are stored, set per-tile ready flag ──
            // Only leader CTA sets the flag (matches the writer in Phase 2)
            if (is_leader_cta and epilogue_warp_idx == 0) {
                cute::tma_store_wait<0>();
                if (cute::elect_one_sync()) {
                    // Set ready flag: other ranks can now pull this tile from us
                    auto* ready_ptr = workspace.get_ready_ptr(dst_rank, local_m_block_idx, n_block_idx);
                    __threadfence_system();  // Ensure TMA writes are visible across NVLink
                    ptx::st_rel_sys(ready_ptr, 1u);
                }
            }
        }
    }

    // ── Final synchronization ──
    kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

    constexpr uint32_t kFinalBarrierTag = 42;
    comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, 0, kFinalBarrierTag>(
        workspace, sym_buffer, sm_idx, thread_idx, []() { __syncthreads(); }, true, true);

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
