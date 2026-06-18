#pragma once

#include <algorithm>
#include <iostream>
#include <tuple>
#include <unordered_set>

#include "sm100.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

struct GemmRSConfig {
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode;
    int num_stages, smem_size;
    int num_rs_threads;
    int num_non_epilogue_threads, num_epilogue_threads;
    int num_multicast;
    bool is_multicast_on_a;
    bool swap_ab;
    bool with_accumulation;
    int reduce_num_threads;

    friend std::ostream& operator << (std::ostream& os, const GemmRSConfig& config) {
        os << "GemmRSConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", swizzle_a=" << config.swizzle_a_mode << ", swizzle_b=" << config.swizzle_b_mode
           << ", swizzle_cd=" << config.swizzle_cd_mode
           << ", num_multicast=" << config.num_multicast
           << ", is_multicast_on_a=" << config.is_multicast_on_a
           << ", swap_ab=" << config.swap_ab
           << ", with_accumulation=" << config.with_accumulation
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_comm_threads=" << config.num_rs_threads
           << ", reduce_num_threads=" << config.reduce_num_threads << ")";
        return os;
    }
};

// ════════════════════════════════════════════════════════════════
//  Push-based single-kernel GEMM + RS 配置
//
//  Warp layout (MegaMoE style):
//    W0~W3: Comm (Dispatch) Warps — 128T, 48 regs — poll local flags + reduce + write output
//    W4: TMA Load Warp (A+B) — 32T, 40 regs — unified TMA multicast load
//    W5: Reserved — 32T, 40 regs
//    W6: MMA Issue Warp — 32T, 40 regs — single-warp UMMA (Blackwell)
//    W7: Reserved — 32T, 40 regs
//    W8~W11: Epilogue Warps — 128T, 208 regs — TMEM → smem → NVLink push + flag
//
//  Total: 128 + 128 + 128 = 384 threads = 12 warps
//  Registers: 48×128 + 40×128 + 208×128 = 6144 + 5120 + 26624 = 37888 (within SM100 max 64512)
//
//  TMA Multicast = 2 (2-CTA cluster):
//    A matrix read once from HBM, multicast to 2 SMs' smem
//    Effective 2x HBM bandwidth for A (critical for compute-bound tiles)
//
// ════════════════════════════════════════════════════════════════
static GemmRSConfig get_gemm_rs_config(const int& m, const int& n, const int& k, const int& num_sms,
                                       const int& elem_size_ab = 1, const int& num_ranks = 1) {
    const int m_per_rank = num_ranks > 1 ? m / num_ranks : m;
    const bool is_fp8 = (elem_size_ab == 1);

    // MegaMoE-style warp allocation:
    //   Comm Warps (W0-W3): 128 threads, 48 regs — poll local flags + vectorized reduce + write
    //   Non-Epilogue (W4-W7): 128 threads, 40 regs — TMA Load A+B, MMA Issue, Reserved
    //   Epilogue (W8-W11): 128 threads, 208 regs — TMEM → smem → NVLink push store
    constexpr int num_comm_threads = 128;           // W0-W3: Comm/Dispatch warps
    constexpr int num_non_epilogue_threads = 128;   // W4-W7: Load + MMA + Reserved
    constexpr int num_epilogue_threads = 128;       // W8-W11: Epilogue (1 warpgroup, 4 warps)
    // Total = 384 threads = 12 warps

    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    // Block M selection:
    // 主线策略按 SM100 2-CTA cluster（mc=2）优先，遵循 SM100_2CTA_CLUSTER 文档。
    // 仅当 M 过小或无法形成偶数 M-block 对时才回退到 mc=1。
    int block_m;
    int num_multicast;
    bool is_multicast_on_a = false;  // A is multicast, B is split (standard non-swap config)

    // For multicast=2 (2-CTA cluster): the scheduler assigns M-tile PAIRS to clusters.
    // So num_m_blocks_per_rank must be even (divisible by 2).
    const int num_m_blocks_mc2 = (m_per_rank + 128 - 1) / 128;
    const bool m_blocks_even = (num_m_blocks_mc2 % 2 == 0);
    const bool prefer_mc2_cluster = (m_per_rank >= 128 && m_blocks_even);

    if (prefer_mc2_cluster) {
        // mc=2 主线下，小K+大N中大shape尝试更大M tile，减少cluster调度与同步次数
        if (k <= 2048 && n >= 7168 && m_per_rank <= 1024) {
            block_m = 256;
        } else {
            block_m = 128;
        }
        num_multicast = 2;
    } else if (m_per_rank >= 128) {
        // Fallback when M-block pairing for cluster is not possible
        block_m = 128;
        num_multicast = 1;
    } else {
        // Very small M (< 128 tokens per rank): use block_m=128 with multicast=1
        block_m = 128;
        num_multicast = 1;
    }

    const int load_block_m = block_m / (is_multicast_on_a ? num_multicast : 1);
    const int load_block_n = block_n / (is_multicast_on_a ? 1 : num_multicast);
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    constexpr int reduce_num_threads = 0;  // No separate reduce kernel needed

    // Pipeline stages (with comm fetch stages)
    constexpr int kNumTMAStoreStages = 2;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumCommFetchStages = 2;

    const int store_block_m = std::min(block_m, 128);
    const int store_block_n = swizzle_cd_mode / (is_fp8 ? 1 : 2);  // sizeof(comm_dtype_t)

    // smem sizing
    const int smem_cd = store_block_m * store_block_n * (is_fp8 ? 1 : 2) * kNumTMAStoreStages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    // Comm fetch buffer: currently unused (comm warps do direct P2P global loads)
    const int smem_comm = 0;
    const int smem_comm_barriers = 0;

    const int barriers_per_stage = 2;  // full + empty
    const int smem_barriers_fixed = kNumEpilogueStages * 2 * 8 + 4;  // tmem barriers + tmem ptr

    const int smem_fixed = smem_cd + smem_barriers_fixed + smem_comm + smem_comm_barriers + 256;  // +256 for alignment
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + barriers_per_stage * 8;

    int num_stages = (SM100ArchSpec::smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    // mc=2 主线下对小K+大N场景降低stage深度，减少流水线启动/同步开销。
    if (num_multicast == 2 && k <= 2048 && n >= 7168 && m_per_rank <= 2048) {
        num_stages = std::max(2, std::min(num_stages, 4));
    }

    const int smem_size = smem_fixed + num_stages * smem_per_stage;

    constexpr bool swap_ab = false;
    constexpr bool with_accumulation = false;

    const auto config = GemmRSConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_size,
        num_comm_threads,
        num_non_epilogue_threads, num_epilogue_threads,
        num_multicast,
        is_multicast_on_a,
        swap_ab,
        with_accumulation,
        reduce_num_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("GemmRSConfig(m={}, n={}, k={}, num_sms={}, num_ranks={})",
                                     m, n, k, num_sms, num_ranks);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
