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

struct GemmRSComputeConfig {
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode;
    int num_stages, smem_size;
    int num_non_epilogue_threads, num_epilogue_threads;
    int num_multicast;
    bool is_multicast_on_a;
    bool swap_ab;
    bool with_accumulation;

    friend std::ostream& operator << (std::ostream& os, const GemmRSComputeConfig& config) {
        os << "GemmRSComputeConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", swizzle_a=" << config.swizzle_a_mode << ", swizzle_b=" << config.swizzle_b_mode
           << ", swizzle_cd=" << config.swizzle_cd_mode
           << ", num_multicast=" << config.num_multicast
           << ", is_multicast_on_a=" << config.is_multicast_on_a
           << ", swap_ab=" << config.swap_ab
           << ", with_accumulation=" << config.with_accumulation
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads << ")";
        return os;
    }
};

// ====================================================================
//  Dual-kernel GEMM Compute configuration
//
//  Warp layout (256T = 8 warps, no Comm Warps):
//    W0: TMA Load A+B (elect_one)       — 32T, 40 regs
//    W1: MMA Issue (is_leader_cta)      — 32T, 40 regs
//    W2: Reserved / TMEM Allocator      — 32T, 40 regs
//    W3: Reserved                       — 32T, 40 regs
//    W4-W7: Epilogue Warps              — 128T, 208 regs
//
//  Total: 128 + 128 = 256 threads = 8 warps
//  Registers: 40×128 + 208×128 = 5120 + 26624 = 31744 (within SM100 max 64512)
//
// ====================================================================
static GemmRSComputeConfig get_gemm_rs_compute_config(const int& m, const int& n, const int& k, const int& num_sms,
                                                       const int& elem_size_ab = 1, const int& num_ranks = 1) {
    const int m_per_rank = num_ranks > 1 ? m / num_ranks : m;
    const bool is_fp8 = (elem_size_ab == 1);

    // Dual-kernel warp allocation (no comm threads):
    //   Non-Epilogue (W0-W3): 128 threads, 40 regs — TMA Load A+B, MMA Issue, Reserved
    //   Epilogue (W4-W7): 128 threads, 208 regs — Epilogue (1 warpgroup, 4 warps)
    // Total = 256 threads = 8 warps
    constexpr int num_non_epilogue_threads = 128;   // W0-W3: Load + MMA + Reserved
    constexpr int num_epilogue_threads = 128;       // W4-W7: Epilogue (1 warpgroup, 4 warps)

    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    // Block M selection (same logic as single-kernel GEMM+RS):
    // When multicast=2 (2-CTA cluster), UMMA 2x1SM requires each CTA to have 128 rows.
    // Therefore block_m must be >= 128 when multicast is enabled.
    const int num_n_blocks = (n + block_n - 1) / block_n;
    auto compute_waves = [&](int bm, int mc) {
        int num_m_blocks = (m_per_rank + bm - 1) / bm;
        int total_blocks = num_m_blocks * num_n_blocks * num_ranks;
        int effective_sms = num_sms / mc;
        return static_cast<float>(total_blocks) / effective_sms;
    };

    int block_m;
    int num_multicast;
    bool is_multicast_on_a = false;

    const int num_m_blocks_mc2 = (m_per_rank + 128 - 1) / 128;
    const bool m_blocks_even = (num_m_blocks_mc2 % 2 == 0);

    if (m_per_rank >= 256 && m_blocks_even && compute_waves(128, 2) >= 0.5f) {
        block_m = 128;
        num_multicast = 2;
    } else if (m_per_rank >= 128) {
        block_m = 128;
        num_multicast = 1;
    } else {
        block_m = 128;
        num_multicast = 1;
    }

    const int load_block_m = block_m / (is_multicast_on_a ? num_multicast : 1);
    const int load_block_n = block_n / (is_multicast_on_a ? 1 : num_multicast);
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;

    // Pipeline stages
    constexpr int kNumTMAStoreStages = 2;
    constexpr int kNumEpilogueStages = 2;

    const int store_block_m = std::min(block_m, 128);
    const int store_block_n = swizzle_cd_mode / (is_fp8 ? 1 : 2);

    // smem sizing (no comm fetch buffer in dual-kernel)
    const int smem_cd = store_block_m * store_block_n * (is_fp8 ? 1 : 2) * kNumTMAStoreStages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    const int barriers_per_stage = 2;
    const int smem_barriers_fixed = kNumEpilogueStages * 2 * 8 + 4;

    const int smem_fixed = smem_cd + smem_barriers_fixed + 256;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + barriers_per_stage * 8;

    // Optionally reserve smem headroom so that a separate RS-reduce block can become
    // CO-RESIDENT on the same SM as the GEMM block (enabling true compute/comm overlap).
    // By default the GEMM fills all of smem (max stages) → no room for a 2nd resident block,
    // so the reduce can only run in the GEMM tail (no overlap). DG_GEMM_RS_RESERVE_SMEM_KB
    // caps the GEMM stage count to leave `KB` KiB free for the reduce kernel.
    const int reserve_bytes = get_env<int>("DG_GEMM_RS_RESERVE_SMEM_KB", 0) * 1024;
    const int avail_smem = SM100ArchSpec::smem_capacity - reserve_bytes;
    const int num_stages = (avail_smem - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    const int smem_size = smem_fixed + num_stages * smem_per_stage;

    constexpr bool swap_ab = false;
    constexpr bool with_accumulation = false;

    const auto config = GemmRSComputeConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_size,
        num_non_epilogue_threads,
        num_epilogue_threads,
        num_multicast,
        is_multicast_on_a,
        swap_ab,
        with_accumulation
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("GemmRSComputeConfig(m={}, n={}, k={}, num_sms={}, num_ranks={})",
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
