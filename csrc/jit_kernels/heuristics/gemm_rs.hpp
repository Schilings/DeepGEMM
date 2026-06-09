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
           << ", reduce_num_threads=" << config.reduce_num_threads << ")";
        return os;
    }
};

// ════════════════════════════════════════════════════════════════
//  Pull-based single-kernel GEMM + RS 配置
//  需要额外的 comm threads 用于 pull + reduce
// ════════════════════════════════════════════════════════════════
static GemmRSConfig get_gemm_rs_v2_config(const int& m, const int& n, const int& k, const int& num_sms,
                                           const int& elem_size_ab = 1, const int& num_ranks = 1) {
    const int m_per_rank = num_ranks > 1 ? m / num_ranks : m;
    const bool is_fp8 = (elem_size_ab == 1);

    // V2 warp allocation:
    //   W0: TMA Load (32 threads)
    //   W1: MMA Issue (32 threads)
    //   W2-3: Epilogue (64 threads) — TMEM → smem → local partial + set flag
    //   W4-7: Comm Warps (128 threads) — Pull + Reduce from peer ranks
    constexpr int num_non_epilogue_threads = 128;  // W0-W3 (GEMM control)
    constexpr int num_epilogue_threads = 64;       // W2-W3 (Epilogue, lighter since only local write)
    constexpr int num_rs_threads = 128;            // W4-W7 (Comm: pull + reduce)
    // Note: num_non_epilogue_threads includes W0(TMA) + W1(MMA) + W2-3(epilogue base)
    // Actually for V2: kNumGemmThreads=128, kNumEpilogueThreads=64, kNumCommThreads=128
    // Total = 320 threads = 10 warps

    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    // Block M selection based on occupancy
    const int num_n_blocks = (n + block_n - 1) / block_n;
    auto compute_waves = [&](int bm) {
        int num_m_blocks = (m_per_rank + bm - 1) / bm;
        int total_blocks = num_m_blocks * num_n_blocks * num_ranks;  // total across all ranks' chunks
        return static_cast<float>(total_blocks) / num_sms;
    };

    int block_m;
    if (compute_waves(128) >= 1.5f) {
        block_m = 128;
    } else if (compute_waves(64) >= 1.5f) {
        block_m = 64;
    } else {
        block_m = 32;
    }

    const int load_block_m = block_m;
    const int load_block_n = block_n;
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    constexpr int num_multicast = 1;
    constexpr int reduce_num_threads = 0;  // V2 不需要单独的 reduce kernel

    // Pipeline stages (same formula as V1, but with additional comm smem)
    constexpr int kNumTMAStoreStages = 2;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumCommStages = 2;

    const int store_block_m = std::min(block_m, 128);
    const int store_block_n = swizzle_cd_mode / (is_fp8 ? 1 : 2);  // sizeof(comm_dtype_t)

    // smem sizing
    const int smem_cd = store_block_m * store_block_n * (is_fp8 ? 1 : 2) * kNumTMAStoreStages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    const int smem_comm = block_m * store_block_n * (is_fp8 ? 1 : 2) * kNumCommStages;

    const int barriers_per_stage = 2;  // full + empty
    const int smem_barriers_fixed = kNumEpilogueStages * 2 * 8 + 4;  // tmem barriers + tmem ptr

    const int smem_fixed = smem_cd + smem_barriers_fixed + smem_comm + 128;  // +128 for alignment
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + barriers_per_stage * 8;

    const int num_stages = (SM100ArchSpec::smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    const int smem_size = smem_fixed + num_stages * smem_per_stage;

    constexpr bool is_multicast_on_a = true;
    constexpr bool swap_ab = false;
    constexpr bool with_accumulation = false;

    const auto config = GemmRSConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_size,
        num_rs_threads,
        num_non_epilogue_threads, num_epilogue_threads,
        num_multicast,
        is_multicast_on_a,
        swap_ab,
        with_accumulation,
        reduce_num_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("GemmRSV2Config(m={}, n={}, k={}, num_sms={}, num_ranks={})",
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
