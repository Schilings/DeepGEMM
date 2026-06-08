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
    int num_rs_threads;  // FP8 路径仍使用 RS warps，BF16 路径设为 0
    int num_non_epilogue_threads, num_epilogue_threads;
    int num_multicast;
    bool is_multicast_on_a;
    bool swap_ab;
    bool with_accumulation;
    int reduce_num_threads;  // BF16 路径: reduce epilogue kernel 的线程数

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

// ── 动态选择 block_m 和 epilogue 线程数 ──
// 类似 mega_moe 根据问题规模自适应选择最优配置
static std::tuple<int, int, int, int> get_block_config_for_gemm_rs(
    const int& m_per_rank, const int& n, const int& num_sms) {
    // m_per_rank: 每个 rank 处理的行数
    // 返回: {block_m, store_block_m, num_epilogue_threads, num_non_epilogue_threads}
    //
    // 设计原则:
    // - 小 batch: 小 block_m 保证 wave 占满 SM
    // - 大 batch: 大 block_m 减少 epilogue overhead 占比
    // - epilogue_threads 决定 TMEM→smem 吞吐量和 TMA store 并行度
    //
    // 约束:
    // - block_m 必须是 32 的倍数（TMEM 32dp）
    // - block_m 可选: 32, 64, 128（LAYOUT_AD_M = 128）
    // - store_block_m ≤ min(block_m, 128)
    // - num_epilogue_threads 必须 ≥ store_block_m（每个线程处理一行 TMEM→smem）

    const int num_n_blocks = (n + 127) / 128;  // block_n = 128

    // 计算不同 block_m 下的 wave 数目（需要足够的 block 填满 SM）
    auto compute_waves = [&](int bm) {
        int num_m_blocks = (m_per_rank + bm - 1) / bm;
        int total_blocks = num_m_blocks * num_n_blocks;
        return static_cast<float>(total_blocks) / num_sms;
    };

    float waves_128 = compute_waves(128);
    float waves_64 = compute_waves(64);
    float waves_32 = compute_waves(32);

    int block_m, store_block_m, num_epilogue_threads, num_non_epilogue_threads;

    if (waves_128 >= 1.5f) {
        // 足够多的 block 填满 SM，用大 block_m 减少 epilogue 次数
        block_m = 128;
        store_block_m = 128;
        num_non_epilogue_threads = 128;  // warp 0 (TMA load) + warp 1 (MMA) + warp 2-3 (spare)
        num_epilogue_threads = 128;      // warp 4-7: 128 threads = 128 rows 并行
    } else if (waves_64 >= 1.5f) {
        // 中等 batch
        block_m = 64;
        store_block_m = 64;
        num_non_epilogue_threads = 128;
        num_epilogue_threads = 128;      // 多于 store_block_m(64)，多余线程不参与 TMEM 写入
    } else if (waves_32 >= 1.5f) {
        // 小 batch
        block_m = 32;
        store_block_m = 32;
        num_non_epilogue_threads = 128;
        num_epilogue_threads = 128;
    } else {
        // 极小 batch（block 不足以填满 SM）
        // 仍使用 block_m=32，减少 waste
        block_m = 32;
        store_block_m = 32;
        num_non_epilogue_threads = 128;
        num_epilogue_threads = 128;
    }

    return {block_m, store_block_m, num_epilogue_threads, num_non_epilogue_threads};
}

// ── Pipeline stage 数量计算 ──
// 根据 shared memory 容量和 tile 大小自动计算最大 stage 数
// is_fp8: FP8 需要额外的 SFA/SFB shared memory 和 with_sf_full barriers
static std::pair<int, int> get_pipeline_config_for_gemm_rs(
    const int& block_m, const int& block_n, const int& block_k,
    const int& store_block_m, const int& elem_size_ab,
    const int& num_epilogue_threads, const int& swizzle_cd_mode,
    const bool& is_fp8 = false) {
    constexpr int kNumTMAStoreStages = 2;
    constexpr int kNumEpilogueStages = 2;

    const int load_block_m = block_m;  // No multicast split for now (num_multicast=1)
    const int load_block_n = block_n;

    // C/D output region: STORE_BLOCK_M × (swizzle_cd_mode / sizeof(comm_dtype_t)) × sizeof(comm_dtype_t)
    // = STORE_BLOCK_M × swizzle_cd_mode (bytes) per stage
    const int smem_cd = store_block_m * swizzle_cd_mode * kNumTMAStoreStages;

    // Per-stage: A tile + B tile
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    // FP8: scale factor buffers per stage
    // SF_BLOCK_M = align(block_m, 128), SF_BLOCK_N = align(block_n, 128)
    const int sf_block_m = is_fp8 ? ((block_m + 127) / 128 * 128) : 0;
    const int sf_block_n = is_fp8 ? ((block_n + 127) / 128 * 128) : 0;
    const int smem_sfa_per_stage = sf_block_m * 4;  // sizeof(uint32_t)
    const int smem_sfb_per_stage = sf_block_n * 4;

    // Barriers per stage:
    // - BF16: full + empty = 2 barriers per stage
    // - FP8: full + empty + with_sf_full = 3 barriers per stage
    const int barriers_per_stage = is_fp8 ? 3 : 2;
    const int smem_barriers_per_stage = barriers_per_stage * 8;

    // Fixed: epilogue barriers (tmem_full + tmem_empty) + tmem pointer
    const int smem_barriers_fixed = kNumEpilogueStages * 2 * 8;  // tmem full/empty
    const int smem_tmem_ptr = 4;

    const int smem_fixed = smem_cd + smem_barriers_fixed + smem_tmem_ptr;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage
                             + smem_sfa_per_stage + smem_sfb_per_stage
                             + smem_barriers_per_stage;

    // No artificial cap — let shared memory capacity decide
    const int num_stages = (SM100ArchSpec::smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);

    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static GemmRSConfig get_gemm_rs_config(const int& m, const int& n, const int& k, const int& num_sms,
                                        const int& elem_size_ab = 1, const int& num_ranks = 1) {
    const int m_per_rank = num_ranks > 1 ? m / num_ranks : m;
    const bool is_fp8 = (elem_size_ab == 1);  // FP8 = 1 byte, BF16 = 2 bytes

    // ── 动态 block 配置 ──
    const auto [block_m, store_block_m, num_epilogue_threads, num_non_epilogue_threads] =
        get_block_config_for_gemm_rs(m_per_rank, n, num_sms);

    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    const int load_block_m = block_m;
    const int load_block_n = block_n;
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    const int num_rs_threads = 0;
    constexpr int num_multicast = 1;
    constexpr int reduce_num_threads = 256;

    // ── Pipeline stages ──
    const auto [num_stages, smem_size] = get_pipeline_config_for_gemm_rs(
        block_m, block_n, block_k, store_block_m, elem_size_ab,
        num_epilogue_threads, swizzle_cd_mode, is_fp8);

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
        const auto key = fmt::format("GemmRSConfig(m={}, n={}, k={}, num_sms={}, elem_size_ab={}, num_ranks={})",
                                     m, n, k, num_sms, elem_size_ab, num_ranks);

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
