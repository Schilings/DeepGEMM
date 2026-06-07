#pragma once

#include <iostream>
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
    int reduce_num_threads;  // BF16 路径: reduce epilogue kernel 的线程数

    friend std::ostream& operator << (std::ostream& os, const GemmRSConfig& config) {
        os << "GemmRSConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_rs_threads=" << config.num_rs_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", reduce_num_threads=" << config.reduce_num_threads << ")";
        return os;
    }
};

static GemmRSConfig get_gemm_rs_config(const int& m, const int& n, const int& k, const int& num_sms,
                                        const int& elem_size_ab = 1) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    constexpr int load_block_m = block_m;
    constexpr int load_block_n = block_n;
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    // BF16 (elem_size_ab=2): 不再有 RS warps，由独立 PDL reduce kernel 处理
    // FP8  (elem_size_ab=1): 仍保留 RS warps (旧架构)
    const int num_rs_threads = (elem_size_ab == 2) ? 0 : 128;
    constexpr int num_non_epilogue_threads = 128;  // Warp 0 (TMA load) + Warp 1 (MMA issue)
    constexpr int num_epilogue_threads = 128;      // Warp 2-3: TMEM → smem → TMA store
    constexpr int num_multicast = 1;
    constexpr int num_tma_store_stages = 2;
    constexpr int num_epilogue_stages = 2;
    constexpr int reduce_num_threads = 256;        // BF16: Reduce epilogue kernel 线程数

    const int smem_cd = 128 * swizzle_cd_mode * num_tma_store_stages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    const int smem_sfa_per_stage = 128 * 4;
    const int smem_sfb_per_stage = 128 * 4;
    const int smem_barriers = 32 * 8 * 3 + num_epilogue_stages * 8 * 2 + 8;
    const int smem_tmem_ptr = 4;
    const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    const int num_stages = std::min((SM100ArchSpec::smem_capacity - smem_extra) / smem_per_stage, 8);
    DG_HOST_ASSERT(num_stages >= 2);

    const auto config = GemmRSConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_extra + num_stages * smem_per_stage,
        num_rs_threads,
        num_non_epilogue_threads, num_epilogue_threads,
        num_multicast,
        reduce_num_threads
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("GemmRSConfig(m={}, n={}, k={}, num_sms={}, elem_size_ab={})", m, n, k, num_sms, elem_size_ab);

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
