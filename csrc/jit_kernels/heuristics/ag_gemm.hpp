#pragma once

#include <iostream>
#include <unordered_set>

#include "sm100.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm {

struct AGGemmConfig {
    int block_m, block_n, block_k;
    int load_block_m, load_block_n;
    int sf_block_m, sf_block_n;
    int swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode;
    int num_stages, smem_size;
    int num_ag_threads, num_non_epilogue_threads, num_epilogue_threads;
    int num_multicast;

    friend std::ostream& operator << (std::ostream& os, const AGGemmConfig& config) {
        os << "AGGemmConfig("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", load_block_m=" << config.load_block_m << ", load_block_n=" << config.load_block_n
           << ", sf_block_m=" << config.sf_block_m << ", sf_block_n=" << config.sf_block_n
           << ", swizzle_a_mode=" << config.swizzle_a_mode << ", swizzle_b_mode=" << config.swizzle_b_mode
           << ", swizzle_cd_mode=" << config.swizzle_cd_mode
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_ag_threads=" << config.num_ag_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", num_multicast=" << config.num_multicast << ")";
        return os;
    }
};

static AGGemmConfig get_ag_gemm_config(const int& m, const int& n, const int& k, const int& num_sms,
                                        const int& elem_size_ab = 1) {
    constexpr int block_m = 128;
    constexpr int block_n = 128;
    const int block_k = 128 / elem_size_ab;

    constexpr int num_multicast = 1;
    constexpr int load_block_n = block_n;
    constexpr int load_block_m = block_m;
    constexpr int load_block_n = block_n;
    constexpr int swizzle_a_mode = 128;
    constexpr int swizzle_b_mode = 128;
    constexpr int swizzle_cd_mode = 128;
    constexpr int num_ag_threads = 0;
    constexpr int num_non_epilogue_threads = 128;
    constexpr int num_epilogue_threads = 128;
    const auto [sf_block_m, sf_block_n] = SM100ArchSpec::get_sf_uttcp_aligned_block_sizes(block_m, block_n, MmaKind::MXFP8FP4);

    constexpr int num_epilogue_stages = 2;
    constexpr int num_tma_store_stages = 2;
    const int smem_cd = 128 * swizzle_cd_mode * num_tma_store_stages;
    const int smem_a_per_stage = load_block_m * block_k * elem_size_ab;
    const int smem_b_per_stage = load_block_n * block_k * elem_size_ab;

    const int smem_sfa_per_stage = sf_block_m * 4;
    const int smem_sfb_per_stage = sf_block_n * 4;
    const int smem_barriers = 32 * 8 * 3 + num_epilogue_stages * 8 * 2 + 8;
    const int smem_tmem_ptr = 4;
    const int smem_extra = smem_cd + smem_barriers + smem_tmem_ptr;
    const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
    const int num_stages = std::min((SM100ArchSpec::smem_capacity - smem_extra) / smem_per_stage, 8);
    DG_HOST_ASSERT(num_stages >= 2);

    const auto config = AGGemmConfig{
        block_m, block_n, block_k,
        load_block_m, load_block_n,
        sf_block_m, sf_block_n,
        swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode,
        num_stages, smem_extra + num_stages * smem_per_stage,
        num_ag_threads, num_non_epilogue_threads, num_epilogue_threads,
        num_multicast
    };

    if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = fmt::format("AGGemmConfig(m={}, n={}, k={}, num_sms={}, elem_size_ab={})", m, n, k, num_sms, elem_size_ab);

        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
