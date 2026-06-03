#pragma once

#include <deep_gemm/common/math.cuh>

namespace deep_gemm::sched {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs>
struct AGGemmScheduler {
    uint32_t shape_m, shape_n;
    uint32_t block_idx;
    uint32_t current_iter = 0;

    CUTLASS_DEVICE explicit AGGemmScheduler(const uint32_t& shape_m, const uint32_t& shape_n):
        shape_m(shape_m), shape_n(shape_n), block_idx(blockIdx.x) {}

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const uint32_t num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        const uint32_t num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        if (block_idx >= num_m_blocks * num_n_blocks)
            return false;
        m_block_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx - m_block_idx * num_n_blocks;
        block_idx += kNumSMs;
        ++ current_iter;
        return true;
    }
};

} // namespace deep_gemm::sched
