#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

struct GemmRSWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    uint32_t hidden;
    uint32_t elem_size;
    uint32_t block_m;
    uint32_t block_n;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    GemmRSWorkspace(void* base,
                    const uint32_t& num_ranks,
                    const uint32_t& num_max_tokens_per_rank,
                    const uint32_t& hidden,
                    const uint32_t& elem_size,
                    const uint32_t& block_m = 128,
                    const uint32_t& block_n = 128):
        base(base), num_ranks(num_ranks),
        num_max_tokens_per_rank(num_max_tokens_per_rank), hidden(hidden), elem_size(elem_size),
        block_m(block_m), block_n(block_n) {
        DG_UNIFIED_ASSERT(elem_size == 2 or elem_size == 4);
    }

    CUTLASS_HOST_DEVICE uint32_t get_num_m_blocks_per_rank() const {
        return math::ceil_div(num_max_tokens_per_rank, block_m);
    }

    CUTLASS_HOST_DEVICE uint32_t get_num_n_blocks() const {
        return math::ceil_div(hidden, block_n);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_partial_bytes() const {
        return static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * hidden * elem_size;
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_ready_bytes() const {
        return static_cast<uint64_t>(num_ranks) * get_num_m_blocks_per_rank() * get_num_n_blocks() * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += get_num_partial_bytes();
        num_bytes += get_num_ready_bytes();
        return math::align<uint64_t>(num_bytes, 16);
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE uint32_t* get_grid_sync_count_ptr() const {
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + 4;
    }

    CUTLASS_DEVICE int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(base, 5 * sizeof(uint32_t) + phase * sizeof(int));
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_partial_ptr(const uint32_t& slot_idx = 0,
                                                 const uint32_t& token_idx = 0,
                                                 const uint32_t& hidden_idx = 0) const {
        const uint64_t offset = kNumBarrierSignalBytes +
            (static_cast<uint64_t>(slot_idx) * num_max_tokens_per_rank * hidden +
             static_cast<uint64_t>(token_idx) * hidden + hidden_idx) * elem_size;
        return math::advance_ptr<dtype_t>(base, offset);
    }

    CUTLASS_HOST_DEVICE uint32_t* get_ready_ptr(const uint32_t& slot_idx = 0,
                                                const uint32_t& m_block_idx = 0,
                                                const uint32_t& n_block_idx = 0) const {
        const uint64_t base_offset = kNumBarrierSignalBytes + get_num_partial_bytes();
        const uint64_t idx = (static_cast<uint64_t>(slot_idx) * get_num_m_blocks_per_rank() + m_block_idx) * get_num_n_blocks() + n_block_idx;
        return math::advance_ptr<uint32_t>(base, base_offset + idx * sizeof(uint32_t));
    }
};

} // namespace deep_gemm::layout
