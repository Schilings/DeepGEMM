#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

struct AGGemmWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    uint32_t hidden;
    uint32_t gran_k;
    uint32_t num_slots;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    AGGemmWorkspace(void* base,
                    const uint32_t& num_ranks,
                    const uint32_t& num_max_tokens_per_rank,
                    const uint32_t& hidden,
                    const uint32_t& gran_k,
                    const uint32_t& num_slots):
        base(base),
        num_ranks(num_ranks),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        hidden(hidden), gran_k(gran_k), num_slots(num_slots) {
        DG_UNIFIED_ASSERT(gran_k == 32 or gran_k == 128);
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_num_sf_cols() const {
        return math::ceil_div(hidden, gran_k * 4);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_token_bytes_per_rank() const {
        return static_cast<uint64_t>(num_max_tokens_per_rank) * hidden;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_sf_bytes_per_rank() const {
        return static_cast<uint64_t>(num_max_tokens_per_rank) * get_num_sf_cols() * sizeof(uint32_t);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += get_num_token_bytes_per_rank();            // local input tokens
        num_bytes += get_num_sf_bytes_per_rank();               // local input scale factors
        num_bytes += num_slots * get_num_token_bytes_per_rank(); // ring/staging token slots
        num_bytes += num_slots * get_num_sf_bytes_per_rank();    // ring/staging SF slots
        num_bytes += num_slots * sizeof(uint32_t) * 4;           // per-slot ready/state counters
        return math::align<uint64_t>(num_bytes, 16);
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        return static_cast<uint32_t*>(base) + kIndex;
    }


    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + 4;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        return math::advance_ptr<int>(base, 5 * sizeof(uint32_t) + phase * sizeof(int));
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_local_x_ptr(const uint32_t& token_idx = 0) const {
        return math::advance_ptr<dtype_t>(base, kNumBarrierSignalBytes + static_cast<uint64_t>(token_idx) * hidden);
    }

    CUTLASS_HOST_DEVICE uint32_t* get_local_x_sf_ptr(const uint32_t& token_idx = 0) const {
        return math::advance_ptr<uint32_t>(get_local_x_ptr(num_max_tokens_per_rank), static_cast<uint64_t>(token_idx) * get_num_sf_cols() * sizeof(uint32_t));
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_slot_x_ptr(const uint32_t& slot_idx = 0, const uint32_t& token_idx = 0) const {
        auto* base_ptr = math::advance_ptr(get_local_x_sf_ptr(num_max_tokens_per_rank), slot_idx * get_num_token_bytes_per_rank());
        return math::advance_ptr<dtype_t>(base_ptr, static_cast<uint64_t>(token_idx) * hidden);
    }

    CUTLASS_HOST_DEVICE uint32_t* get_slot_x_sf_ptr(const uint32_t& slot_idx = 0, const uint32_t& token_idx = 0) const {
        auto* base_ptr = math::advance_ptr(get_slot_x_ptr(num_slots, 0), slot_idx * get_num_sf_bytes_per_rank());
        return math::advance_ptr<uint32_t>(base_ptr, static_cast<uint64_t>(token_idx) * get_num_sf_cols() * sizeof(uint32_t));
    }

    CUTLASS_DEVICE uint32_t* get_slot_state_ptr(const uint32_t& slot_idx = 0) const {
        auto* base_ptr = math::advance_ptr<uint32_t>(get_slot_x_sf_ptr(num_slots, 0), 0);
        return base_ptr + slot_idx * 4;
    }
};

} // namespace deep_gemm::layout
