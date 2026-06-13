#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

// Workspace layout for BF16 All2All + GEMM fusion (Flux-style).
//
// Each rank has:
//   - local_x: input data [num_ranks * M_per_rank, K] — the full input to scatter
//     * Chunk j (rows j*M_per_rank..(j+1)*M_per_rank) is sent to rank j
//   - slots[0..num_slots-1]: receive buffers [M_per_rank, K] each
//     * slot[j] receives data FROM rank j
//   - slot_state[0..num_slots-1][0..kNumReadyChunksPerSlot-1]: per-rank per-chunk ready flags
//
// Communication pattern (Host-side CE DMA, Flux-style):
//   Local:  local_x[rank_idx] → slot[rank_idx], set slot_state[rank_idx][chunk] = 1
//   Remote: For each j ≠ rank_idx:
//     cudaMemcpyAsync(slot[j] ← rank_j's local_x[rank_idx]), set slot_state[j][chunk] = 1
//
// GEMM:
//   After slots filled, A matrix = concat(slot[0], slot[1], ..., slot[N-1])
//   = [num_ranks * M_per_rank, K]
//   GEMM: A × B^T → D [num_ranks * M_per_rank, N]
//
struct BF16A2AGemmWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;
    uint32_t hidden;  // K dimension
    uint32_t num_slots;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;
    static constexpr uint32_t kNumReadyChunksPerSlot = 4;

    CUTLASS_HOST_DEVICE
    BF16A2AGemmWorkspace(void* base,
                         const uint32_t& num_ranks,
                         const uint32_t& num_max_tokens_per_rank,
                         const uint32_t& hidden,
                         const uint32_t& num_slots):
        base(base), num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        hidden(hidden), num_slots(num_slots) {}

    CUTLASS_HOST_DEVICE uint64_t get_num_token_bytes_per_rank() const {
        return static_cast<uint64_t>(num_max_tokens_per_rank) * hidden * sizeof(nv_bfloat16);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        // local_x: num_ranks chunks, each M_per_rank × K
        num_bytes += num_ranks * get_num_token_bytes_per_rank();
        // slots: num_slots receive buffers, each M_per_rank × K
        num_bytes += num_slots * get_num_token_bytes_per_rank();
        // slot_state: num_slots × kNumReadyChunksPerSlot × sizeof(uint32_t)
        num_bytes += static_cast<uint64_t>(num_slots) * sizeof(uint32_t) * kNumReadyChunksPerSlot;
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

    // local_x: the full input [num_ranks * M_per_rank, K]
    // Chunk j = rows [j*M_per_rank, (j+1)*M_per_rank) — to be sent to rank j
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_local_x_ptr(const uint32_t& chunk_idx = 0, const uint32_t& token_idx = 0) const {
        return math::advance_ptr<dtype_t>(base,
            kNumBarrierSignalBytes +
            (static_cast<uint64_t>(chunk_idx) * num_max_tokens_per_rank + token_idx) * hidden * sizeof(nv_bfloat16));
    }

    // slots: receive buffers. slot[j] = data received FROM rank j
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_slot_x_ptr(const uint32_t& slot_idx = 0, const uint32_t& token_idx = 0) const {
        auto* slots_base = math::advance_ptr(base,
            kNumBarrierSignalBytes + num_ranks * get_num_token_bytes_per_rank());
        return math::advance_ptr<dtype_t>(slots_base,
            (static_cast<uint64_t>(slot_idx) * num_max_tokens_per_rank + token_idx) * hidden * sizeof(nv_bfloat16));
    }

    // Per-chunk ready flags: slot_state[slot_idx][chunk_idx]
    CUTLASS_HOST_DEVICE uint32_t* get_slot_state_ptr(const uint32_t& slot_idx = 0, const uint32_t& chunk_idx = 0) const {
        auto* base_ptr = math::advance_ptr<uint32_t>(get_slot_x_ptr(num_slots, 0), 0);
        return base_ptr + slot_idx * kNumReadyChunksPerSlot + chunk_idx;
    }
};

} // namespace deep_gemm::layout
