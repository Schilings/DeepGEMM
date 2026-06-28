#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

// Workspace for the Ulysses-SP pre-attn GEMM + All2All-transpose (single-kernel).
//
// Unlike GemmRSWorkspace (which keeps num_ranks partial slots + per-tile ready flags for the
// downstream FP32 reduce), the A2A here is a pure permutation: every output position is written
// EXACTLY ONCE by exactly one source rank. So there is no reduce, no per-rank slots, and no
// zeroing — the symmetric output buffer IS the result for this rank's head group.
//
// Memory layout (per rank, symmetric):
//   [0 .. 32)            barrier/signal region (kNumBarrierSignalBytes), same scheme as
//                        GemmRSWorkspace: grid_sync_count(idx0) / nvl_barrier_counter(idx4) /
//                        nvl_barrier_signal[2](idx5,6). Used by comm::nvlink_barrier.
//   [32 .. 32+OUT)       output region `out`: 2D [bs*seq, local_n], row-major, stride local_n.
//                        local_n = (nheads/num_ranks)*head_dim = N/num_ranks. This is BSHD
//                        [bs, seq, local_nheads, head_dim] flattened.
struct GemmA2ATransposeWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t bs;
    uint32_t seq;
    uint32_t local_n;       // N / num_ranks = local_nheads * head_dim
    uint32_t elem_size;     // 2 (bf16) or 4 (fp32)

    static constexpr uint64_t kNumBarrierSignalBytes = 32;

    CUTLASS_HOST_DEVICE
    GemmA2ATransposeWorkspace(void* base,
                              const uint32_t& num_ranks,
                              const uint32_t& bs,
                              const uint32_t& seq,
                              const uint32_t& local_n,
                              const uint32_t& elem_size):
        base(base), num_ranks(num_ranks), bs(bs), seq(seq), local_n(local_n), elem_size(elem_size) {
        DG_UNIFIED_ASSERT(elem_size == 2 or elem_size == 4);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_output_bytes() const {
        return static_cast<uint64_t>(bs) * seq * local_n * elem_size;
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += get_num_output_bytes();
        return math::align<uint64_t>(num_bytes, 16);
    }

    // -- NVLink barrier accessors (identical scheme to GemmRSWorkspace) --
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

    // -- Output region base pointer --
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_output_ptr() const {
        return math::advance_ptr<dtype_t>(base, kNumBarrierSignalBytes);
    }
};

} // namespace deep_gemm::layout
