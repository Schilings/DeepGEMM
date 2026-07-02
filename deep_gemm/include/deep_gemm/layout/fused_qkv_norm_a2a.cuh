#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

// Workspace for the Fused QKV GEMM + RMSNorm + A2A-transpose operator.
//
// Layout (per rank, symmetric):
//   [0 .. 32)            barrier/signal region (kNumBarrierSignalBytes)
//   [32 .. 32+RMS)       rms region: [bs*seq, 2] float32 (Q rms + K rms, per-row)
//   [32+RMS .. 32+RMS+OUT) output region `out`: 2D [bs*seq, local_n_total], bf16
//
// Additionally, non-symmetric local buffer:
//   - sum_buffer [bs*local_seq, 2] (fp32) — per-row x² partial sum (filled by pre_cast)
//     This is LOCAL only (not symmetric), allocated by host.
struct FusedQKVNormA2AWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t bs;
    uint32_t seq;
    uint32_t local_n_total;
    uint32_t elem_size;

    static constexpr uint64_t kNumBarrierSignalBytes = 32;
    static constexpr uint64_t kRMSBytesBase = 0;  // computed at runtime (bs*seq*2*4)

    CUTLASS_HOST_DEVICE
    FusedQKVNormA2AWorkspace(void* base,
                              const uint32_t& num_ranks,
                              const uint32_t& bs,
                              const uint32_t& seq,
                              const uint32_t& local_n_total,
                              const uint32_t& elem_size):
        base(base), num_ranks(num_ranks), bs(bs), seq(seq),
        local_n_total(local_n_total), elem_size(elem_size) {
        DG_UNIFIED_ASSERT(elem_size == 2 or elem_size == 4);
    }

    // RMS region: [bs*seq, 2] float32, right after barrier region
    CUTLASS_HOST_DEVICE uint64_t get_rms_region_bytes() const {
        return static_cast<uint64_t>(bs) * seq * 2 * sizeof(float);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_output_bytes() const {
        return static_cast<uint64_t>(bs) * seq * local_n_total * elem_size;
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;
        num_bytes += kNumBarrierSignalBytes;
        num_bytes += get_rms_region_bytes();   // rms region (new)
        num_bytes += get_num_output_bytes();
        return math::align<uint64_t>(num_bytes, 16);
    }

    // -- NVLink barrier accessors (identical scheme to GemmA2ATransposeWorkspace) --
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

    // -- RMS region: [bs*seq, 2] float32, at offset 32 --
    template <typename dtype_t = float>
    CUTLASS_HOST_DEVICE dtype_t* get_rms_ptr() const {
        return math::advance_ptr<dtype_t>(base, kNumBarrierSignalBytes);
    }

    // -- Output region: after rms region --
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_output_ptr() const {
        return math::advance_ptr<dtype_t>(base, kNumBarrierSignalBytes + get_rms_region_bytes());
    }
};

} // namespace deep_gemm::layout
