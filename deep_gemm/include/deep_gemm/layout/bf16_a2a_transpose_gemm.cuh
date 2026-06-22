#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

// Workspace layout for BF16 Ulysses-SP post-attention All2All-transpose + Wo GEMM.
//
// Per rank symmetric buffer regions (one contiguous symmetric allocation):
//   [0] barrier/signal region (int32):
//         - per-M-tile barriers: bs * ceil(local_seq / kTileM) entries (for the fused/overlap path)
//         - + 1 trailing a2a launch signal
//   [1] input region (bf16): this rank's attention output  x[bs, local_nheads, seq, head_dim]
//         (rank r owns heads [r*local_nheads:(r+1)*local_nheads], full seq)
//   [2] gathered region (bf16): the A2A-transpose result for THIS rank's seq shard,
//         xt[bs, local_seq, hidden] = [bs, local_seq, nheads*head_dim]  (written by all peers)
//         This is the A matrix [M = bs*local_seq, K = hidden] consumed by the Wo GEMM.
//
// Transpose-scatter (each rank r pushes into every dst_rank's gathered region):
//   for global token (b, gs): dst_rank = gs / local_seq, dst_seq = gs % local_seq
//   gathered_dst[b, dst_seq, (r*local_nheads + nh), hd] = input_r[b, nh, gs, hd]
//   i.e. rank r's hidden slice lands at hidden-column offset r*local_hidden.
//
// NOTE: input and gathered regions have identical byte size (both = bs*nheads*seq*head_dim / num_ranks).
struct BF16A2ATransposeGemmWorkspace {
    void* base;
    uint32_t num_ranks;
    uint32_t bs;
    uint32_t nheads;
    uint32_t seq;
    uint32_t head_dim;

    static constexpr uint32_t kTileM = 128;          // M-tile granularity for the per-tile barrier
    static constexpr uint64_t kBarrierAlignBytes = 128;

    CUTLASS_HOST_DEVICE
    BF16A2ATransposeGemmWorkspace(void* base,
                                  const uint32_t& num_ranks,
                                  const uint32_t& bs,
                                  const uint32_t& nheads,
                                  const uint32_t& seq,
                                  const uint32_t& head_dim):
        base(base), num_ranks(num_ranks), bs(bs), nheads(nheads), seq(seq), head_dim(head_dim) {}

    CUTLASS_HOST_DEVICE uint32_t local_nheads() const { return nheads / num_ranks; }
    CUTLASS_HOST_DEVICE uint32_t local_seq() const { return seq / num_ranks; }
    CUTLASS_HOST_DEVICE uint32_t hidden() const { return nheads * head_dim; }
    CUTLASS_HOST_DEVICE uint32_t local_hidden() const { return local_nheads() * head_dim; }

    CUTLASS_HOST_DEVICE uint32_t num_m_tiles() const {
        const uint32_t ls = local_seq();
        return bs * ((ls + kTileM - 1) / kTileM);
    }

    // bytes of the barrier region (per-tile barriers + 1 trailing launch signal), aligned.
    CUTLASS_HOST_DEVICE uint64_t get_barrier_bytes() const {
        const uint64_t n = static_cast<uint64_t>(num_m_tiles()) + 1;
        return math::align<uint64_t>(n * sizeof(int32_t), kBarrierAlignBytes);
    }

    // bytes of one data region (input == gathered): bs * local_nheads * seq * head_dim bf16.
    CUTLASS_HOST_DEVICE uint64_t get_data_bytes() const {
        return static_cast<uint64_t>(bs) * local_nheads() * seq * head_dim * sizeof(nv_bfloat16);
    }

    CUTLASS_HOST_DEVICE uint64_t get_num_bytes() const {
        uint64_t n = 0;
        n += get_barrier_bytes();
        n += get_data_bytes();   // input
        n += get_data_bytes();   // gathered
        return math::align<uint64_t>(n, 16);
    }

    // ── region pointers ──
    CUTLASS_HOST_DEVICE int32_t* get_barrier_ptr(const uint32_t& idx = 0) const {
        return math::advance_ptr<int32_t>(base, 0) + idx;
    }
    // trailing entry used as the "a2a launched" signal
    CUTLASS_HOST_DEVICE int32_t* get_a2a_signal_ptr() const {
        return get_barrier_ptr(num_m_tiles());
    }
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_input_ptr() const {
        return math::advance_ptr<dtype_t>(base, get_barrier_bytes());
    }
    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_gathered_ptr() const {
        return math::advance_ptr<dtype_t>(base, get_barrier_bytes() + get_data_bytes());
    }
};

} // namespace deep_gemm::layout
