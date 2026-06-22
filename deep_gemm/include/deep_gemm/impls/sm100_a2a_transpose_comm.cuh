#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cuda_bf16.h>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/bf16_a2a_transpose_gemm.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

namespace deep_gemm {

// ============================================================================================
//  sm100_a2a_transpose_comm — Ulysses SP post-attention All2All-transpose scatter (cuda core)
// ============================================================================================
//
//  Each rank r pushes its attention output x[bs, local_nheads, seq, head_dim] to every dst_rank,
//  writing into dst_rank's *gathered* region at hidden-column offset r*local_hidden, with the
//  seq<->head transpose:
//      gathered_dst[b, dst_seq, (r*local_nheads + nh), hd] = x[b, nh, gs, hd]
//      where dst_rank = gs / local_seq,  dst_seq = gs % local_seq.
//  After all ranks finish, each rank's gathered region holds the full-hidden
//  [bs, local_seq, hidden] = A matrix for the Wo GEMM.
//
//  Tile-granular: each CTA handles one (dst_rank, m_tile) — copies that tile's rows for dst and
//  (if kSetBarrier) atomically decrements dst's per-tile barrier; when all kNumRanks sources have
//  contributed, the barrier is set to 1 (consumed by the fused GEMM's per-tile wait). The M-tile
//  granularity (kTileM) MUST equal the GEMM's BLOCK_M so barrier idx == GEMM m_block.
//
//  Vectorized over head_dim with uint4 (8 bf16 = 16B); requires head_dim % 8 == 0.
//
template <uint32_t kNumRanks, uint32_t kTileM, bool kSetBarrier>
__global__ void sm100_a2a_transpose_comm_impl(
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const uint32_t bs,
        const uint32_t nheads,
        const uint32_t seq,
        const uint32_t head_dim) {
    const uint32_t rank = sym_buffer.rank_idx;
    const uint32_t local_nheads = nheads / kNumRanks;
    const uint32_t local_seq = seq / kNumRanks;
    const uint32_t hidden = nheads * head_dim;

    constexpr uint32_t kPack = 8;                         // bf16 per uint4
    const uint32_t vec_hd = head_dim / kPack;
    const uint32_t vec_hidden = hidden / kPack;

    const layout::BF16A2ATransposeGemmWorkspace ws(
        sym_buffer.template get_base_ptr<void*>(), kNumRanks, bs, nheads, seq, head_dim);

    const uint4* in_vec = reinterpret_cast<const uint4*>(ws.template get_input_ptr<void>());
    uint4* gathered_local = reinterpret_cast<uint4*>(ws.template get_gathered_ptr<void>());
    int32_t* barrier_local = ws.get_barrier_ptr();        // mapped to dst below

    const uint32_t tiles_per_seq = (local_seq + kTileM - 1) / kTileM;
    const uint32_t tiles_per_dst = bs * tiles_per_seq;
    const uint32_t total_work = kNumRanks * tiles_per_dst;

    // Signal that the comm kernel has launched (for the GEMM stream's stream-wait-value).
    if (kSetBarrier and blockIdx.x == 0 and threadIdx.x == 0) {
        asm volatile("st.relaxed.gpu.global.b32 [%0], 1;" : : "l"(ws.get_a2a_signal_ptr()));
    }

    for (uint32_t work = blockIdx.x; work < total_work; work += gridDim.x) {
        // dst order = step, OPTIONALLY rotated per rank: dst_rank = (rank + step) % R.
        //
        // Rotation (ring/shifted all-to-all): without it every rank pushes to dst 0 first, then 1,
        // ... so all ranks hammer the SAME destination at once -> only that GPU's NVLink ingress is
        // used (~1/R of the switch bisection). Rotating by rank makes each step a permutation (rank
        // r -> dst r+s), so all R ingress links are busy -> ~+12% comm bandwidth (measured).
        //
        // BUT rotation conflicts with fused overlap: it spreads each rank's INCOMING contributions
        // across all steps (peer p reaches "push to me" only at step (my_rank-p)%R), so my tiles
        // finish only near the end of comm -> the consumer GEMM can't overlap. So we rotate ONLY for
        // the standalone/M0 comm (bandwidth-bound, no consumer); the fused path (kSetBarrier) keeps
        // the un-rotated order so each rank's tiles complete in one early window for overlap.
        const uint32_t step = work / tiles_per_dst;
        const uint32_t dst_rank = kSetBarrier ? step : ((rank + step) % kNumRanks);
        const uint32_t tile = work % tiles_per_dst;
        const uint32_t b = tile / tiles_per_seq;
        const uint32_t t = tile % tiles_per_seq;
        const uint32_t s0 = t * kTileM;
        const uint32_t s1 = (s0 + kTileM < local_seq) ? (s0 + kTileM) : local_seq;
        const uint32_t tile_rows = s1 - s0;
        const uint32_t nelems = tile_rows * local_nheads * vec_hd;

        for (uint32_t i = threadIdx.x; i < nelems; i += blockDim.x) {
            const uint32_t hd_v = i % vec_hd;
            const uint32_t r1 = i / vec_hd;
            const uint32_t nh = r1 % local_nheads;
            const uint32_t s = r1 / local_nheads;          // [0, tile_rows)
            const uint32_t s_local = s0 + s;                // within dst's local_seq
            const uint32_t global_seq = dst_rank * local_seq + s_local;

            const uint64_t in_off = (static_cast<uint64_t>(b) * local_nheads + nh) * seq * vec_hd +
                                    static_cast<uint64_t>(global_seq) * vec_hd + hd_v;
            const uint64_t out_off = (static_cast<uint64_t>(b) * local_seq + s_local) * vec_hidden +
                                     static_cast<uint64_t>(rank * local_nheads + nh) * vec_hd + hd_v;
            uint4* dst_ptr = sym_buffer.map(gathered_local + out_off, dst_rank);
            *dst_ptr = in_vec[in_off];
        }

        if constexpr (kSetBarrier) {
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t barrier_idx = b * tiles_per_seq + t;
                int32_t* bptr = sym_buffer.map(barrier_local + barrier_idx, dst_rank);
                asm volatile("fence.acq_rel.sys;\n");
                int32_t prev = atomicAdd_system(bptr, -1);
                if (prev - 1 == -static_cast<int32_t>(kNumRanks))
                    asm volatile("st.release.sys.b32 [%0], 1;\n" : : "l"(bptr));
            }
        }
    }
}

} // namespace deep_gemm

#pragma clang diagnostic pop
