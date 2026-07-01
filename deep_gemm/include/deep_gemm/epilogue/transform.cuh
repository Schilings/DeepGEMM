#pragma once

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>

namespace deep_gemm::epilogue::transform {

// ════════════════════════════════════════════════════════════════
//  Epilogue transform 接口
//
//  sm100_store_cd 的 epilogue_type_t 需实现以下方法（均可选，默认 no-op）：
//
//  1. apply_index_n<STORE_BLOCK_N>(n_idx) → uint32_t
//     N 维索引变换（如 head splits）。默认恒等。
//
//  2. pre_cast(values, num_elems, global_m, n_col, ctx) → void
//     在 TMEM→SMEM 的 STSM 阶段，拿到 fp32 GEMM 结果后、cast 成 bf16 前调用。
//     用于：RMSNorm 的 x² partial sum 计算（atomic add 到 sum_buffer）。
//     默认 no-op。
//
//  pre_cast 的 ctx 参数由 kernel 传入，包含 sum_buffer 指针、Q/K 段范围等。
//  通过 ctx 传运行时参数，避免 transform 类型本身携带状态。
// ════════════════════════════════════════════════════════════════

// 默认 transform：所有方法均为 no-op
struct EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        return n_idx;
    }

    // pre_cast: no-op（values 是 fp32 GEMM 结果，num_elems 个元素）
    struct Context {};  // 空 context
    template <uint32_t kNumElems>
    CUTLASS_DEVICE static void pre_cast(float* values,
                                         const uint32_t& global_m,
                                         const uint32_t& n_col,
                                         const Context& ctx) {}
};

// Head splits transform（原有功能，不变）
template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits: EpilogueIdentity {
    template <uint32_t STORE_BLOCK_N>
    CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) {
        DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 and kMid % STORE_BLOCK_N == 0 and
                         kRight % STORE_BLOCK_N == 0, "Invalid head splits config");
        return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
    }
};

// ════════════════════════════════════════════════════════════════
//  EpilogueX2Sum: 在 STSM 的 cast 前计算 x² partial sum
//
//  对 Q/K 段的 tile，把 8 个 fp32 值的平方和 atomic add 到 sum_buffer[row, q_or_k_idx]。
//  V 段不做 sum。
//
//  Context 携带运行时参数：
//    sum_buffer: [shape_m, 2] fp32，per-row x² sum (Q=idx0, K=idx1)
//    q_dim, kv_dim: Q/K 段范围（N 维）
// ════════════════════════════════════════════════════════════════
struct EpilogueX2Sum: EpilogueIdentity {
    struct Context {
        float* sum_buffer;   // [local_m, 2] fp32 (Q=0, K=1), indexed by GEMM M row
        uint32_t q_dim, kv_dim;       // GEMM-space segment boundaries
        int32_t gemm_m_offset;        // gemm_m = out_m + gemm_m_offset (per-tile)
        int32_t gemm_n_offset;        // gemm_n = out_n + gemm_n_offset (per-tile)
    };

    template <uint32_t kNumElems>
    CUTLASS_DEVICE static void pre_cast(float* values,
                                         const uint32_t& out_m,   // output M row
                                         const uint32_t& n_col,    // output N col (post-scatter)
                                         const Context& ctx) {
        // Convert output coords → GEMM coords
        uint32_t gemm_m = static_cast<uint32_t>(static_cast<int32_t>(out_m) + ctx.gemm_m_offset);
        uint32_t gemm_n = static_cast<uint32_t>(static_cast<int32_t>(n_col) + ctx.gemm_n_offset);
        const bool is_q = (gemm_n < ctx.q_dim);
        const bool is_k = (!is_q && gemm_n < ctx.q_dim + ctx.kv_dim);
        if (is_q || is_k) {
            const uint32_t sum_idx = is_q ? 0 : 1;
            float sq_sum = 0.f;
            #pragma unroll
            for (uint32_t i = 0; i < kNumElems; ++i)
                sq_sum += values[i] * values[i];
            atomicAdd(&ctx.sum_buffer[gemm_m * 2 + sum_idx], sq_sum);
        }
    }
};

} // namespace deep_gemm::epilogue::transform
