#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::tma {

// ═══════════════════════════════════════════════════════════════════════════
// 计算 swizzle atom 大小（即一条 TMA 指令在 inner 维度上加载的元素数）
//  - 无 swizzle(kSwizzleMode==0): 整块 BLOCK_INNER 作为一个 atom → 循环 1 次
//  - 有 swizzle: atom = kSwizzleMode / sizeof(dtype_t)
//    例: swizzle=128B, bf16(2B) → atom=64, BLOCK_K=128 → 循环 2 次
// ═══════════════════════════════════════════════════════════════════════════
template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

// ═══════════════════════════════════════════════════════════════════════════
// TMA copy 的统一入口：编译期根据 2D/3D、unicast/multicast、SM90/SM100 分派
//
// 模板参数:
//   BLOCK_INNER  - 需要切分为 swizzle atom 的维度（K-major 时是 BLOCK_K）
//   BLOCK_OUTER  - 另一个维度，每个 atom 内连续加载（K-major 时是 LOAD_BLOCK_M）
//   kSwizzleMode - swizzle 模式 (0/16/32/64/128 字节)
//   dtype_t      - 数据类型
//   kIs3DTMA     - 是否 3D TMA（批处理 batched GEMM 时使用）
//
// 函数参数:
//   desc_ptr         - TMA tensor map 描述符
//   barrier_ptr      - 事务 barrier，TMA 完成后 activate
//   smem_ptr         - 目标 shared memory 地址（当前 stage 的起始位置）
//   inner_idx        - inner 维度的全局起始索引
//   outer_idx        - outer 维度的全局起始索引
//   num_tma_multicast- multicast 数量 (1=单播, 2=双播到 2-CTA cluster)
//   batch_idx        - 3D TMA 的 batch 索引
//
// 决策树:
//   2D  unicast        → SM90_TMA_LOAD_2D         (单 CTA，标准 TMA)
//   2D  multicast SM100 → SM100_TMA_2SM_LOAD_2D  (Blackwell: 所有 CTA 各自发包)
//   2D  multicast SM90  → SM90_TMA_LOAD_MULTICAST_2D (Hopper: 仅 leader 发包)
//   3D  unicast        → SM90_TMA_LOAD_3D
//   3D  multicast SM100 → SM100_TMA_2SM_LOAD_3D
//   3D  multicast SM90  → SM90_TMA_LOAD_MULTICAST_3D
//
// ═══════════════════════════════════════════════════════════════════════════
template <uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t, bool kIs3DTMA = false>
CUTLASS_DEVICE void
copy(void const* desc_ptr, cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
     dtype_t* smem_ptr, const uint32_t& inner_idx, const uint32_t& outer_idx,
     const uint32_t& num_tma_multicast = 1, const uint32_t& batch_idx = 0) {
    DG_STATIC_ASSERT(static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
                     static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL), "Invalid cache hint");

    // 一条 TMA 指令在 inner 维度上加载的元素数（无 swizzle 时等于 BLOCK_INNER）
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

    // ═══════════════════════════════════════════
    //  分支 A: 2D TMA (Normal GEMM, Grouped GEMM)
    // ═══════════════════════════════════════════
    if constexpr (not kIs3DTMA) {

        // ─── A1: 2D 单播 (num_tma_multicast == 1) ───
        // 最基础场景：kNumMulticast=1，单 CTA，标准 TMA load
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                // 每次 TMA 指令加载 BLOCK_OUTER × BLOCK_INNER_ATOM 个元素
                // smem_ptr: 逐 atom 递增 BLOCK_OUTER * BLOCK_INNER_ATOM 个元素，保持 smem 内数据连续
                // inner_idx: 逐 atom 递增 BLOCK_INNER_ATOM，对应 HBM 中 inner 维度的偏移
                cute::SM90_TMA_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                             static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                             smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                             inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
            }
        } else {
            // ─── A2: 2D multicast ───
            // 2-CTA cluster 场景，两个 CTA 共享 TMA 加载的数据

            // A2a: Blackwell SM100 (CUDA arch >= 1000)
            // 使用 SM100_TMA_2SM_LOAD_2D，通过 shared::cluster 将数据分发到两个 CTA 的 smem
            // 所有 CTA 都执行 —— 不需要 block_rank 守卫
            // TMA 完成信号只发给 leader CTA 的 barrier
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                }

            // A2b: Hopper SM90 (CUDA arch >= 900)
            // 使用 SM90_TMA_LOAD_MULTICAST，通过 multicast bitmask 选择目标 CTA
            // 只有 leader CTA(block_rank==0) 发射 TMA 指令 —— 非 leader 跳过
            // bitmask = (1 << num_tma_multicast) - 1，例: 2 CTA → 0b11
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1,
                                                               static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx);
                    }
                }
            #endif
        }

    // ═══════════════════════════════════════════
    //  分支 B: 3D TMA (Batched GEMM: BHR_HDR_BHD 等)
    // ═══════════════════════════════════════════
    } else {

        // ─── B1: 3D 单播 ───
        // 3D TMA 多一个 batch_idx 参数，TMA 在第三维上按 batch 索引加载
        if (num_tma_multicast == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                cute::SM90_TMA_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                            inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
            }
        } else {
            // ─── B2: 3D multicast ───
            // 与 2D multicast 逻辑对称，增加 batch_idx

            // B2a: Blackwell SM100 (CUDA arch >= 1000)
            #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                    cute::SM100_TMA_2SM_LOAD_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                      static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                                                      smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                      inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                }

            // B2b: Hopper SM90 (CUDA arch >= 900)
            #elif (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
                if (cute::block_rank_in_cluster() == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++ i) {
                        cute::SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, reinterpret_cast<uint64_t*>(barrier_ptr),
                                                               (1 << num_tma_multicast) - 1,
                                                               static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                                               smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                                               inner_idx + i * BLOCK_INNER_ATOM, outer_idx, batch_idx);
                    }
                }
            #endif
        }
    }
}

} // namespace deep_gemm::tma
