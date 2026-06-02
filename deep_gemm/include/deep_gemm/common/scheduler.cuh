#pragma once

#include <deep_gemm/common/types.hpp>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

enum class IndexType {
    MN,
    K,
    SF_K,
};

template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto& candidate: {8u, 16u}) {
        const auto& usage = kIsMulticastOnA ?
                    candidate * BLOCK_N + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
                    candidate * BLOCK_M + constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,  // for k-grouped GEMM only: 128 (SM90 float SF) or 512 (SM100 UE8M0 SF)
          uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<kGemmType, BLOCK_M, BLOCK_N, kNumSMs, kIsMulticastOnA>()>
struct Scheduler {
    int current_iter = -1;

    // Block configs
    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;

    // For SM90 multicast checks
    uint32_t num_blocks_in_group;
    bool is_peer_cta_alive = true;

    // For grouped GEMM
    int* grouped_layout;
    uint32_t current_group_idx = 0;
    // Only used for masked layout
    uint32_t current_m_cumsum = 0;
    // Only used for countiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    uint32_t next_group_idx, next_shape_k;

    // Only used for k-grouped gemm
    __device__ __forceinline__ void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = __ldg(grouped_layout + group_idx);
            if (shape_k > 0)
                break;
        }
    }

    // ReSharper disable once CppPossiblyUninitializedMember
    __device__ __forceinline__ explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n, const uint32_t& shape_k,
                                                  int* grouped_layout = nullptr) {
        num_m_blocks = ceil_div(shape_m, BLOCK_M);
        num_n_blocks = ceil_div(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            this->grouped_layout = grouped_layout;
            current_psum_m = __ldg(grouped_layout);
            num_m_blocks = ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            this->grouped_layout = grouped_layout;
            get_next_k_group(current_group_idx, current_shape_k);
            next_group_idx = current_group_idx + 1;
            get_next_k_group(next_group_idx, next_shape_k);
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        const auto& primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto& secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto& num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto& group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    // ========================================================================
    // get_global_idx — 将逻辑块索引转换为全局内存偏移
    // ========================================================================
    // 模板参数：
    //   kWithGroupOffset — 是否需要加上分组偏移（多 expert 拼接时的跨组地址跳转）
    //                      调用方根据当前 GEMM 类型决定是否需要，如 MGroupedMasked 需要，
    //                      Normal 则不需要（编译期优化掉）
    //   kIndexType       — 索引类型，决定偏移计算方式：
    //                      MN   = M/N 维度（按 shape_dim 跳转）
    //                      K    = K 维度（按 K 累计和跳转）
    //                      SF_K = SF 的 K 维度（按 SF_K 累计和跳转）
    //
    // 函数参数：
    //   shape_dim   — 该维度总大小（如 shape_m / shape_n / shape_k）
    //   block_size  — 每个块的该维度大小（如 BLOCK_M / BLOCK_N / BLOCK_K）
    //   block_idx   — 当前块在组内的逻辑索引
    //   m_block_idx — M 方向块索引（仅 MGroupedContiguous 需要，用于查找该行所属 expert）
    //
    // 返回值：全局内存中的元素偏移量（单位：元素个数）
    // ========================================================================
    //-
    // `kGemmType` 有 6 种值，对应不同的**矩阵组织方式**：
    // ### 6 种 GEMM 类型
    // |                    类型               |               含义            |                     典型场景                                      |
    // | **Normal**                           | 单次矩阵乘 C=A×B               | 单个大矩阵乘法                                                     |
    // | **MGroupedContiguous**               | M 维分组，**连续拼接**          | MoE 推理：多个 expert 的 token 在 M 方向拼成一个大矩阵              |
    // | **MGroupedMasked**                   | M 维分组，**独立存储+掩码**     | MoE 推理：每个 expert 矩阵独立，用掩码标记哪些 token 属于哪个 expert |
    // | **KGroupedContiguous**               | K 维分组，连续拼接              | 多组权重在 K 方向拼接（如 gate+up 拼接后一次算完）                   |
    // | **Batched**                          | Batch 矩阵乘：多个独立的小矩阵乘 | Batch MatMul，每个 batch 独立的 A×B                                |
    // | **MGroupedContiguousWithPsumLayout** | MGroupedContiguous + Psum 布局 | MoE 训练：需要部分和累加的场景                                      |

    // ### 核心区别图示
    // Normal:           一个 [M,K] × [K,N] → [M,N]
    // MGroupedContiguous:  [M0+M1+M2, K] × [K, N] → [M0+M1+M2, N]
    //                      ↑ 所有 expert 的 token 连续拼在一起，共享同一个 B
    // MGroupedMasked:      expert0: [M0,K]×[K,N]   expert1: [M1,K]×[K,N]
    //                      ↑ 每个 expert 独立存储，通过 group_idx 索引不同的 B
    // KGroupedContiguous:  [M, K0+K1] × [K0+K1, N] → [M, N] (分两次累加)
    //                      ↑ K 方向多组拼接，如 SwiGLU 的 gate+up
    // Batched:             batch0: [M,K]×[K,N]   batch1: [M,K]×[K,N]
    //                      ↑ 完全独立的多次矩阵乘

    // 不同类型影响：
    // 1. **调度方式**：MGrouped 需要按 expert 分配 tile，Normal 直接线性扫描
    // 2. **内存寻址**：MGroupedContiguous 需要 `grouped_layout` 查 expert 编号，MGroupedMasked 用 `current_group_idx` 跳转
    // 3. **SF 加载**：KGrouped 的 SF 在 K 方向按累计和偏移
    // 4. **Epilogue**：MGroupedMasked 的输出需要写回各 expert 独立的位置
    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            // 普通矩阵乘：无需分组偏移，直接 块号 × 块大小
            // 例：m_block_idx=3, BLOCK_M=128 → 返回 384
            return block_idx * block_size;

        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            // M 分组连续布局：多个 expert 的矩阵在 M 方向连续拼接
            // 需要先查出当前行属于哪个 expert（offset），再跳到该 expert 的区域
            // grouped_layout[m_block_idx * BLOCK_M] 存的是该 token 对应的 expert 编号
            // offset * shape_dim 跳过前面 offset 个 expert 的所有行
            // 例：expert=2, shape_m=4096, m_block_idx=3, BLOCK_M=128
            //     → 2*4096 + 3*128 = 8192 + 384 = 8576
            const auto offset = kWithGroupOffset ? cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M)) : 0;
            return offset * shape_dim + block_idx * block_size;

        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // M 分组掩码布局：每个 expert 的矩阵独立存储，通过 current_group_idx 索引
            // current_group_idx 是调度器维护的当前 expert 编号
            // offset * shape_dim 跳过前 offset 个 expert 的矩阵区域
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;

        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            // K 分组连续布局：多个 GEMM 在 K 方向拼接
            // 不同 IndexType 的偏移方式不同：
            //   MN   — 组号 × 维度大小（同 MGroupedMasked）
            //   K    — K 累计和（前面各组 K 维度之和，因为每组 K 大小可能不同）
            //   SF_K — SF 的 K 累计和（类似 K，但粒度不同）
            auto offset = 0;
            if constexpr (kWithGroupOffset) {
                if constexpr (kIndexType == IndexType::MN)
                    offset = current_group_idx * shape_dim;
                else if constexpr (kIndexType == IndexType::K)
                    offset = current_k_cumsum;
                else if constexpr (kIndexType == IndexType::SF_K)
                    offset = current_sf_k_cumsum;
            }
            return offset + block_idx * block_size;

        } else if constexpr (kGemmType == GemmType::Batched) {
            // Batch 矩阵乘：各矩阵独立，M/N 维度无需偏移
            // 仅 SF_K 类型需要按 batch 索引偏移（SF 按 batch 分区存储）
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + current_group_idx)), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = align(current_psum_m, 128u);
                current_psum_m = __ldg(grouped_layout + current_group_idx);
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with 128
            m_block_idx += last_psum_m / BLOCK_M;
            DG_STATIC_ASSERT(128 % BLOCK_M == 0, "Invalid BLOCK_M");
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                if (next_block_idx < (current_num_valid_groups + 1) * num_m_blocks * num_n_blocks)
                    break;

                // Move to check the next group
                current_k_cumsum += current_shape_k;
                current_sf_k_cumsum += ceil_div(current_shape_k, SF_K_ALIGNMENT);
                current_num_valid_groups ++;

                current_group_idx = next_group_idx ++;
                current_shape_k = next_shape_k;
                get_next_k_group(next_group_idx, next_shape_k);
            }

            get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_m_blocks * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto& block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            // For SM90 only
            // NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            is_peer_cta_alive = num_n_blocks % kNumMulticast == 0 or                  // Always aligned on N (constant bypass)
                                num_m_blocks % kNumMulticast == 0 or                  // Always aligned on M (constant bypass)
                                (next_block_idx ^ 1) < num_blocks;                    // Peer CTA in bound
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }

    // For SM90 only
    __device__ __forceinline__ bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                const auto& group_idx = __ldg(grouped_layout + m_block_idx * BLOCK_M);
                const auto& peer_group_idx = __ldg(grouped_layout + (m_block_idx ^ 1) * BLOCK_M);
                return group_idx == peer_group_idx;
            }
        }
    }

    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    __device__ __forceinline__ bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return __ldg(grouped_layout + m_offset + m_block_idx * BLOCK_M) >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < __ldg(grouped_layout + current_group_idx);
        } else {
            // Unreachable 
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm
