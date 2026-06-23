#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

/*
═══════════════════════════════════════════════════════════════════════
GEMM Scheduler — work-stealing tile 调度器

核心问题: 给定 shape [M,K]×[K,N], 每个 CTA 处理 BLOCK_M×BLOCK_N 的 tile,
总共有 num_m_blocks × num_n_blocks 个 tile, 但可能有 kNumSMs 个 CTA 并发.
如何高效地将 tile 分配给 CTA, 同时最大化 L2 cache 命中率?

解决方案:
  1. Swizzle 调度: 将 tile 分组, CTA 在组间跳转时利用 L2 局部性
  2. Persistently-scheduled: 每个 CTA 处理完一个 tile 后立即抢下一个,
     无需新的 kernel launch, 实现动态负载均衡

当前文件实现了:
  - get_num_1d_blocks_per_group() — 选择最优分组大小
  - Scheduler 结构体:
      get_swizzled_block_idx() — swizzle 分组映射
      get_global_idx()       — 逻辑 tile → HBM 全局偏移
      get_next_block()       — 主调度循环的 next-tile 获取
═══════════════════════════════════════════════════════════════════════
*/

namespace deep_gemm::sched {

/*
╔══════════════════════════════════════════════════════════════════════╗
║  IndexType — get_global_idx 的索引维度选择                           ║
╚══════════════════════════════════════════════════════════════════════╝

MN   = M/N 维度偏移 (按 shape_m/shape_n 计算)
K    = K 维度偏移 (按 K 累积和计算, 用于 KGroupedContiguous)
SF_K = SF 的 K 维度偏移 (按 SF_K 累积和计算, 用于 KGroupedContiguous 的 SF 寻址)
*/
enum class IndexType {
    MN,
    K,
    SF_K,
};

/*
╔══════════════════════════════════════════════════════════════════════╗
║  get_num_1d_blocks_per_group — 自动选择最优 swizzle 分组大小          ║
╚══════════════════════════════════════════════════════════════════════╝

Swizzle 调度将 tile 分成若干 "组", CTA 在组间跳转.
分组大小 (8 还是 16) 直接影响:
  - 组内 CTA 数 = kNumSMs / 组大小 → 太少则 SM 空闲, 太多则组间跳跃频繁
  - L2 命中率: 组间跳跃会刷新 L2, 合适的 group size 减少 L2 miss

计算方式: 使 total_usage = tile 覆盖的元素总数最小化
  - kIsMulticastOnA → 分组在 N 维: usage = candidate * BLOCK_N + ceil(SMs/candidate) * BLOCK_M
    例: 8*128 + ceil(132/8)*128 = 1024 + 17*128 = 3200
        16*128 + ceil(132/16)*128 = 2048 + 9*128 = 3200
  - 否则分组在 M 维: usage = candidate * BLOCK_M + ceil(SMs/candidate) * BLOCK_N

只从 {8, 16} 中选: 太小(SM 空闲)和太大(频繁换行)都不好
*/
template <GemmType kGemmType, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs, bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
    // Select the best from candidates
    uint32_t num_best_blocks = 0, min_usage = cute::numeric_limits<uint32_t>::max();
    for (const auto candidate: {8u, 16u}) {
        const auto usage = kIsMulticastOnA ?
            candidate * BLOCK_N + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M: // Grouping on N
            candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N; // Grouping on M
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"

/*
╔══════════════════════════════════════════════════════════════════════╗
║  Scheduler — persistency-scheduled tile 分发器                       ║
╚══════════════════════════════════════════════════════════════════════╝

模板参数 (编译期常量, 零运行时开销):
  kGemmType        — 6 种 GEMM 类型之一
  BLOCK_M/BLOCK_N  — 每个 CTA 处理的 tile 大小
  kNumGroups       — 分组数 (MGroupedMasked/KGroupedContiguous)
  kNumMulticast    — 2-CTA multicast 数 (1 或 2)
  kIsMulticastOnA  — multicast 在 A 侧还是 B 侧 (true = A 侧/N 维分组)
  kNumSMs          — 可用 SM 数, 用于 swizzle 分组和持久化调度计算
  SF_K_ALIGNMENT   — SF K 对齐 (SM90: 128, SM100: gran_k*4)
  kNum1DBlocksPerGroup — 自动计算的最优分组大小 (8 或 16)

成员变量 (运行时, 每个 CTA 一份):
  current_iter         — 当前迭代计数, 用于 work-stealing
  num_blocks / num_m_blocks / num_n_blocks — tile 总数 / M 方向数 / N 方向数
  num_blocks_in_group  — 当前 swizzle 组的 tile 数
  is_peer_cta_alive    — SM90 multicast 时 peer CTA 是否有效
  current_group_idx    — 当前处理的分组/group 索引
  current_m_cumsum     — M 维累积 (MGroupedMasked)
  current_shape_k      — 当前 K 维大小 (KGroupedContiguous)
  current_k_cumsum     — K 维累积偏移 (KGroupedContiguous)
  current_sf_k_cumsum  — SF K 维累积偏移 (KGroupedContiguous)

调度原理:
  每个 CTA 以 blockIdx.x 为 ID, 通过 current_iter × kNumSMs + blockIdx.x
  计算自己下一个 tile, 实现 persistency-scheduled work-stealing:
    - CTA 0: tile 0, tile 132, tile 264, ...  (假设 132 SM)
    - CTA 1: tile 1, tile 133, tile 265, ...
    - 所有 CTA 无需同步, 独立递增 current_iter 即可
    - 天然负载均衡: 快的 CTA 多抢, 慢的 CTA 少抢
*/
template <GemmType kGemmType,
          uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,  // for k-grouped GEMM only: 128 on SM90 (float SF), gran_k * 4 on SM100 (packed UE8M0 SF)
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
    // Only used for contiguous psum layout
    uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
    // Only used for k-grouped layout
    uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, current_sf_k_cumsum = 0;
    uint32_t next_group_idx, next_shape_k;

    // Only used for k-grouped gemm
    // 在 grouped_layout 中查找下一个 non-zero K 维度
    CUTLASS_DEVICE void get_next_k_group(uint32_t &group_idx, uint32_t &shape_k) const {
        for (; group_idx < kNumGroups; ++ group_idx) {
            shape_k = grouped_layout[group_idx];
            if (shape_k > 0)
                break;
        }
    }

    /*
    ═══════════════════════════════════════════════════════════════════════
    构造函数 — 根据 GEMM 类型初始化 tile 计数和分组参数
    ═══════════════════════════════════════════════════════════════════════

    参数:
      shape_m / shape_n / shape_k — GEMM 的总维度
      grouped_layout — 分组布局数组 (Null 表示非分组, 否则含义因类型而异)
        - MGroupedContiguous: grouped_layout[i] = 第 i 个 expert 的 token 数
        - MGroupedMasked: grouped_layout[i] = 第 i 个 expert 的 token 数
        - KGroupedContiguous: grouped_layout[i] = 第 i 组 K 的大小
    */
    // ReSharper disable once CppPossiblyUninitializedMember
    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m, const uint32_t& shape_n,
                                       const uint32_t& shape_k, int* grouped_layout = nullptr) {
        num_m_blocks = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks = math::ceil_div(shape_n, BLOCK_N);
        current_shape_k = shape_k;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            // Normal: 简单 M×N 网格, num_blocks = num_m × num_n
            num_blocks = num_m_blocks * num_n_blocks;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            // MGroupedContiguous: 所有 expert 的 token 拼接成一个大矩阵
            // grouped_layout 记录每个 token 行属于哪个 expert
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            // MGroupedMasked: 每个 expert 独立存储, num_blocks 在 get_next_block 中动态计算
            this->grouped_layout = grouped_layout;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // MGroupedContiguous + 部分和: 第一组的 psum_m 决定初始 tile 数
            this->grouped_layout = grouped_layout;
            current_psum_m = grouped_layout[0];
            num_m_blocks = math::ceil_div(current_psum_m, BLOCK_M);
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            // KGroupedContiguous: 多组 K 拼接 (如 SwiGLU gate+up)
            // current_shape_k 从小到大的 K 组起始大小
            num_blocks = num_m_blocks * num_n_blocks;
            this->grouped_layout = grouped_layout;
            get_next_k_group(current_group_idx, current_shape_k);
            next_group_idx = current_group_idx + 1;
            get_next_k_group(next_group_idx, next_shape_k);
        }
    }

    /*
    ═══════════════════════════════════════════════════════════════════════
    get_swizzled_block_idx — swizzle 分组: 将扁平 block_idx 映射为 (m, n)
    ═══════════════════════════════════════════════════════════════════════

    目标: 最大化 L2 cache 命中率.
    原理: 将 tile 分成 kNum1DBlocksPerGroup 个一组的 "swizzle 组".
    组内 tile 在 M 或 N 方向连续, CTA 处理完一组内所有 tile 后跳到下一组.

    示例 (kNum1DBlocksPerGroup=8, kIsMulticastOnA=false, 分组在 M):
      平铺顺序: (M0,N0), (M0,N1), ..., (M0,Nk), (M1,N0), ...
      分组后在 M: 组的顺序按 M 展开 [(M0,N0)..(M7,N0), (M8,N0)..(M15,N0), ...]

    kIsMulticastOnA 的影响:
      - A 侧 multicast → 分组在 N (M 对齐不重要)
      - B 侧 multicast → 分组在 M (N 对齐不重要)

    SM90 特殊处理: 如果组大小不能被 kNumMulticast 整除, 则拆分末尾
    不齐组, 因为 SM90 multicast 可以动态禁用.
    */
    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");

        // Swizzle for better L2 usages
        // primary = 分组维度 (swizzle 展开方向)
        const auto primary_num_blocks = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
        const auto secondary_num_blocks = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
        const auto num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup;
        const auto group_idx = block_idx / num_blocks_per_group;
        auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
        auto in_group_idx = block_idx % num_blocks_per_group;
        num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

        // Fix unaligned TMA multicast
        // NOTES: for SM90 only, as SM90 can dynamically disable TMA multicast
        // while SM100 uses 2-CTA, which can not be dynamically disabled
#if __CUDA_ARCH__ < 1000
        // SM90 multicast 对齐修正: 组大小奇数时, 末尾 1 个只能单播
        if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {
            if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
                num_blocks_in_group = num_blocks_in_group ^ 1;  // 截断为偶数
            } else {
                in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
                first_block_idx += num_blocks_in_group ^ 1;
                num_blocks_in_group = 1;  // 最后一个 tile 单播
            }
        }
#endif

        // Convert to final M/N block indices
        // `kIsMulticastOnA == true` leads to groups on N
        if constexpr (kIsMulticastOnA) {
            // 分组在 N: (m0,n0), (m1,n0), ..., (mk,n0), (m0,n1), ...
            // m_block_idx: 组内除分组维度外逐个递增
            // n_block_idx: first_block_idx 作为分组的起始
            m_block_idx = in_group_idx / num_blocks_in_group;
            n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        } else {
            // 分组在 M: (m0,n0), (m0,n1), ..., (m0,nk), (m1,n0), ...
            m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
            n_block_idx = in_group_idx / num_blocks_in_group;
        }
    }

    /*
    ═══════════════════════════════════════════════════════════════════════
    get_global_idx — 逻辑 tile 索引 → 全局 HBM 偏移 (单位: 元素)
    ═══════════════════════════════════════════════════════════════════════

    这是 kernel 中连接调度器和内存寻址的核心桥梁.

    模板参数:
      kWithGroupOffset — 是否需要分组偏移 (编译期 bool)
        - Normal: 忽略
        - MGroupedContiguous: true 时用 grouped_layout 查 expert 编号
        - MGroupedMasked: true 时用 current_group_idx
        - KGroupedContiguous: true 时按 IndexType 选 k_cumsum / sf_k_cumsum / group_idx
      kIndexType — MN / K / SF_K, 决定偏移计算方式

    参数:
      shape_dim  — 该维度总大小 (如 shape_m, shape_n, shape_k, shape_sfa_k)
      block_size — tile 在该维度的大小 (如 BLOCK_M, BLOCK_N, BLOCK_K)
      block_idx  — 逻辑 tile 索引
      m_block_idx— M 块索引 (仅 MGroupedContiguous 需要, 用于查 expert 编号)

    返回值: HBM 中该 tile 起始位置的元素偏移量
    */
    template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                             const uint32_t& block_idx, const uint32_t& m_block_idx = 0) {
        if constexpr (kGemmType == GemmType::Normal) {
            // Normal GEMM: 简单线性映射
            // offset = block_idx × block_size
            return block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            // MGroupedContiguous: A 矩阵所有 expert 连续拼接
            // grouped_layout[m_row] = expert_id (每个 token 行属于哪个 expert)
            // B 矩阵的 N 维按 expert 分组: expert_id * shape_n 偏移
            // offset = expert_id * shape_dim + block_idx * block_size
            const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked or kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            // MGroupedMasked: 每个 expert 独立存储, 通过 current_group_idx 跳转
            // offset = current_group_idx * shape_dim + block_idx * block_size
            const auto offset = kWithGroupOffset ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            // KGroupedContiguous: 多组 K 拼接
            // IndexType::MN   → offset = current_group_idx * shape_dim (按组跳跃 M/N 维度)
            // IndexType::K    → offset = current_k_cumsum (按 K 累积和跳跃)
            // IndexType::SF_K → offset = current_sf_k_cumsum (按 SF K 累积和跳跃)
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
            // Batched: 多个独立矩阵乘
            // 普通维度 (MN): offset = 0 (batch_idx 通过 TMA 3D 索引)
            // SF_K: offset = current_group_idx * shape_dim (batch SF 寻址)
            // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    // For swap A/B and psum layout only
    // 获取 M block 内的有效对齐 M (考虑 psum layout 的尾部不完整 block)
    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(const uint32_t& m_block_idx) const {
        constexpr uint32_t UMMA_STEP_N = 16;
        DG_STATIC_ASSERT(BLOCK_M % UMMA_STEP_N == 0, "Invalid alignment");
        if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout)
            return math::align(m_block_idx == last_psum_m / BLOCK_M + num_m_blocks - 1 ? current_psum_m - m_block_idx * BLOCK_M : BLOCK_M, UMMA_STEP_N);
        return BLOCK_M;
    }

    /*
    ═══════════════════════════════════════════════════════════════════════
    get_next_block — 主调度循环: 获取下一个要处理的 (m_block_idx, n_block_idx)
    ═══════════════════════════════════════════════════════════════════════

    调度公式:
      next_block_idx = (++current_iter) × kNumSMs + blockIdx.x

    这是典型的 persistency-scheduled work-stealing:
      - 按 SM 数分组, 每个 CTA 按 blockIdx.x 偏移
      - current_iter 递增 → 自动下一轮
      - 快的 CTA 多迭代, 慢的 CTA 少迭代 → 天然负载均衡

    6 种 GEMM 类型的分支:
      Normal / MGroupedContiguous:
        直接 swizzle 调度, 无额外分组管理
      MGroupedMasked:
        while 循环跨 expert 组, 每组独立计算 num_m_blocks
        当前 expert 的 tile 用完后 current_group_idx++ 切到下一 expert
      MGroupedContiguousWithPsumLayout:
        while 循环跨 partial-sum 组, 每组动态计算 num_m_blocks
        并更新 last_psum_m / current_psum_m
      KGroupedContiguous:
        while 循环跨 K 组, 维护 current_k_cumsum / current_sf_k_cumsum
        当前 K 组的 tile 用完后切到下一组
      Batched:
        线性块索引, current_group_idx = block_idx / num_blocks
        不需要 swizzle (每个 batch 独立)
    */
    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        // persistency work-stealing: 每个 CTA 独立计算下一个 tile 编号
        const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked) {
            // ─── MGroupedMasked: 循环跨 expert 组 ───
            // 每个 expert 有独立的 M 维度, 用完后跳到下一个 expert
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                // Move to check the next group
                current_group_idx ++, current_m_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { 
            // ─── MGroupedContiguousWithPsumLayout: 循环跨 partial-sum 组 ───
            while (true) {
                // Within current group
                if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks)
                    break;

                // Move to check the next group
                if (++ current_group_idx == kNumGroups)
                    return false;

                // NOTES: `num_m_blocks` varies with the increase of the group index
                last_psum_m = math::align(current_psum_m, BLOCK_M);
                current_psum_m = grouped_layout[current_group_idx];
                current_m_block_cumsum += num_m_blocks;
                num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M);
            }

            get_swizzled_block_idx(next_block_idx - current_m_block_cumsum * num_n_blocks, m_block_idx, n_block_idx);

            // NOTES: `last_psum_m` is aligned with block M
            m_block_idx += last_psum_m / BLOCK_M;
        } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            // ─── KGroupedContiguous: 循环跨 K 组 ───
            // 所有 CTA 先处理第一组的所有 M/N tile, 再处理下一组
            // 维护 K 累积偏移: current_k_cumsum 和 current_sf_k_cumsum
            while (true) {
                // End of the task
                if (current_group_idx == kNumGroups)
                    return false;

                // Within current group
                // num_blocks 是固定的 M×N 网格, 每处理完一轮 (num_blocks 个 tile)
                // 就前进到下一个 K 组
                if (next_block_idx < (current_num_valid_groups + 1) * num_blocks)
                    break;

                // Move to check the next group
                current_k_cumsum += current_shape_k;
                current_sf_k_cumsum += math::ceil_div(current_shape_k, SF_K_ALIGNMENT);
                current_num_valid_groups ++;

                current_group_idx = next_group_idx ++;
                current_shape_k = next_shape_k;
                get_next_k_group(next_group_idx, next_shape_k);
            }

            get_swizzled_block_idx(next_block_idx - current_num_valid_groups * num_blocks, m_block_idx, n_block_idx);
        } else if constexpr (kGemmType == GemmType::Batched) {
            // ─── Batched: 跨 batch 线性调度 ───
            if (next_block_idx >= num_blocks * kNumGroups)
                return false;

            current_group_idx = next_block_idx / num_blocks;
            const auto block_idx = next_block_idx - current_group_idx * num_blocks;
            if constexpr (kIsMulticastOnA) {
                m_block_idx = block_idx / num_n_blocks;
                n_block_idx = block_idx % num_n_blocks;
            } else {
                m_block_idx = block_idx % num_m_blocks;
                n_block_idx = block_idx / num_m_blocks;
            }
        } else {
            // ─── Normal / MGroupedContiguous: 标准 swizzle 调度 ───
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

    /*
    ═══════════════════════════════════════════════════════════════════════
    is_tma_multicast_valid — SM90 专用: 当前 tile 的 TMA multicast 是否有效
    ═══════════════════════════════════════════════════════════════════════

    SM90 multicast 要求两个 CTA 的 tile 在同一 expert 内 (MGroupedContiguous 时).
    这确保了 multicast 的目标 SMEM 地址在两个 CTA 间是连续且对齐的.
    若跨 expert, 则不能使用 multicast (SM90 可动态禁用, SM100 的 2-CTA 不可).
    */
    // For SM90 only
    CUTLASS_DEVICE bool is_tma_multicast_valid(const uint32_t& m_block_idx) const {
        if (num_blocks_in_group == 1)
            return false;
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::MGroupedMasked or
                      kGemmType == GemmType::KGroupedContiguous or kGemmType == GemmType::Batched or
                      kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return true;
        } else {
            DG_STATIC_ASSERT(kGemmType == GemmType::MGroupedContiguous, "Invalid Gemm type");
            if constexpr (kIsMulticastOnA) {
                return true;
            } else {
                // 关键检查: 两个相邻 M tile 属于同一个 expert?
                const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];
                const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];
                return group_idx == peer_group_idx;
            }
        }
    }

    /*
    ═══════════════════════════════════════════════════════════════════════
    is_computation_valid — SM90 专用: 当前 M 偏移位置的 MMA 计算是否有效
    ═══════════════════════════════════════════════════════════════════════

    MGroupedContiguous 时, 末尾 expert 的 token 可能不足 BLOCK_M.
    某些 M 范围内的 MMA 波需要跳过 (无效行). grouped_layout[i] < 0 表示无效.
    MGroupedMasked 时, 若 m_offset 超出当前 expert 的 token 数则跳过.
    */
    // For SM90 only
    // ReSharper disable once CppNotAllPathsReturnValue
    CUTLASS_DEVICE bool is_computation_valid(const uint32_t& m_block_idx, const uint32_t& m_offset) const {
        if constexpr (kGemmType == GemmType::Normal or kGemmType == GemmType::Batched) {
            return true;
        } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
            return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;
        } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
            return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
        } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
            return m_offset + m_block_idx * BLOCK_M < current_psum_m;
        } else {
            // Unreachable — KGroupedContiguous 不需要此检查
            DG_TRAP_ONLY_DEVICE_ASSERT(false);
        }
    }
};

#pragma clang diagnostic pop

} // namespace deep_gemm::sched
