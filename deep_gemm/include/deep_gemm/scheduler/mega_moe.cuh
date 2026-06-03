#pragma once

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm::sched {

// Computation phase for the current block
enum class BlockPhase {
    None = 0, 
    Linear1 = 1,
    Linear2 = 2
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumExpertsPerWave,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1BlockKs = L1_SHAPE_K / BLOCK_K,
          uint32_t kNumL2BlockKs = L2_SHAPE_K / BLOCK_K>
struct MegaMoEScheduler {
    DG_STATIC_ASSERT(L1_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % BLOCK_N == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(kNumExpertsPerRank % kNumExpertsPerWave == 0, "Invalid wave config");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL1BlockNs % 2 == 0, "L1 N block count must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumL2BlockNs % 2 == 0, "L2 N block count must be even for 2-CTA cluster");

    // Arrival counts
    const layout::Workspace& workspace;

    // ⚠️ Scheduler state
    // 初始值 为  BlockPhase::Linear1
    BlockPhase next_phase = BlockPhase::Linear1;

    // Current expert and block indices
    uint32_t current_local_expert_idx = 0; // 当前正在处理的本地 expert 编号（0 ~ kNumExpertsPerRank-1）
    uint32_t current_num_tokens = 0; // 当前 expert 接收到的 token 总数（由 get_num_tokens() 填充）
    uint32_t current_pool_block_offset = 0; //当前 expert 在 token 池中的起始 block 偏移（前面所有 expert 占了多少 block）
    // ⚠️ block_idx 是调度引擎的驱动力——它递增步进，映射到 (m, n) 二维坐标；其余变量描述当前 expert 的上下文。
    uint32_t block_idx = 0; // 初始值blockIdx.x。 全局一维任务索引，所有 SM 以 += kNumSMs 步进交错取任务
    uint32_t m_block_idx = 0; //当前 GEMM block 的 M 维度块号（由 block_idx / kNumL?BlockNs 算出）
    uint32_t n_block_idx = 0;  // 当前 GEMM block 的 N 维度块号（由 block_idx % kNumL?BlockNs 算出）
    /*
    block_idx (一维) ──分解──► m_block_idx, n_block_idx (二维)
                         m_block_idx = block_idx / kNumL?BlockNs
                         n_block_idx = block_idx - m_block_idx * kNumL?BlockNs

    current_pool_block_offset ──标记──► 当前 expert 的 token 在池中的起始位置
                            pool_token_idx = current_pool_block_offset * BLOCK_M + token_idx_in_expert

    */


    // Pre-cached per-expert token counts (filled during `for_each_block` init)
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

    CUTLASS_DEVICE explicit MegaMoEScheduler(const layout::Workspace& workspace): workspace(workspace) {
        block_idx = blockIdx.x;
    }

    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {
        // 当前 wave 的结束 expert 索引
        return math::align(current_local_expert_idx + 1, kNumExpertsPerWave);
        /*
        假设 kNumExpertsPerWave=4：
        current_local_expert_idx	+1	align(_, 4)	含义
        0	1	4	wave [0,4)
        1	2	4	wave [0,4)
        2	3	4	wave [0,4)
        3	4	4	wave [0,4)
        4	5	8	wave [4,8)
        5	6	8	wave [4,8)
        就是算出当前 expert 所在 wave 的右边界（不含），
        供 fetch_next_l1/l2_block() 的 while 循环判断：超出这个边界就说明当前 wave 的所有 expert 都分配完了，该切换 L1→L2 或进入下一个 wave。
        */
    }

    // ========================================================================
    // get_num_tokens(): 查询本地第 expert_idx 个 expert 接收到的 token 总数
    // ========================================================================
    // 数据来源：寄存器数组 stored_num_tokens_per_expert[]，由 fetch_expert_recv_count() 填充
    // 存储布局：每 lane 负责 expert (i*32 + lane_idx) 的计数（warp 级分片）
    //
    // 查询方式：
    //   1. 当前 lane 检查 expert_idx 是否落在自己负责的范围 (i*32 + lane_idx)
    //   2. 命中的 lane 从自己的 stored_num_tokens_per_expert[i] 取值到 valid_value
    //   3. ptx::exchange(valid_value, lane_idx) 在 warp 内按 lane_idx 做洗牌，
    //      将持有目标 expert 数据的 lane 的 valid_value 广播到所有 lane
    // ========================================================================
    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // 只有负责 expert_idx == i*32 + lane_idx 的 lane 会更新 valid_value
            // 其他 lane 的 valid_value 保持原值（但最终只有命中 lane 的值会被 exchange 取走）
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        // ptx::exchange: 等价于 __shfl_sync，将 lane (expert_idx % 32) 的 valid_value 广播到所有 lane
        // 因为 expert_idx 对应的数据恰好存储在 lane_idx == expert_idx % 32 的 lane 上
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // 当前 lane 负责的 expert 编号 = i * 32 + lane_idx
            // 只有当这个编号 < expert_idx 时，才把它的 block 数累加
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        // warp 内所有 lane 的 num_blocks 求和 = 所有 < expert_idx 的 expert 的 block 总数
        return __reduce_add_sync(0xffffffff, num_blocks);

        /*
        具体例子
        假设 kNumExpertsPerRank=4，BLOCK_M=128，4 个 expert 分别有 100, 200, 50, 300 个 token：

        get_pool_block_offset(0) = 0（没有前面的 expert）
        get_pool_block_offset(1) = ceil(100/128) = 1
        get_pool_block_offset(2) = 1 + ceil(200/128) = 1 + 2 = 3
        get_pool_block_offset(3) = 3 + ceil(50/128) = 3 + 1 = 4
        这样 expert 3 的 token 就从池的第 4 个 block 开始存放。

        Warp 并行求和
        关键在于 __reduce_add_sync(0xffffffff, num_blocks)——这不是简单的串行累加：

        每个 lane 只负责部分 expert（i * 32 + lane_idx），各自累加自己负责范围内 < expert_idx 的 block 数
        __reduce_add_sync 把 32 个 lane 的部分和加起来，得到总和
        这比单个线程遍历所有 expert 快得多，是典型的 warp 级并行前缀和（只是求和，不求前缀）。
        */
    }

    CUTLASS_DEVICE void advance_expert_idx() {
        // 推进到下一个 expert
        current_pool_block_offset += get_current_num_m_blocks();
        current_local_expert_idx += 1;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
    }

    CUTLASS_DEVICE void set_expert_idx(const uint32_t& expert_idx) {
        current_local_expert_idx = expert_idx;
        current_num_tokens = get_num_tokens(expert_idx);
        current_pool_block_offset = get_pool_block_offset(expert_idx);
    }

    CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
        return current_pool_block_offset;
    }

    CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
        // 当前 expert 有多少个 M 方向的 block
        return math::ceil_div(current_num_tokens, BLOCK_M);
    }

    template <bool kDoUMMAAligned = false>
    CUTLASS_DEVICE uint32_t get_valid_m() const {
        // 当前 block 实际有效的 M 行数
        const auto m = cute::min(current_num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
        //kDoUMMAAligned=true：向上对齐到 16 的倍数（如 72→80），用于 UMMA 指令——UMMA 要求 M 维度 16 对齐，否则会读越界
        return kDoUMMAAligned ? math::align(m, 16u) : m;
    }

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        /*
        核心思想：block_idx 一维索引映射到二维 (m, n)
        所有 GEMM block 被逻辑排成一维序列，block_idx 就是这个一维索引
        */
         // 当前 wave 结束 expert
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            // 当前 expert 的 M block 数
            const auto num_m_blocks = get_current_num_m_blocks();
             // 一维 → 二维：取 m
             // block_idx 映射到 m_block
            m_block_idx = block_idx / kNumL1BlockNs;
            // m 没超出范围 → 找到了
            if (m_block_idx < num_m_blocks)
                return true;
            // ⚠️ 当 m_block_idx >= num_m_blocks 时，说明 block_idx 已经超出了当前 expert 的 M 范围，
            // 需要减去当前 expert 消耗的 block 数（num_m_blocks * kNumL1BlockNs），然后推进到下一个 expert。
            // Current expert is fully assigned, move to the next
            
            // ⚠️ block_idx 是调度引擎的驱动力——它递增步进，映射到 (m, n) 二维坐标；其余变量描述当前 expert 的上下文。
            // 当前 expert 的 block 已经全部分配完，跳到下一个 expert
            block_idx -= num_m_blocks * kNumL1BlockNs;
            // 推进到下一个 expert  // current_expert_idx++
            advance_expert_idx();
        }
         // 当前 wave 所有 expert 的 L1 都分配完了
        return false;
    }

    CUTLASS_DEVICE bool fetch_next_l2_block() {
        const auto wave_end_expert_idx = get_wave_expert_end_idx();
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            if (block_idx < num_m_blocks * kNumL2BlockNs) { // ← 区别：判断条件不同
                m_block_idx = block_idx / kNumL2BlockNs; // ← 区别：用 L2 的 N block 数
                return true;
            }

            // Current expert is fully assigned, move to the next
            block_idx -= num_m_blocks * kNumL2BlockNs;  // ← 区别：用 L2 的 N block 数
            advance_expert_idx();
        }
        return false;
    }

    // Core state machine: assigns the next block
    CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
        while (true) {
            // 所有expert计算结束
            if (current_local_expert_idx >= kNumExpertsPerRank)
                break;

            //⚠️ 关键变量 block_idx — 全局任务分配器
            //⚠️ block_idx 是整个调度的核心，它是一个 全局递增的块索引，所有 SM 通过 block_idx += kNumSMs 来 交错分配任务：
            //初始值：block_idx = blockIdx.x（每个 SM 从自己的编号开始）
            //每分配一个 block：block_idx += kNumSMs（跳过其他 SM 的任务）
            //这意味着所有 GEMM block 被逻辑上排成一个一维序列，kNumSMs 个 SM 以 round-robin 方式各取各的。

            // ⚠️ 初始值 next_phase = BlockPhase::Linear1
            if (next_phase == BlockPhase::Linear1) {
                if (fetch_next_l1_block()) { // true / false
                    // ⚠️ Found a new L1 block， block_idx 到 (m_block, n_block) 的映射
                    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
                    // ⚠️ Jump to next block，kNumSMs 个 SM 以 round-robin 方式各取各的
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // L1 for the current wave is complete, transition to L2
                    // ⚠️ Wave 回退机制.当 L1 完成后切换到 L2 时，有一个关键操作：
                    next_phase = BlockPhase::Linear2;
                    // ⚠️ L1 完成后，回退到当前 wave 的起始 expert
                    // math::align<false> 是向下对齐。例如 kNumExpertsPerWave=4，当前 current_local_expert_idx=4（已推进到下一个 wave 的起点），4-1=3，向下对齐到 4 的倍数得到 0，回退到 wave 的起始 expert 0。
                    // 这样 L2 可以从同一个 wave 的第一个 expert 重新开始分配 block。
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));
                }

            } else {
                if (fetch_next_l2_block()) {
                    // Found a new L2 block
                    n_block_idx = block_idx - m_block_idx * kNumL2BlockNs;
                    // Jump to next block
                    block_idx += kNumSMs;
                    return {BlockPhase::Linear2, current_local_expert_idx, m_block_idx, n_block_idx};
                } else {
                    // Move to L1 of the next wave
                    next_phase = BlockPhase::Linear1;
                }
            }
        }
        /*
        完整执行示例
        假设 2 个 SM、2 个 expert、kNumExpertsPerWave=2、每个 expert 3 个 M block、kNumL1BlockNs=2：
        SM 0 的 block_idx 变化:
        初始: block_idx=0
        L1: expert0_m0_n0 → block_idx=2
        L1: expert0_m0_n1 → block_idx=4
        L1: expert0_m1_n0 → block_idx=6
        ... (expert0 完成, 跳到 expert1)
        L1: expert1_m0_n0 → block_idx=8
        ...
        (wave 0 的 L1 全部完成 → 切换到 L2, 回退到 expert 0)
        L2: expert0_m0_n0 → block_idx=10
        ...
        核心设计思想：用单一的 block_idx 一维索引 + += kNumSMs 步进，实现了无锁的跨 SM 负载均衡——token 数多的 expert 自然分到更多 block，少的 expert 快速跳过。
        */

        // 所有expert计算结束
        // All waves and experts are fully processed
        return {BlockPhase::None, 0, 0, 0};
    }

    // ========================================================================
    // fetch_expert_recv_count(): 从 workspace 读取每个本地 expert 的 token 接收总数
    // ========================================================================
    // 填充 stored_num_tokens_per_expert[]，供后续 get_num_tokens() 查询
    //
    // 数据来源：workspace 中的 expert_recv_count_sum[expert_idx]
    //   - 低 32 位：该 expert 从所有 rank 接收到的 token 总数（dispatch 阶段各 rank 原子累加）
    //   - 高 32 位：原子累加的次数（达到 kNumSMs * kNumRanks 表示所有 SM 的 dispatch 都已完成）
    //
    // 存储布局：warp 级分片
    //   - 每个 lane 负责 expert (i*32 + lane_idx) 的计数
    //   - 例如 48 个 expert / 32 lanes = 每 lane 负责 2 个 expert (kNumExpertsPerLane=2)
    //     lane 0: expert 0, 32;  lane 1: expert 1, 33;  ...  lane 15: expert 15, 47
    //
    // 同步机制：自旋等待高 32 位 == kNumSMs * kNumRanks
    //   - kNumSMs 个 SM 各自做一轮 dispatch 写入，每个 SM 写入时高 32 位 +1
    //   - kNumRanks 个 rank 各自做一轮，每个 rank 的写入也 +1
    //   - 当计数达到 kNumSMs * kNumRanks 时，说明所有 rank 的所有 SM 都已完成写入
    // ========================================================================
    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do { 
                    // volatile 读：确保每次循环都从内存读取最新值，不被缓存
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                // 自旋等待：高 32 位 == kNumSMs * kNumRanks 表示所有 dispatch 都已完成
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            // 取低 32 位 = 该 expert 接收到的 token 总数
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();
    }

    template <typename Func>
    CUTLASS_DEVICE void for_each_block(Func&& func) {
        // Wait for all expert counters to be finalized
        fetch_expert_recv_count();

        // Initialize current expert with 0
        set_expert_idx(0);

        // Iterate over all blocks
        // TODO: add swizzle within expert waves for better L2 cache utilization
        while (true) {
            CUTE_TIE_DECL(get_next_block(), block_phase, current_local_expert_idx, m_block_idx, n_block_idx);
            if (block_phase == BlockPhase::None)
                break;

            func(block_phase, current_local_expert_idx,
                 block_phase == BlockPhase::Linear2 ? kNumL2BlockKs : kNumL1BlockKs,
                 m_block_idx, n_block_idx);
        }
    }
};

} // namespace deep_gemm::sched
