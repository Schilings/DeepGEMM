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
// kSeqMajor: input is seq-major [bs, seq, local_nheads, head_dim] (BSHD, FlashAttention's native
// output layout) instead of head-major [bs, local_nheads, seq, head_dim] (BHSD). With BSHD the
// per-token [local_nheads, head_dim] = local_hidden slice is CONTIGUOUS in the source (and lands
// in a contiguous local_hidden column block of the gathered output), so the seq<->head transpose
// degenerates into a contiguous block copy — no scattered strided reads. (BHSD strides the source
// by seq*head_dim across heads.) Lets us drop the redundant transpose when attention emits BSHD.

// ════════════════════════════════════════════════════════════════════════════════════════════
// 模板参数 (编译期常量, 由 JIT 的 generate_impl() 实例化)
// ════════════════════════════════════════════════════════════════════════════════════════════
//   kNumRanks   : SP 组内的 GPU 数 (e.g. 8). 用于计算 local_nheads = nheads/R, local_seq = seq/R
//   kTileM      : M-tile 粒度. 必须等于下游 GEMM 的 BLOCK_M, 这样 barrier[m_tile] 的索引
//                 与 GEMM 的 m_block 完全对齐, GEMM 消费者直接按 m_block_idx 等 barrier
//   kSetBarrier : true  = M1 fused 模式. 每写完一个 (dst, tile) 后写 per-M-tile barrier,
//                          供 GEMM 消费者逐 tile 等待所有 rank 到齐再开始该 tile 的 MMA
//                 false = M0 独立模式. 纯 comm 不写 barrier, 由上层 Python barrier 同步
//   kSeqMajor  : true  = 输入为 BSHD [bs, seq, local_nheads, head_dim] (FlashAttention 原生),
//                         此时 per-token 的 [local_nheads, hd] 在内存中连续, seq↔head
//                         转置退化为连续块拷贝, 无跨 stride 读取
//                 false = 输入为 BHSD [bs, local_nheads, seq, head_dim] (默认)
// ════════════════════════════════════════════════════════════════════════════════════════════
// 运行时参数 (由 launch_impl 传入)
// ════════════════════════════════════════════════════════════════════════════════════════════
//   sym_buffer  : SymBuffer<kNumRanks> — 对称内存句柄. 每个 rank 持有所有 peer 的 buffer 基址.
//                 核心方法: map(local_ptr, dst_rank) 将本地地址转换为 dst 的对应 P2P 地址
//   bs, nheads, seq, head_dim : 输入张量 x 的全局 shape (跨所有 rank 之和)
template <uint32_t kNumRanks, uint32_t kTileM, bool kSetBarrier, bool kSeqMajor = false>
__global__ void sm100_a2a_transpose_comm_impl(
        const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,
        const uint32_t bs,
        const uint32_t nheads,
        const uint32_t seq,
        const uint32_t head_dim) {

    // ── Step 1: 计算本 rank 的 local 维度 ──
    const uint32_t rank = sym_buffer.rank_idx;
    const uint32_t local_nheads = nheads / kNumRanks;   // 本 rank 拥有的 head 数
    const uint32_t local_seq = seq / kNumRanks;          // 本 rank 负责的 seq 长度 (post-attn 分片)
    const uint32_t hidden = nheads * head_dim;           // 完整的 hidden 维度 = 所有 rank 的 head 拼起来

    // uint4 向量化: 每个 uint4 = 8 个 bf16 = 16 字节, 一次访存 16B
    // vec_hd    = head_dim 方向有多少个 uint4  (e.g. hd=128 → 16)
    // vec_hidden = hidden 方向有多少个 uint4   (e.g. hd=128, nheads=32, R=8 → 4*16=64)
    constexpr uint32_t kPack = 8;
    const uint32_t vec_hd = head_dim / kPack;
    const uint32_t vec_hidden = hidden / kPack;

    // ── Step 2: 获取 buffer 布局指针 ──
    // ws (workspace) 封装了 sym buffer 的 3 个区域:
    //   input   : 本 rank 的输入 x [bs, local_nheads, seq, head_dim] (只读)
    //   gathered: 整个 SP 组收集后的大矩阵 [bs, local_seq, hidden] (本 rank 只写属于自己的列段)
    //             列偏移: rank * local_hidden → rank * (local_nheads * head_dim)
    //   barrier : per-M-tile 的 int32 数组 [bs * tiles_per_seq], 供 M1 fused 模式同步
    const layout::BF16A2ATransposeGemmWorkspace ws(
        sym_buffer.template get_base_ptr<void*>(), kNumRanks, bs, nheads, seq, head_dim);

    const uint4* in_vec = reinterpret_cast<const uint4*>(ws.template get_input_ptr<void>());
    uint4* gathered_local = reinterpret_cast<uint4*>(ws.template get_gathered_ptr<void>());
    int32_t* barrier_local = ws.get_barrier_ptr();

    // ── Step 3: 计算工作量 (总 CTA 数) ──
    // 每个 dst_rank 有 bs * tiles_per_seq 个 M-tile
    // 总 work = R 个 dst × bs × ceil(local_seq/kTileM) 个 tile
    const uint32_t tiles_per_seq = (local_seq + kTileM - 1) / kTileM;
    const uint32_t tiles_per_dst  = bs * tiles_per_seq;
    const uint32_t total_work      = kNumRanks * tiles_per_dst;

    // M1 fused 模式: CTA 0 的 thread 0 置 a2a_signal=1 (通知 GEMM stream comm 已启动)
    // GEMM stream 通过 cuStreamWaitValue 等一个非零值才能 launch
    if (kSetBarrier and blockIdx.x == 0 and threadIdx.x == 0) {
        asm volatile("st.relaxed.gpu.global.b32 [%0], 1;" : : "l"(ws.get_a2a_signal_ptr()));
    }

    // ═══════════════════════════════════════════════════════════════════════════════════════
    // Step 4: 主循环 — 遍历所有 (dst_rank, m_tile) 工作项
    // ═══════════════════════════════════════════════════════════════════════════════════════
    //
    // 平面 work 编号 = step * tiles_per_dst + tile
    //   step ∈ [0, R)   → 目标 dst_rank (可选旋转)
    //   tile ∈ [0, tiles_per_dst) → [b, t] 分解出 batch 和 M-tile
    //
    // grid-stride loop: 每个 CTA 处理 blockIdx.x, blockIdx.x+gridDim.x, ... 的工作项
    for (uint32_t work = blockIdx.x; work < total_work; work += gridDim.x) {

        // ── 4a. 决定目标 dst_rank ──
        //
        // 旋转 (rotation, 仅 M0):
        //   dst_rank = (rank + step) % R
        //   不旋转时所有 rank 在 step=0 同时打 dst 0 → 只有 1 条 NVLink ingress 在工作
        //   旋转后每个 step 是一个置换: rank 0→dst 0, rank 1→dst 1, ..., rank R-1→dst R-1
        //   → R 条 ingress 全忙, comm 带宽 +12%
        //
        // 不旋转 (M1 fused):
        //   保持全 rank 同时打同一个 dst → 该 dst 的 tile 集中到齐 → GEMM 可以尽早开始 overlap
        //   旋转会分散贡献时间, 让 GEMM 的 overlap 变差, 所以 M1 放弃旋转
        const uint32_t step     = work / tiles_per_dst;
        const uint32_t dst_rank = kSetBarrier ? step : ((rank + step) % kNumRanks);

        // ── 4b. 分解 tile → (batch, M-tile 起始行) ──
        const uint32_t tile = work % tiles_per_dst;
        const uint32_t b    = tile / tiles_per_seq;          // batch index [0, bs)
        const uint32_t t    = tile % tiles_per_seq;          // M-tile index [0, tiles_per_seq)
        const uint32_t s0   = t * kTileM;                    // 起始行号 (在 dst 的 local_seq 内)
        const uint32_t s1   = (s0 + kTileM < local_seq) ? (s0 + kTileM) : local_seq;
        const uint32_t tile_rows = s1 - s0;                  // 有效行数 (处理最后一 tile 的尾巴)

        // 总拷贝量 (uint4 粒度):
        //   这个 tile 有多少行 × 每行有多少 local_nheads 个 head × 每个 head 有多少 uint4
        const uint32_t nelems = tile_rows * local_nheads * vec_hd;

        // ═══════════════════════════════════════════════════════════════════════════════════
        // 4c. 数据搬运循环: 逐 uint4 拷贝, 完成 seq↔head 转置
        // ═══════════════════════════════════════════════════════════════════════════════════
        //
        // 关键: 转置在"读地址"和"写地址"的计算中完成:
        //
        //   输入 x[b, nh, global_seq, hd]  → head 是外层, seq 随 token 变
        //   输出 gathered[b, s_local, rank*local_nheads+nh, hd]
        //        → seq 是外层, head 按 rank 排 (rank 0 的 head 在列 0..local-1,
        //          rank 1 的 head 在列 local..2*local-1, ...)
        //
        // 一个线程在一次迭代中拷贝 1 个 uint4 (8 bf16 元素, 即 head_dim 方向的 8 个连续元素)
        //
        // 具体索引推导:
        //   i = 线程在一次 tile 内的元素偏移 (uint4 粒度)
        //   hd_v = i % vec_hd           → head_dim 方向的 uint4 编号
        //   r1   = i / vec_hd           → 剩余维度编号
        //   nh   = r1 % local_nheads    → 哪个 head
        //   s    = r1 / local_nheads    → 哪一行 (0..tile_rows-1)
        for (uint32_t i = threadIdx.x; i < nelems; i += blockDim.x) {
            const uint32_t hd_v = i % vec_hd;
            const uint32_t r1   = i / vec_hd;
            const uint32_t nh   = r1 % local_nheads;
            const uint32_t s    = r1 / local_nheads;
            const uint32_t s_local    = s0 + s;                // 目标行在 local_seq 内的偏移
            const uint32_t global_seq = dst_rank * local_seq + s_local;  // 全局 seq 编号

            // 源地址 (读取): 根据 kSeqMajor 选择 BHSD 或 BSHD 索引公式
            //
            // BHSD: x[bs, local_nheads, seq, head_dim]
            //   offset = b*(lh*s*hd) + nh*(s*hd) + global_seq*hd + hd_v*8
            //   注意 stride 在 head 间是 seq*head_dim (很大, 跨 head 读不是连续的)
            //
            // BSHD: x[bs, seq, local_nheads, head_dim] (FlashAttention 原生)
            //   offset = b*(s*lh*hd) + global_seq*(lh*hd) + nh*hd + hd_v*8
            //   per-token 的 [nh, hd] 空间连续 → 转置退化为一整块连续搬运
            const uint64_t in_off = kSeqMajor
                ? ((static_cast<uint64_t>(b) * seq + global_seq) * local_nheads + nh) * vec_hd + hd_v
                : (static_cast<uint64_t>(b) * local_nheads + nh) * seq * vec_hd +
                      static_cast<uint64_t>(global_seq) * vec_hd + hd_v;

            // 目标地址 (写入): gathered[bs, local_seq, hidden]
            //   offset = b*(ls*hidden) + s_local*(hidden) + rank*local_hidden + nh*hd + hd_v*8
            //
            // 关键: rank*local_hidden 列偏移: 每个 rank 在 gathered 中占 hidden 的一段连续列
            //   列 [0,         local_hidden) → rank 0 的 head 切片
            //   列 [local_hidden,   2*local_hidden) → rank 1 的 head 切片
            //   ...
            //   列 [rank*local_hidden, (rank+1)*local_hidden) → 本 rank 自己的 head 切片
            // 所以 gathered[s_local, rank*local_nheads+nh, hd] 填的就是本 rank 的数据
            const uint64_t out_off = (static_cast<uint64_t>(b) * local_seq + s_local) * vec_hidden +
                                     static_cast<uint64_t>(rank * local_nheads + nh) * vec_hd + hd_v;

            // P2P 写入: sym_buffer.map(ptr, dst) 将本地地址转为 dst 的物理地址 (NVLink)
            uint4* dst_ptr = sym_buffer.map(gathered_local + out_off, dst_rank);
            *dst_ptr = in_vec[in_off];
        }

        // ── 4d. Barrier (仅 M1 fused 模式) ──
        //
        // 写完一个 tile 后, 所有线程 sync, 然后 thread 0 做:
        //   1. fence.acq_rel.sys — 保证本次 P2P 写入对所有 GPU 可见
        //   2. atomicAdd_system(barrier[tile], -1) — 跨 GPU 原子减 1
        //   3. 当 barrier 降到 -kNumRanks (即所有 R 个 rank 都对这个 tile 写了数据):
        //      st.release.sys barrier[tile]=1 — 通知 GEMM 消费者 "这个 tile 的 full-K 已就绪"
        //
        // 消费者 (sm100_bf16_a2a_transpose_gemm.cuh):
        //   while (barrier[m_block] != 1) { __nanosleep(...); }
        //   → fence.acquire.sys → 开始 TMA load A
        if constexpr (kSetBarrier) {
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t barrier_idx = b * tiles_per_seq + t;
                int32_t* bptr = sym_buffer.map(barrier_local + barrier_idx, dst_rank);
                asm volatile("fence.acq_rel.sys;\n");
                int32_t prev = atomicAdd_system(bptr, -1);
                // 计数归零 → 所有 rank 到齐
                if (prev - 1 == -static_cast<int32_t>(kNumRanks))
                    asm volatile("st.release.sys.b32 [%0], 1;\n" : : "l"(bptr));
            }
        }
    }
}

} // namespace deep_gemm

#pragma clang diagnostic pop
