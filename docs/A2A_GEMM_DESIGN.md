# All2All + GEMM 融合算子设计

## 目标场景

**Ulysses Sequence Parallelism** — Attention 计算后的 All2All + Wo GEMM：

```
Attention 输出: x[seq/tp, num_heads, head_dim]  (每 rank 有部分 sequence，所有 heads)
                    ↓
            All2All (head 维度 → sequence 维度重分布)
                    ↓
Wo GEMM 输入: x'[seq, num_heads/tp, head_dim]  (每 rank 有所有 sequence，部分 heads)
              = x'[seq, K]  where K = num_heads/tp * head_dim
                    ↓
            Wo GEMM: x'[seq, K] × Wo[K, N]
                    ↓
输出: y[seq, N]  (每 rank 独立，无需 reduce)
```

## 与 AG+GEMM 的对比

| | AG+GEMM | A2A+GEMM |
|--|---------|----------|
| 通信 pattern | rank i 广播 x[i] 给所有 rank | rank i 发 x[i→j] 给 rank j (各不同) |
| 数据到达 | per-rank: rank j 的整块 M 数据 | per-rank: rank j 发来的一块 M 数据 |
| A 矩阵组装 | slots[rank] = 同一数据的不同 M slice | slots[rank] = 不同 rank 发来的不同 M slice |
| B 矩阵 | 所有 rank 相同的 [N, K] | 所有 rank 相同的 Wo[K, N] |
| 输出 | 各 rank 取不同 M slice | 各 rank 得到完整 [seq, N] |
| GEMM 触发 | slot[j] ready → 算 M[j*M/tp : (j+1)*M/tp, :] | slot[j] ready → 算 M[j*seq/tp : (j+1)*seq/tp, :] |

**结论：实现几乎完全相同**，只是 Comm warps 的数据搬运逻辑不同（scatter vs broadcast）。

## 通信 Warps 设计

AG+GEMM 的 Comm warps 执行 Ring All-Gather：
- Step 0: local copy x → slot[rank_idx]
- Step k: 从 prev_rank 拷贝 slot[src] → next_rank 的 slot[src]

A2A+GEMM 的 Comm warps 执行 P2P All-to-All：
- 每个 rank 有 tp 份数据，第 j 份要发给 rank j
- 实现方式 1 (P2P Direct): rank i 直接 NVLink write 到 rank j 的 slot[i]
- 实现方式 2 (Ring): Ring All-to-All (类似 NCCL)

**推荐 P2P Direct**（NVLink Gen5 全连接，无需 ring relay）：
```
for dst_rank in range(num_ranks):
    if dst_rank == rank_idx:
        local copy: input[dst_rank] → slot[rank_idx]  (本地)
    else:
        NVLink write: input[dst_rank] → remote slot[rank_idx] on dst_rank
    set slot_state[rank_idx] = 1 on dst_rank
```

## 内存布局

```
Workspace per rank:
┌──────────────────────────────────────────┐
│ barrier_signals (32B)                     │
│ slot[0]: 来自 rank 0 的数据 [M/tp, K]     │
│ slot[1]: 来自 rank 1 的数据 [M/tp, K]     │
│ ...                                       │
│ slot[tp-1]: 来自 rank tp-1 的数据          │
│ slot_state[0..tp-1]: ready flags          │
└──────────────────────────────────────────┘

输入 (per rank):
  x[M, K]  where M = seq (total), K = num_heads * head_dim
  其中 x 被切成 tp 份 (每份 M/tp × K)，第 j 份要发给 rank j

输出:
  y[M, N]  (各 rank 独立)
```

## Tile 调度

与 AG+GEMM 完全相同：
- 固定遍历所有 (m_block, n_block) tiles
- TMA Load warp 在加载 tile 前检查 slot_state[src_rank]
- src_rank = m_block_idx / (M/tp / BLOCK_M)

## 实现计划

1. 基于 `sm100_bf16_ag_gemm.cuh` 复制创建 `sm100_bf16_a2a_gemm.cuh`
2. 修改 Comm warps: Ring All-Gather → P2P All-to-All scatter
3. 修改 Workspace 布局: 输入数据分 tp 份
4. 创建 JIT runtime `sm100_bf16_a2a_gemm.hpp`
5. 创建 Python API `deep_gemm/a2a_gemm/__init__.py`
6. 创建测试 `tests/test_a2a_gemm.py`
7. 创建 benchmark `benchmarks/bench_a2a_gemm.py`
