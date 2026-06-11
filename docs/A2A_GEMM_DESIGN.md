# A2A+GEMM 融合算子设计 (Ulysses SP: All2All + Wo GEMM)

## 场景

Ulysses Sequence Parallelism: Attention 输出后做 All2All（head→sequence 重分布）+ Wo GEMM。

```
Attention → x[seq/tp, all_heads, dim]
         → All2All → x'[seq, heads/tp, dim]
         → Wo GEMM → y[seq, hidden]
```

## 核心设计：Ring-Push + Compute Overlap

### Rank 层次调度

对于 n 个 ranks，rank i：

**计算顺序**（GEMM pipeline）：`i, (i-1+n)%n, (i-2+n)%n, ..., (i+1)%n`
- 先算 self（无通信依赖，零等待）
- 再按逆序算远端数据（先到的先算）

**通信顺序**（Push warps）：`(i+1)%n, (i+2)%n, ..., (i+n-1)%n, i`
- 先 push 远端（越早到达越好）
- self copy 最后做（反正 GEMM 先算 self）

### 4 Ranks 示例 (rank 0)

**通信**:
```
Step 0: push local_x[1] → rank 1 的 slot[0]
Step 1: push local_x[2] → rank 2 的 slot[0]
Step 2: push local_x[3] → rank 3 的 slot[0]
Step 3: local_x[0] → 本地 slot[0] (self copy)
```

**计算**（src_rank 顺序: 0, 3, 2, 1）:
```
Round 0: src_rank=0 (self, 无等待)   → m_blocks (0,1), n_blocks (0..3)
Round 1: src_rank=3 (等 slot[3])     → m_blocks (6,7), n_blocks (0..3)
Round 2: src_rank=2 (等 slot[2])     → m_blocks (4,5), n_blocks (0..3)
Round 3: src_rank=1 (等 slot[1])     → m_blocks (2,3), n_blocks (0..3)
```

### 为什么这个顺序最优

通信和计算形成完美流水线对齐：
- Rank 3 第一个 push 给 rank 0 → rank 0 第二轮算 rank 3
- Rank 2 第二个 push 给 rank 0 → rank 0 第三轮算 rank 2
- Rank 1 最后 push 给 rank 0 → rank 0 最后算 rank 1

**先到的数据先被计算！**

## 5 类 Warp 架构

| Warp | 线程数 | 角色 | 调度依据 |
|------|--------|------|---------|
| W0-W3 | 128T | **Push** | 通信顺序，全 SM 协作 strided copy，atomic counter signal |
| W4 | 32T | **Load A** | 计算顺序，poll slot_state → TMA load A from slot |
| W5 | 32T | **Load B** | 计算顺序，无脑 TMA load B（不等 flag） |
| W6 | 32T | **MMA Issue** | UMMA tensor core |
| W7 | 32T | **Reserved** | TMEM allocator |
| W8-W11 | 128T | **Epilogue** | TMEM → smem → TMA 2D store to output |

总线程 = 128 + 128 + 128 = **384T = 12 warps**

### Load A vs Load B 分离

- Load A 依赖通信（poll slot_state[src_rank] >= kNumSMs）
- Load B 不依赖通信（权重矩阵始终可用）
- 分离后 B 可以提前加载，不被 A 的 polling 阻塞

### Push Warps 工作方式

- 按 chunk 粒度（整个 M_per_rank × K 矩阵块）
- 所有 SM 的 push warps 全局 strided 协作搬运
- 每搬完一个 chunk，每个 SM 的 thread 0 做 `red_add_rel(remote_flag, 1)`
- 当 flag == kNumSMs 时该 slot 完全 ready
- Push warps 完成后 idle，不影响 GEMM pipeline

### Flag 机制

- `slot_state[src_rank]`: uint32_t atomic counter
- Push: 每个 SM 写完一个 chunk 后 `red_add_rel(flag, 1)`
- Load A: `while (ld_acq_sys(slot_state[src_rank]) < kNumSMs)` 轮询
- Self-rank: push warps 做完 self-copy 后 flag 自然到 kNumSMs

## 时间线

```
Push:   [push→rank1][push→rank2][push→rank3][self]  ← 全程与 GEMM 并行
GEMM:   [self tiles ][rank3 tiles][rank2 tiles][rank1 tiles]
         ↑零等待      ↑rank3 先到   ↑rank2 第二   ↑rank1 最后
```

## multicast

暂时 multicast=1（单 CTA），后续可加 2-CTA cluster。
2-CTA 时 Load A 需要等 256 行（2×BLOCK_M）ready。
