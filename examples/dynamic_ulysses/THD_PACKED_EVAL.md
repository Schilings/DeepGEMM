# THD Packed 评估报告

## 1. 当前方案的问题

当前 benchmark 逐条序列 forward：

```
batch = [32768, 2048, 2048, 2048, 2048, 2048, 2048, 2048]
                                      one_long_tail 场景

Step 1: assign SP
  32768 → SP=4, 7×2048 → SP=1

Step 2: 按 (sp_size, seq_len) 分组
  Group A: (SP=4, 32768) ×1  → 1 round, 2 DP copies (1 real + 1 dummy)
  Group B: (SP=1, 2048)  ×7  → 1 round, 8 DP copies (7 real + 1 dummy)
```

问题：
1. **kernel launch 开销**：每条序列独立 forward/backward，40 层 × 2（fwd+bwd）= 80 次 kernel launch × 8 条 = 640 次
2. **dummy 浪费**：Group A 只有 1 条序列但 dp=2，rank 4-7 做 dummy forward+backward
3. **负载不均衡**：Group A 的 1 条 32K 序列计算量远大于 Group B 的 7 条 2K 序列
4. **显存浪费**：每条序列独立分配 activation buffer，没有复用

## 2. THD Packed 方案

### 核心思路

同一 SP size 的**多条序列打包**成 THD 格式，一次 forward：

```
Group B: (SP=1, 7×2048) → 打包成 [7×2048, dim] = [14336, dim]
  cu_seqlens = [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336]
  flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens)
  → 一次 forward 处理 7 条序列
```

### 对调度的影响

**当前调度**：按 `(sp_size, seq_len)` 分组 → 同长度的序列才在一组
**THD 调度**：按 `sp_size` 分组 → 不同长度但相同 SP 的序列打包在一起

这是**重大改进**：

```
当前（mixed 场景）: [32768, 16384, 8192, 8192, 4096, 4096, 2048, 2048]
  → 6 个 (sp_size, seq_len) 组 → 6 rounds → 大量 dummy

THD packed:
  SP=4: [32768, 16384] → 打包 [49152, dim], 1 round, dp=2
  SP=2: [8192, 8192, 4096, 4096] → 打包 [24576, dim], 1 round, dp=4
  SP=1: [2048, 2048] → 打包 [4096, dim], 1 round, dp=8
  → 3 rounds，几乎无 dummy
```

**mixed 场景从 6 rounds 降到 3 rounds，dummy 从大量降到接近零！**

### 对显存的影响

| 场景 | 当前（逐条） | THD packed | 节省 |
|------|---|---|---|
| activation buffer 数量 | N 条 × 40 层 | 1 个 × 40 层 | N 倍 |
| dummy activation | 有 | 无 | 100% |
| padding 浪费 | 无（逐条） | 无（varlen） | - |

THD packed 后，同一 SP 组的所有序列共享一个 activation buffer，显存峰值 = `max(total_tokens_in_group)` 而非 `sum(max_seq × dp_size)`。

### 对负载均衡的影响

**当前**：同一 (sp_size, seq_len) 组内所有 DP copy 处理相同长度 → 天然均衡
**THD packed**：不同 DP copy 可能打包不同长度的序列 → 需要均衡分配

解决方案：**按 FLOPs 加权分配**
```python
# SP=4, dp=2, sequences = [32768, 16384]
# DP copy 0: [32768]      → FLOPs = 32768²
# DP copy 1: [16384]      → FLOPs = 16384²
# 不均衡！→ 需要重新分配

# 更好的方案：按 FLOPs 排序，greedy 装箱
# DP copy 0: [32768]      → FLOPs = 32768² = 1.07B
# DP copy 1: [16384]      → FLOPs = 16384² = 268M
# → 还是不均衡（4:1）

# 如果有更多序列：
# seqs = [32768, 16384, 8192, 4096]
# DP copy 0: [32768, 4096]  → 32768² + 4096² = 1.09B
# DP copy 1: [16384, 8192]  → 16384² + 8192² = 335M
# → 3.3:1，好一些但仍然不均衡
```

**关键洞察**：THD packed 的负载均衡问题比逐条更复杂，因为 attention 是 O(S²) 非线性的。长序列的 FLOPs 远大于短序列。

### 对 Mixed 场景的改善预测

当前 mixed 场景是唯一减速的（0.885x），原因是 6 个 (sp_size, seq_len) 组太多 rounds。THD packed 后：

```
当前: 6 rounds × (forward + dummy + backward + barrier) = 大量开销
THD:  3 rounds × (packed forward + backward + barrier) = 开销减半
```

**预测：mixed 场景可能从 0.885x 翻转到 >1.0x**

## 3. 实现方案

### 3.1 BalancedDataLoader 改造

新增 `schedule_packed` 方法：

```python
def schedule_packed(self, sequence_lengths):
    """按 SP size 分组，同组序列打包成 THD 格式。

    Returns:
        List[PackedMicrobatch], 每个 microbatch 包含:
        - sp_size: SP 大小
        - seq_lens: [seq1, seq2, ...] 打包的序列长度列表
        - total_tokens: sum(seq_lens)
        - cu_seqlens: 累积偏移 [0, seq1, seq1+seq2, ...]
        - dp_copy: DP 副本索引
    """
    # 1. 每条序列分配 SP size
    # 2. 按 SP size 分桶
    # 3. 每个 SP 桶内按 FLOPs 贪心装箱到 dp_size 个 DP copy
    # 4. 返回 packed microbatches
```

### 3.2 贪心装箱算法

```python
def greedy_pack(seqs, dp_size):
    """按 FLOPs (S²) 贪心装箱，使各 DP copy 负载均衡。"""
    # 按 FLOPs 降序排列
    sorted_seqs = sorted(seqs, key=lambda s: s**2, reverse=True)
    # 贪心：每次把序列放到当前 FLOPs 最小的 DP copy
    bins = [[] for _ in range(dp_size)]
    bin_flops = [0] * dp_size
    for s in sorted_seqs:
        min_bin = bin_flops.index(min(bin_flops))
        bins[min_bin].append(s)
        bin_flops[min_bin] += s ** 2
    return bins
```

### 3.3 Attention 改造

`_attn_forward` 从 `flash_attn_func` 改为 `flash_attn_varlen_func`：

```python
# 当前（固定形状）:
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)  # q: [bs, seq, nh, hd]

# THD packed（变长）:
from flash_attn import flash_attn_varlen_func
out = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seq,
    max_seqlen_k=max_seq,
    causal=True,
)  # q: [total_tokens, nh, hd], 无 padding
```

### 3.4 A2A 改造

当前 A2A 假设所有序列等长（`all_to_all_single` 用固定 split_size）。THD packed 后需要：
- **方案 A**：按序列对齐 padding 到 `max_seq`，A2A 后裁剪 — 简单但有浪费
- **方案 B**：用 `all_to_all` + `all_to_all_single` 的 unequal split — 复杂但无浪费
- **方案 C**：SP=1 时无 A2A，SP>1 时按 `total_tokens / sp` 等分 A2A — **推荐**

方案 C 的关键：THD packed 后 `total_tokens` 是所有序列长度之和，按 `total_tokens / sp` 等分给各 rank。但这要求 **每个 SP rank 的 local_tokens 相同**，即 `total_tokens % sp == 0`。如果不整除，padding 到整除。

## 4. 影响评估总结

| 维度 | 当前（逐条） | THD Packed | 改善 |
|------|---|---|---|
| kernel launch | N 条 × 80 次 | 3 组 × 80 次 | **~20x 减少** |
| dummy 计算 | 有（不足 dp_size 时填充） | 无（打包后天然填满） | **100% 消除** |
| rounds 数 | = (sp_size, seq_len) 组数 | = SP size 种类数 | **mixed: 6→3** |
| 显存 | N 个独立 buffer | 1 个 packed buffer | **~N 倍节省** |
| 负载均衡 | 天然均衡（同长度） | 需 FLOPs 贪心装箱 | **新挑战** |
| mixed 场景 | 0.885x（退化） | 预测 >1.0x | **可能翻转** |
| A2A | 固定 split | total_tokens/sp 等分 | 需 padding 处理 |

## 5. 风险

1. **flash_attn_varlen 兼容性**：需确认 FA4 的 varlen 接口与当前 `WanSelfAttention` 的调用方式兼容
2. **A2A padding 开销**：如果 `total_tokens % sp != 0`，需要 padding，可能抵消部分收益
3. **贪心装箱不一定最优**：对于极端长尾分布（1 条 32K + 7 条 2K），装箱后仍可能不均衡
4. **SP 切分粒度**：THD packed 后 A2A 按 `total_tokens/sp` 等分，不再是按序列边界切分 — 可能导致同一条序列被切到两个 SP rank，需要 attention 正确处理

## 6. 建议

**值得做**。THD packed 对 mixed 场景的改善预测最大（从退化到加速），且能消除 dummy 开销。建议分两步：
1. 先改 `BalancedDataLoader.schedule_packed` + 贪心装箱
2. 再改 attention 层用 `flash_attn_varlen_func`
