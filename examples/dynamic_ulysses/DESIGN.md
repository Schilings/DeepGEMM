# Dynamic Ulysses SP 设计方案

## 1. 问题背景

### 1.1 静态 SP×DP 的瓶颈

在 Wan2.1 14B 训练中，我们使用 SP×DP 并行（如 8 卡 = 4SP×2DP 或 8SP×1DP），采用 THD 输入格式。

**问题**：训练数据序列长度天然异构（文本、多模态场景）。静态 SP×DP 下：
- DP 组间各 SP 组计算 step 耗时不同（长序列的 SP 组慢，短序列的 SP 组快）
- DP 梯度同步需要所有 rank 参与 AllReduce/ReduceScatter
- **长尾现象**：所有 rank 等待最慢的 SP 组，GPU 利用率低

```
静态 4SP×2DP（8卡）:
  DP group 0: rank[0-3] 处理 seq=8192   ← 慢
  DP group 1: rank[4-7] 处理 seq=2048   ← 快，但必须等 DP group 0

  时间线:
  rank[4-7]: ████░░░░░░░░░░  (快，但空等)
  rank[0-3]: ████████████████  (慢，决定 wall-clock)
```

### 1.2 解决方案

| 方案 | 来源 | 核心思想 |
|------|------|----------|
| **HDP (Hybrid Data Parallelism)** | ByteScale (字节) | 统一 DP+CP，动态网格，每个序列用最少 ranks 处理 |
| **Hybrid Context Parallel** | Megatron-LM (NVIDIA) | 预创建 2 的幂次通信组，运行时动态选择 CP 组大小 |

## 2. 方案调研

### 2.1 ByteScale HDP

**核心设计**：
1. **动态网格**：打破静态 2D mesh，每个 step 按序列长度动态决定 SP 大小
2. **数据感知分片**：短序列不需跨设备分片（1 rank 即可），长序列才用多 rank
3. **Balance Scheduler**：按 FLOPs 分桶，短执行时间的 rank 分配更多序列
4. **全局通信组复用**：一个全局 NCCL 组 + P2P 通信，避免动态创建组的开销
5. **选择性卸载**：长序列的激活值卸载到 CPU 内存

**梯度等价性**：参数在所有 HDP ranks 间复制，令牌均匀分布，本地梯度是部分和，最终 AllReduce 聚合。损失按 token 数缩放。

**关键限制**：
- 需要 P2P 通信基础设施（NVLink/IB）
- Balance Scheduler 的成本模型需要 Profiler 预分析
- 选择性卸载需要 CPU 内存和 PCIe 带宽

### 2.2 Megatron Hybrid Context Parallel

**核心设计**：
1. **预创建 2 的幂次组**：`create_hybrid_dp_cp_groups` 为每个 2^k 大小创建 NCCL 组
   - 8 卡 → 预创建 size=2 和 size=4 的组
2. **运行时动态选择**：每个 microbatch 根据序列长度选择合适的 CP 组大小
3. **`HybridCPDataLoaderWrapper`**：数据加载器感知 CP 组大小，分配序列到合适的组
4. **`hybrid_context_parallel_forward_backward`**：专用调度，支持不同 microbatch 用不同 CP 组

**组创建算法**（Megatron `parallel_state.py`）：
```python
# 8 ranks [0,1,2,3,4,5,6,7]
# group_sizes = [2, 4]
# size=2: [0,1], [2,3], [4,5], [6,7]
# size=4: [0,1,2,3], [4,5,6,7]
# 每个 rank 记录自己所属的组: {2: group([2,3]), 4: group([0,1,2,3])}
```

**关键限制**：
- 只支持 2 的幂次组大小（避免 NCCL 组爆炸）
- 组大小上限 = DP×CP 总 rank 数的一半
- 需要偶数 rank 数

### 2.3 我们的方案：Dynamic Ulysses

结合两者优点，适配我们的 8 卡 B300 环境和 Wan2.1 训练场景：

| 特性 | ByteScale | Megatron HCP | **Dynamic Ulysses (ours)** |
|------|-----------|-------------|---------------------------|
| 组大小 | 任意 | 2 的幂 | **2 的幂**（8卡下: 2/4/8） |
| 通信组 | 全局 P2P | 预创建 NCCL | **预创建 NCCL** |
| SP 算子 | Ring-attn | Ring-attn | **Ulysses A2A**（DeepGEMM 融合） |
| 负载均衡 | Balance Scheduler | HybridCPDataLoader | **FLOPs 分桶 + 动态分配** |
| 梯度同步 | AllReduce | AllReduce | **Bucketed ReduceScatter** |
| 模型 | LLM (GPT/Llama) | GPT | **Wan2.1 14B (DiT)** |

## 3. 系统设计

### 3.1 整体架构

```
                    ┌─────────────────────┐
                    │   DynamicSPScheduler │
                    │  (per-step decision)  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  DataLoader (balanced)│
                    │  (FLOPs-aware packing)│
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
   ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
   │ SP group A │        │ SP group B │        │ SP group C │
   │ size=4     │        │ size=2     │        │ size=2     │
   │ seq=32K    │        │ seq=8K     │        │ seq=4K     │
   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Gradient Sync       │
                    │  (bucketed allreduce  │
                    │   across all ranks)   │
                    └─────────────────────┘
```

### 3.2 核心组件

#### 3.2.1 DynamicSPGroupManager

管理预创建的 SP 通信组，运行时按需选择。

```python
class DynamicSPGroupManager:
    """预创建 2 的幂次 SP 组，运行时动态选择。

    8 卡 B300:
      size=2: [0,1], [2,3], [4,5], [6,7]  (4 个 SP 组, 4 个 DP 副本)
      size=4: [0,1,2,3], [4,5,6,7]        (2 个 SP 组, 2 个 DP 副本)
      size=8: [0,1,2,3,4,5,6,7]           (1 个 SP 组, 1 个 DP 副本 = 纯 SP)
      size=1: 每个 rank 独立               (8 个 DP 副本 = 纯 DP, 无 SP)
    """
```

#### 3.2.2 BalancedDataLoader

按 FLOPs 分桶，将序列分配到不同大小的 SP 组。

```python
class BalancedDataLoader:
    """FLOPs-aware sequence packing。

    1. 按 token 数排序序列
    2. 计算每个序列的 attention FLOPs (O(S²))
    3. 分桶：长序列 → 大 SP 组，短序列 → 小 SP 组
    4. 确保所有 SP 组的 wall-clock FLOPs 近似相等
    """
```

#### 3.2.3 DynamicUlyssesLayer

运行时根据 SP 组大小调整 Ulysses 通信。

```python
class DynamicUlyssesLayer:
    """根据当前 step 的 SP 组大小动态调整。

    - SP=1: 纯 DP，无 A2A 通信
    - SP=2: 2 卡 Ulysses A2A
    - SP=4: 4 卡 Ulysses A2A
    - SP=8: 8 卡 Ulysses A2A

    使用 DeepGEMM 的 UnifiedSymmBuffer，按 SP 大小分配。
    """
```

#### 3.2.4 DynamicGradientSync

跨不同 SP 组大小的梯度同步。

```python
class DynamicGradientSync:
    """所有 rank 参与 AllReduce（或 ReduceScatter）。

    - SP=1 的 rank: Wo 完整 → 直接 AllReduce
    - SP=4 的 rank: Wo 复制 → AllReduce 后除以 DP 数
    - SP=8 的 rank: Wo 复制 → 同上

    梯度按 token 数缩放（非 sample 数）。
    """
```

### 3.3 通信模式

#### Forward
```
每个 SP 组独立执行:
  1. QKV projection (fused GEMM+Norm+A2A)
  2. FlashAttention (local heads, full seq)
  3. Wo projection (fused A2A+GEMM)
```

#### Backward
```
每个 SP 组独立执行:
  1. Wo backward (GEMM+A2A inverse)
  2. Attention backward
  3. QKV backward (A2A inverse + Norm backward + GEMM)
```

#### Gradient Sync
```
所有 rank (跨 SP 组) 参与:
  - Bucketed AllReduce / ReduceScatter
  - 按 token 数缩放梯度
  - Wo 梯度: SP 组内已复制 → AllReduce 后除以 DP 副本数
```

### 3.4 8 卡 B300 配置示例

```python
# Step 1: 长序列 (seq=32K)
# → SP=8, DP=1: 所有 8 卡组成 1 个 SP 组
# → A2A 通信在 8 卡间, 无 DP 梯度同步

# Step 2: 中等序列 (seq=8K × 2)
# → SP=4, DP=2: 2 个 SP 组 (各 4 卡)
# → A2A 在组内 4 卡间, DP AllReduce 跨 2 组

# Step 3: 短序列 (seq=2K × 4)
# → SP=2, DP=4: 4 个 SP 组 (各 2 卡)
# → A2A 在组内 2 卡间, DP AllReduce 跨 4 组

# Step 4: 超短序列 (seq=1K × 8)
# → SP=1, DP=8: 8 个独立 rank
# → 无 A2A, DP AllReduce 跨 8 rank
```

## 4. 实现计划

### Phase 1: 基础设施 (MVP) ✓ 已完成
- [x] `DynamicSPGroupManager`: 预创建 {1,2,4,8} SP/DP NCCL 通信组
- [x] `BalancedDataLoader`: FLOPs-aware 序列→SP 组分配
- [x] `DynamicUlyssesLayer`: 运行时 SP 切换（SP=1 纯 DP, SP>1 Ulysses A2A）
- [x] `DynamicGradientSync`: Bucketed AllReduce + token 数缩放
- [x] 基本功能测试（4 个测试全部通过）

### Phase 2: 验证 ✓ 已完成
- [x] `DynamicTrainer`: 完整训练循环（microbatch 调度 + forward/backward + 梯度同步）
- [x] 正确性测试（5 个测试：跨组 AllReduce、SP 组 AllToAll、梯度同步、调度、顺序执行）
- [x] 分析模型 benchmark：几何均值 +7.4% vs SP=8，+20% vs SP=4
- [x] 真实 GPU benchmark（B300×8）：uniform 8K 动态 SP 快 **2.5x**

### Phase 3: 优化 ✓ 已完成
- [x] `SymBufferPool`: UnifiedSymmBuffer 按 SP 大小预分配，懒加载
- [x] `OverlapGradientSync`: 非阻塞 AllReduce 与 backward overlap
- [x] 通信组缓存和复用（DynamicSPGroupManager 一次性预创建）

### 后续方向
- [ ] Pipeline 调度：不同 SP 大小的 microbatch 交错执行
- [ ] DeepGEMM 融合算子集成（替换 PyTorch 原生 A2A）
- [ ] 自适应 SP 分配（运行时 profiling 动态调整阈值）
- [ ] 多节点扩展

## 5. 约束

- **硬件**: 8 × B300 GPU (单机)
- **模型**: Wan2.1 14B (DiT, 40 Transformer blocks)
- **SP 算子**: DeepGEMM Ulysses (bf16_fused_qkv_norm_a2a_nt + bf16_a2a_transpose_gemm_nt)
- **输入格式**: THD (packed sequences)
- **组大小**: 2 的幂 (1, 2, 4, 8)
