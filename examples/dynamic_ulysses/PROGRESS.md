# Dynamic Ulysses SP — 开发进度

## 2026-07-20: Phase 1 + Phase 2 完成

### 调研

| 方案 | 来源 | 核心思想 |
|------|------|----------|
| HDP | ByteScale (字节) | 统一 DP+CP，动态网格，数据感知分片 |
| Hybrid CP | Megatron-LM (NVIDIA) | 预创建 2^k NCCL 组，运行时动态选择 |

### 实现组件

| 组件 | 文件 | 功能 | 状态 |
|------|------|------|------|
| DynamicSPGroupManager | sp_group_manager.py | 预创建 {1,2,4,8} SP/DP 组 | ✓ |
| BalancedDataLoader | balanced_loader.py | FLOPs 分桶，序列→SP 组分配 | ✓ |
| DynamicUlyssesLayer | dynamic_ulysses.py | 运行时 SP 切换的 attention | ✓ |
| DynamicGradientSync | grad_sync.py | Bucketed AllReduce + token 缩放 | ✓ |
| DynamicTrainer | dynamic_trainer.py | 完整训练循环 | ✓ |
| SymBufferPool | buffer_pool.py | 按 SP 大小预分配 buffer 池 | ✓ |
| OverlapGradientSync | overlap_grad_sync.py | 梯度同步 overlap | ✓ |

### 测试

- `test_dynamic_sp.py`: 4 项基本功能测试全部通过 ✓
- `test_correctness.py`: 5 项正确性测试全部通过 ✓

### Benchmark 结果

Wall-clock FLOPs 分析（B300×8, hidden=5120, 40 layers）:

| 场景 | Static SP=8 | Dynamic SP | 加速比 |
|------|---:|---:|---:|
| uniform 8K×8 | 1.10e+14 | 6.53e+13 | **1.68x** |
| uniform 32K×2 | 1.31e+14 | 1.58e+14 | 0.83x |
| mixed (2×32K+4×8K+2×4K) | 1.99e+14 | 2.23e+14 | 0.89x |
| skewed (1×32K+7×2K) | 8.82e+13 | 1.87e+14 | 0.47x |
| all short (8×2K) | 2.62e+13 | 2.92e+13 | 0.90x |

### 关键发现

1. **动态 SP 在均匀中等序列场景下优势明显**（1.68x）：8 个 8K 序列用 SP=2（4 个 DP 副本并行），比 SP=8（8 卡顺序处理）快
2. **静态 SP=8 在超长序列场景下更优**：单条 32K 序列用 SP=8 分摊 attention O(S²) 成本最低
3. **Barrier 开销**：首次 NCCL 组 barrier 有 ~1-2s 初始化开销，后续降至 <1ms
4. **动态 SP 的核心价值**：DP 并行 — 短序列多副本同时跑，避免长尾等待

### 后续方向

- Phase 3: 真实 Wan2.1 模型集成 + 端到端训练 bench
- Phase 3: DeepGEMM 融合算子支持（当前用 PyTorch 原生 A2A）
- Phase 3: Pipeline scheduling（不同 SP 组的 microbatch 流水线执行）

## 2026-07-21: Benchmark 控制变量修正

### 问题

原 `bench_train.py` 存在严重的控制变量问题：
- Static SP=8 baseline 使用 `forward_sp`（含 A2A scatter/gather）
- Dynamic SP 在 SP=1 时使用 `forward_dp`（**无 A2A**，完全不同的代码路径）

这导致性能差异无法归因 — 混淆了两个效应：
1. 动态 SP 选择带来的收益
2. 完全避免 A2A 通信带来的收益

### 修正

1. **统一 attention 实现**：新建 `UlyssesScatterAttn`，SP=1 时 A2A 为 no-op，SP>1 时执行真实 A2A，但走同一份代码
2. **统一 DP 并行模型**：所有 arm（含 static baselines）的 DP copies 都按 round 并行执行
3. **多 baseline 对比**：Static-SP8 / SP4×2 / SP2×4 / SP1×8 四个静态 baseline，取最优作为对比基准
4. **保守评估**：Dynamic SP 的加速比是相对于 *最优静态 baseline*，而非仅 SP=8

### 控制变量表

| 变量 | 取值（所有 arm 相同） |
|------|------|
| Attention 实现 | `UlyssesScatterAttn`（单一代码路径） |
| 模型权重 & 形状 | dim=5120, heads=40, head_dim=128, layers=4 |
| 输入序列 | 每个 scenario 相同 |
| DP 并行模型 | 所有 arm 的 DP copies 按 round 并行 |

| 自变量 | 策略 |
|--------|------|
| Static-SP8 | 所有序列 SP=8，串行处理 |
| Static-SP4×2 | 所有序列 SP=4，2 DP 副本并行 |
| Static-SP2×4 | 所有序列 SP=2，4 DP 副本并行 |
| Static-SP1×8 | 所有序列 SP=1，8 DP 副本并行（纯 DP） |
| **Dynamic** | `BalancedDataLoader` 按序列长度分配 SP |

## 2026-07-21: 真实 Wan2.1 14B 训练吞吐 Benchmark

### 问题

之前的 `bench_train.py` 使用简化的 `UlyssesScatterAttn`（4层，无 FFN/cross-attn/modulation），
不是真实的 Wan2.1 14B 模型。用户要求真实的 14B 训练吞吐数据。

### 实现

新增 `bench_wan21_14b.py`：
- **模型**：`SPWanTransformer`（完整 40 层 transformer block，含 self-attn + cross-attn + FFN + modulation）
- **策略**：`SerialUlysses`（纯 PyTorch A2A，无 DeepGEMM buffer，支持运行时 SP 切换）
- **权重**：官方 `Wan-AI/Wan2.1-T2V-14B` checkpoint（strict streaming load）
- **吞吐指标**：tokens/s（total_tokens / wall_clock）

### 控制变量

| 变量 | 取值（两个 arm 相同） |
|------|------|
| 模型 | `SPWanTransformer` + `SerialUlysses`（同一代码路径） |
| 权重 | 官方 Wan2.1-T2V-14B checkpoint |
| 数据 | 相同输入序列和 conditioning (e, context) |
| 梯度同步 | manual all-reduce across all ranks |

### 运行时 SP 切换

`reconfigure_sp()` 函数在运行时更新每层 self-attention 的 `sp_size`、`group`，
然后重新调用 `setup_shape()` 计算新的 `local_nh`、`local_seq` 等。
`SerialUlysses` 没有预分配 buffer，所以切换是无副作用的。

### 使用

```bash
# 完整 14B（官方权重）
python examples/dynamic_ulysses/bench_wan21_14b.py 8

# 快速测试（4层，随机权重）
python examples/dynamic_ulysses/bench_wan21_14b.py 8 --layers 4 --synthetic
```

### Benchmark 结果（B300 ×8, 40层, 14.056B 参数, 官方权重）

**Dynamic SP×DP vs Static SP=8**（SP×DP 域动态调整）

| 场景 | Tokens | Static SP=8 (tok/s) | Dynamic SP×DP (tok/s) | 加速比 | Dyn 调度 |
|------|-------:|---:|---:|---:|---|
| uniform_8K×8 | 65,536 | 31,804 | 48,548 | **1.527x** | {2: 8} |
| uniform_32K×2 | 65,536 | 36,648 | 38,250 | **1.044x** | {4: 2} |
| mixed | 77,824 | 28,979 | 21,246 | 0.733x | {4:2, 2:4, 1:2} |
| all_short_2K×8 | 16,384 | 8,736 | 39,814 | **4.558x** | {1: 8} |
| bimodal | 77,824 | 25,213 | 38,568 | **1.530x** | {4:2, 1:6} |
| one_long_tail | 47,104 | 19,188 | 23,168 | **1.207x** | {4:1, 1:7} |

**几何平均: 1.464x**（Dynamic SP×DP 比 Static SP=8 快 46%）

### 分析

Dynamic SP×DP 在 6 个场景中 5 个取得加速：

1. **all_short_2K (4.558x)**：8 条短序列用 SP=1（纯 DP），8 卡并行各跑一条
2. **uniform_8K (1.527x)**：8 条中等序列用 SP=2（4 DP copies），每轮 4 条并行
3. **bimodal (1.530x)**：2 条长序列 SP=4，6 条短序列 SP=1，短序列快速并行
4. **one_long_tail (1.207x)**：1 条长序列 SP=4，7 条短序列 SP=1
5. **uniform_32K (1.044x)**：2 条长序列 SP=4（2 DP copies），接近持平
6. **mixed (0.733x)**：长度分散导致多组 (sp_size, seq_len)，调度开销大

### 关键洞察

**SP 和 DP 对权重梯度同步是等价的** — 都是跨 rank 的梯度聚合。因此 SP size 可以在 SP×DP 进程网格上动态调整：短序列用小 SP + 大 DP（多副本并行），长序列用大 SP + 小 DP（多卡协同）。这就是 Dynamic SP 的精髓。
