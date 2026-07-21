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
