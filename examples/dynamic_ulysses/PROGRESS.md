# Dynamic Ulysses SP — 开发进度

## 2026-07-20: Phase 1-3 完成

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

## 2026-07-21: Benchmark 迭代

### 迭代 1: 控制变量修正

原 `bench_train.py` 的 Static 和 Dynamic arm 用了不同代码路径（`forward_sp` vs `forward_dp`），混淆了"动态SP选择"和"避免A2A"两个效应。修正为统一 `UlyssesScatterAttn`，SP=1 时 A2A 为 no-op，走同一份代码。

### 迭代 2: 真实 14B 模型接入

新增 `bench_wan21_14b.py`，使用完整 `SPWanTransformer`（40层, 官方权重 14.056B）。
关键实现：`reconfigure_sp()` 运行时切换 SP size + group + setup_shape。

### 迭代 3: 串行消融实验（无 DP）

纯 SP size 消融（无 DP 并行）：Dynamic 0.577x，证明 SP size 单独调整无收益，小 SP = 浪费 GPU。

### 迭代 4: SP×DP 动态调整（最终版本）

加入 DP 并行：同一 (sp_size, seq_len) 的序列 DP copies 并行执行。
修复 3 个死锁 bug：
1. `dp_idx = rank // sp_size`（非 `rank % dp_size`）
2. 同一 SP group 所有 rank 必须参与 A2A
3. 所有 rank 必须有梯度（dummy forward+backward 填充）

**最终结果: 几何平均 1.464x**（详见 README.md）

### Bug 修复记录

| 问题 | 原因 | 修复 |
|------|------|------|
| A2A 死锁 | `dp_idx = rank % dp_size` 计算错误 | 改为 `rank // sp_size` |
| A2A 死锁 | 同 SP group 不同 DP copy 序列长度不同 | 按 (sp_size, seq_len) 分组 |
| 梯度同步死锁 | 部分 rank 无梯度（跳过 backward） | 所有 rank 都做 forward+backward |
| `randn` 参数错误 | 传了 tensor 而非 int | 用 `x_tensor.shape[0]` |
