# Dynamic Ulysses SP

Dynamic Sequence Parallelism framework for Wan2.1 14B training.

## 一句话总结

在 SP×DP 进程网格上**动态调整 SP/DP 比例**：短序列用小 SP + 大 DP（多副本并行），长序列用大 SP + 小 DP（多卡协同），在完整 Wan2.1 14B（40层，官方权重）上实现 **1.464x** 几何平均吞吐提升。

## 核心思想

> **SP all-reduce 和 DP all-reduce 对权重梯度是等价的** — 都是跨 rank 的梯度聚合。因此 SP size 可以在运行时动态调整。

静态 SP×DP（如固定 8SP×1DP）的问题：序列长度天然异构，短序列用大 SP 浪费 A2A 通信开销，长尾序列拖慢全局。

Dynamic SP 的解法：
- **长序列** → 大 SP（如 SP=8），8 卡协同分摊 O(S²) attention 成本
- **短序列** → 小 SP（如 SP=1），8 卡各跑一条（纯 DP 并行）
- **所有 rank** 统一梯度同步，不受 SP 组大小影响

## 实验结果

**环境**: B300 ×8, Wan2.1 T2V-14B (40层, 14.056B 参数, 官方权重), SerialUlysses

| 场景 | Tokens | Static SP=8 (tok/s) | Dynamic SP×DP (tok/s) | 加速比 |
|------|-------:|---:|---:|---:|
| all_short_2K×8 | 16,384 | 8,736 | 39,814 | **4.558x** |
| bimodal (2×32K+6×2K) | 77,824 | 25,213 | 38,568 | **1.530x** |
| uniform_8K×8 | 65,536 | 31,804 | 48,548 | **1.527x** |
| one_long_tail (1×32K+7×2K) | 47,104 | 19,188 | 23,168 | **1.207x** |
| uniform_32K×2 | 65,536 | 36,648 | 38,250 | **1.044x** |
| mixed (varied) | 77,824 | 28,979 | 21,246 | 0.733x |

**几何平均: 1.464x** (6 个场景中 5 个加速)

## 图表

实验数据自动可视化（`python examples/dynamic_ulysses/plot_bench.py`）：

| 图表 | 文件 | 内容 |
|------|------|------|
| 1 | `figures/fig1_throughput.png` | 吞吐量对比柱状图（6 场景 × 2 arm） |
| 2 | `figures/fig2_speedup.png` | 加速比柱状图 + SP 调度分布 |
| 3 | `figures/fig3_sp_assignment.png` | 序列长度 → SP size 分配阶梯图 |
| 4 | `figures/fig4_breakdown.png` | Dynamic 执行时间按 (SP, seq_len) 分组分解 |
| 5 | `figures/fig5_summary.png` | 吞吐+加速比双面板摘要图 |

### 结果解读

- **all_short_2K (4.6x)**: 短序列不值得 SP 分片，SP=1 纯 DP 让 8 卡各跑一条
- **bimodal/uniform_8K (~1.5x)**: 中等序列用 SP=2，4 个 DP 副本并行
- **uniform_32K (1.04x)**: 长序列本身适合大 SP，DP 并行空间有限
- **mixed (0.73x)**: 长度太分散，(sp_size, seq_len) 组合多，调度开销大

## 控制变量

两个 arm（Static vs Dynamic）**完全一致**的部分：

| 变量 | 取值 |
|------|------|
| 模型 | `SPWanTransformer` + `SerialUlysses`（同一代码路径） |
| 权重 | 官方 `Wan-AI/Wan2.1-T2V-14B` checkpoint (14.056B) |
| 输入数据 | 相同序列和 conditioning (e, context) |
| 梯度同步 | manual all-reduce across all ranks |
| 总 tokens | 每个 scenario 相同 |

**唯一自变量**: SP×DP 调度策略

## 文件说明

### 框架核心

| 文件 | 功能 |
|------|------|
| `sp_group_manager.py` | 预创建 {1,2,4,8} SP/DP NCCL 通信组 |
| `balanced_loader.py` | FLOPs-aware 序列→SP size 分配 |
| `dynamic_ulysses.py` | 运行时 SP 切换的 attention 层 |
| `grad_sync.py` | Bucketed 梯度同步 + token 缩放 |
| `dynamic_trainer.py` | 完整训练循环（microbatch 调度 + fwd/bwd + 梯度同步） |
| `buffer_pool.py` | 按 SP 大小预分配 UnifiedSymmBuffer |
| `overlap_grad_sync.py` | 梯度同步与 backward overlap |

### Benchmark

| 文件 | 用途 | 命令 |
|------|------|------|
| `bench_wan21_14b.py` | **主 benchmark**：真实 14B 训练吞吐 | `python bench_wan21_14b.py 8` |
| `bench_train.py` | 简化版（4层，无FFN），快速迭代用 | `python bench_train.py 8` |
| `bench_dynamic_sp.py` | 分析模型（FLOPs 估算，无需 GPU） | `python bench_dynamic_sp.py 8` |
| `plot_bench.py` | 图表生成（matplotlib，输出到 `figures/`） | `python plot_bench.py` |

### 测试

| 文件 | 内容 |
|------|------|
| `test_dynamic_sp.py` | 4 项基本功能测试 |
| `test_correctness.py` | 5 项正确性测试（跨组 AllReduce、A2A、梯度同步等） |

### 文档

| 文件 | 内容 |
|------|------|
| `DESIGN.md` | 完整设计方案（问题背景、方案调研、系统架构、实现计划） |
| `PROGRESS.md` | 开发进度日志 |

## 快速开始

```bash
# 完整 14B（官方权重，约 10 分钟）
python examples/dynamic_ulysses/bench_wan21_14b.py 8

# 快速测试（4层，随机权重，约 1 分钟）
python examples/dynamic_ulysses/bench_wan21_14b.py 8 --layers 4 --synthetic

# 指定 checkpoint 目录
python examples/dynamic_ulysses/bench_wan21_14b.py 8 --checkpoint-dir /path/to/wan2.1-14b
```

## 技术要点

### 运行时 SP 切换

`reconfigure_sp()` 在运行时更新每层 self-attention 的 `sp_size` 和 `group`，然后重新调用 `setup_shape()`。`SerialUlysses` 没有预分配 buffer，切换无副作用。

### A2A 安全约束

同一 SP group 内所有 rank 必须同时调用 `all_to_all_single`，且发送/接收数据大小一致。因此同一 (sp_size, seq_len) 组内的 DP copies 必须处理**相同长度**的序列。

### 梯度同步安全

所有 rank（无论 SP 大小）都参与最终的 `all_reduce`。Dummy forward+backward（用相同数据填充不足的 DP copy 位）确保所有 rank 都有梯度，避免 all-reduce 死锁。

## 研究背景

| 方案 | 来源 | 核心思想 |
|------|------|----------|
| HDP | ByteScale (字节) | 统一 DP+CP，动态网格，数据感知分片 |
| Hybrid CP | Megatron-LM (NVIDIA) | 预创建 2^k NCCL 组，运行时动态选择 |

本方案结合两者：Megatron 式预创建组 + ByteScale 式 FLOPs 调度 + Ulysses A2A。
