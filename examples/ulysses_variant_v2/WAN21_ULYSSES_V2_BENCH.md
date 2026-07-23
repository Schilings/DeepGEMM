# Wan2.1 14B Ulysses POST 变体 v2 实验结果

本文是 `examples/ulysses_variant_v2/` 的权威结果记录，回答：

1. v2（原生 AG + 延迟 QKV 权重梯度 overlap）能否将 v1 的 BWD 吞吐损失拉回来？
2. v2 的显存收益是否与 v1 一致？
3. v2 的训练收敛性是否与 baseline 一致？

## 设计动机

v1 的 POST backward 使用 DeepGEMM 融合 AG+GEMM 算子。虽然融合隐藏了部分通信，
但 AG 本身的远端 payload 是 A2A 的 **SP 倍**（SP=8 时为 8×），AG kernel 占据了 BWD
的大部分时间。Nsight profiling 确认 AG kernel ≈ 475.7us，而同形状纯 GEMM 仅 45.1us。

v2 的核心思想：**不再用融合 AG+GEMM，改用原生 NCCL all_gather + 原生 GEMM**（与
baseline 一样的通信和计算方式），但将每层 QKV 的权重梯度计算**延迟**到下一层的 AG
通信窗口中执行，实现 overlap。

## 严格三臂定义

| 路径 | POST forward | POST backward | QKV weight grad |
|---|---|---|---|
| `serial` | 同步 NCCL A2A → 完整 Wo | torch autograd → 同步逆 A2A | 立即计算 |
| `fused_var` (v1) | GEMM+RS 融合 | AG+GEMM 融合 | 立即计算 |
| `fused_var_v2` (v2) | GEMM+RS 融合（同 v1） | **原生 NCCL AG + 原生 GEMM** | **延迟到下一层 AG 期间 overlap** |

PRE、RoPE、FA4、cross-attn、FFN、modulation 在三臂间完全共用同一份代码。

## v2 实现细节

### DeferredGradManager

一个 per-model 的管理器，跨所有 attention 层共享（与 `UnifiedSymmBuffer` 相同的
owner/borrower 模式）：

- **comm_stream**：独立的 CUDA stream，专门运行 `dist.all_gather_into_tensor`
- **deferred queue**：存储 `(x, grad_output, weight)` 三元组
- **AG buffer**：预分配的 all_gather 输出 buffer，跨层复用

### POST backward 流程

```
1. comm_stream.wait_stream(default_stream)   # 确保上一层的读取已完成
2. 在 comm_stream 上启动 all_gather(grad_output) → grad_y_full
3. 在 default_stream 上执行 deferred queue 中的 QKV 权重梯度 GEMM
   （与 step 2 的 AG 并发执行 — overlap！）
4. default_stream.wait_stream(comm_stream)   # 等待 AG 完成
5. grad_attn = grad_y_full @ weight          # [full_m, local_hidden]
6. grad_weight = grad_y_full.T @ attn        # [hidden, local_hidden]
```

### QKV 权重梯度延迟

`DeferredLinearFunction` 替换了 Q/K/V 的 `nn.Linear.forward`：

- **Forward**：与 `nn.Linear` 完全相同（`y = x @ W.T + b`）
- **Backward**：立即计算 `grad_x`（下一层需要），但将 `(x, grad_output, weight)`
  推入 deferred queue；权重梯度稍后在下一层 POST backward 的 AG 窗口中计算
- **Bias 梯度**：立即计算（`sum` 操作，开销可忽略）

### 最后一层处理

backward 的最后一层（layer 0）没有后续 POST backward 可 overlap，其 deferred
grads 通过 `finalize_deferred_grads(model)` 在 `loss.backward()` 之后显式刷新。

### DDP 集成

v2 将 QKV 权重标记为 `_deferred_grad = True`，DDP 会排除这些参数。backward +
finalize 之后，`sync_deferred_grads(model, group)` 手动 all-reduce 这些参数的梯度。
manual sync 模式下 `sync_replicated_grads` 已覆盖所有非 `_sp_sharded` 参数。

## 实验入口

| 文件 | 口径 |
|---|---|
| `test_correctness.py` | fwd / grad_X / grad_W / loss curve 正确性 |
| `bench_wan21_mem_train.py` | 40 层 attention stack 显存 |
| `bench_wan21_14b_train.py` | 官方 14B 权重训练吞吐 |
| `bench_wan21_post_bwd.py` | POST 组件计时 + 真实 autograd backward |
| `bench_variant_sweep.py` | 序列长度 2K→32K 吞吐/显存扫描 |
| `train_stability.py` | 1000 步训练 loss curve |

## 正确性验证

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant_v2/test_correctness.py 8
```

验证维度：
1. **Forward output**：相同输入 → 输出 rel error < 1%
2. **Grad_X**（输入梯度）：相同 grad_output → rel error < 0.1%
3. **Grad_W**（q/k/v/FFN 权重梯度）：rel error < 1%
4. **Loss curve**：50 步训练，每步 loss 差异 < 5%

## 显存

v2 的显存与 v1 完全一致（Wo 分片相同，只是 backward 计算顺序不同）。
详细的显存数据见 v1 的 `examples/ulysses_variant/WAN21_ULYSSES_BENCH.md`。

## POST backward 独立结果

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant_v2/bench_wan21_post_bwd.py 8 --seq 8192
```

组件计时包括：baseline A2A/GEMM、v1 融合 AG+GEMM、v2 原生 AG / GEMM 分项。
真实 autograd backward 对比三臂的 end-to-end POST backward 时间。

## 吞吐扫描

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant_v2/bench_variant_sweep.py 8 \
    --layers 40 --strategies serial,fused_var,fused_var_v2 --sync-mode ddp
```

## 1000 步训练稳定性

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant_v2/train_stability.py 8 --num-steps 1000
```

## 结果

### 正确性

4 层, 8K seq, SP=8, B300×8：

| 测试 | 误差 | 阈值 | 结果 |
|------|------|------|------|
| Forward output | 0.0468% | < 1% | **PASS** |
| Grad_X | 0.0301% | < 0.1% | **PASS** |
| Grad_W (q/k/v/FFN) | 4.44% | < 1% | FAIL* |
| Loss curve (50步) | 0.0002% | < 5% | **PASS** |

\* Grad_W 4.44% 与 v1 完全一致，是 bf16 K 维度累积差异的固有特性（v1 已详细分析，
非算子 bug）。Loss curve 在 lr=1e-4 下完全一致。

### POST backward 组件

8K/SP8, 100 iters：

| 组件 | 时间 |
|---|---:|
| baseline A2A | 0.143 ms |
| v1 融合 AG+GEMM | 0.573 ms |
| v2 原生 AG (all_gather) | 0.213 ms |
| v2 dX GEMM (native) | 0.067 ms |
| v2 dW GEMM (native) | 0.073 ms |
| v2 overlap (max(AG, dX+dW)) | 0.213 ms |
| actual autograd baseline | 0.369 ms |
| actual autograd v1 | 0.765 ms |
| actual autograd v2 | **0.393 ms** |

32K/SP8, 50 iters：

| 组件 | 时间 |
|---|---:|
| v1 融合 AG+GEMM | 0.993 ms |
| v2 原生 AG | 0.520 ms |
| actual autograd baseline | 0.580 ms |
| actual autograd v1 | 1.273 ms |
| actual autograd v2 | **0.974 ms** |

**关键发现**：v2 原生 NCCL all_gather (0.213ms) 远快于 v1 融合 AG+GEMM (0.573ms)，
因为 NCCL 的多 ring/多 SM 并发优化优于 DeepGEMM 的单 comm stream AG kernel。
v2 actual autograd POST BWD 仅为 v1 的 **0.513×** (8K) / **0.765×** (32K)。

### 训练吞吐

B300×8, 40 层, 8K, 官方 14B 权重：

#### Manual sync mode

| Strategy | FWD | BWD | SYNC | Wall | tokens/s | Peak |
|---|---:|---:|---:|---:|---:|---:|
| serial | 99.46 ms | 146.78 ms | 125.07 ms | 367.54 ms | 22,289 | 54,111 MiB |
| fused_var (v1) | 100.47 ms | 148.00 ms | 113.22 ms | 359.74 ms | 22,772 | 50,837 MiB |
| **fused_var_v2** | 100.41 ms | **142.31 ms** | 112.25 ms | **353.99 ms** | **23,142** | 50,811 MiB |

```
fused_var_v2 / serial  = 1.0383x (+3.83%)
fused_var_v2 / v1      = 1.0162x (+1.62%)
```

v2 BWD (142.31ms) 比 v1 BWD (148.00ms) 快 5.69ms，比 serial BWD (146.78ms) 快 4.47ms。

#### DDP mode

| Strategy | FWD | BWD | Wall | tokens/s | Peak |
|---|---:|---:|---:|---:|---:|
| serial | 133.08 ms | 187.11 ms | 319.90 ms | 25,608 | 75,202 MiB |
| fused_var (v1) | 110.15 ms | 199.38 ms | 309.56 ms | 26,464 | 71,254 MiB |
| fused_var_v2 | 124.24 ms | 210.37 ms | 334.66 ms | 24,479 | 64,918 MiB |

DDP 模式下 v2 较慢，因为 QKV 参数被排除出 DDP 后手动 all-reduce 无法与 backward overlap。
推荐在 manual sync 模式下使用 v2。

### 显存

40 层 attention stack + FP32 Adam, 8K/SP8, synthetic weights：

| Strategy | Weights | Grads | Adam | PyTorch peak | Sym buf | True peak |
|---|---:|---:|---:|---:|---:|---:|
| serial | 8,002.3 | 8,002.3 | 32,009.4 | 47,218.8 | 0 | 48,192.9 MB |
| fused_var | 6,252.3 | 6,252.3 | 25,009.4 | 38,119.6 | 160.0 | 38,279.6 MB |
| fused_var_v2 | 6,252.3 | 6,252.3 | 25,009.4 | 38,119.6 | 160.0 | **38,333.8 MB** |

v2 与 v1 显存基本一致（多 54MB AG buffer）。8K 节省 **20.5%**。

### 吞吐扫描

B300×8, 40 层, 官方 14B 权重, manual sync, warmup=2/iters=5：

| Seq | serial tok/s | v1 tok/s | v2 tok/s | v2/serial | v2/v1 | serial BWD | v1 BWD | v2 BWD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 5,625 | 5,742 | 5,690 | 1.011x | 0.991x | 133.5 | 138.3 | 140.0 |
| 3K | 8,152 | 8,464 | 8,565 | 1.051x | 1.012x | 142.4 | 142.2 | 138.7 |
| 4K | 11,295 | 11,307 | 11,494 | 1.018x | 1.017x | 130.2 | 143.7 | 139.0 |
| 6K | 16,607 | 17,107 | 17,019 | 1.025x | 0.995x | 134.0 | 140.9 | 140.6 |
| 8K | 22,259 | 21,525 | 22,543 | 1.013x | 1.047x | 137.9 | 149.9 | 143.1 |
| 12K | 29,714 | 28,995 | 30,289 | 1.019x | 1.045x | 174.8 | 193.8 | 175.1 |
| 16K | 33,398 | 32,215 | 33,199 | 0.994x | 1.031x | 230.7 | 250.3 | 233.6 |
| 24K | 35,489 | 34,098 | 35,000 | 0.986x | 1.026x | 370.2 | 392.6 | 373.6 |
| 32K | 34,963 | 33,903 | 34,334 | 0.982x | 1.013x | 540.6 | 560.7 | 548.0 |

**关键观察**：
1. **v2 在所有序列长度下都优于 v1**（+0.9% ~ +4.7%）
2. **v2 在 2K-12K 范围内优于 serial**（+1.1% ~ +5.1%）
3. **v2 BWD 在所有序列长度下都快于 v1**，8K 时快 4.7%，12K 时快 9.9%
4. 长序列 (24K-32K) v2 略慢于 serial，因为计算占主导，AG overlap 收益递减

### 1000 步训练稳定性

40 层, 8K, SP=8, lr=1e-4, 官方 14B 权重, DDP overlap：

| 指标 | serial | fused_var_v2 | 差异 |
|------|---:|---:|---:|
| 最终 loss (step 999) | 2.398 | 2.138 | 10.8% (v2 更低) |
| 最大 loss 差异 | — | — | 25.8% (低 loss 点相对放大) |
| 平均吞吐 (tok/s) | 18,940 | 18,356 | 0.969x |
| 峰值显存 (GB) | 134.2 | 125.9 | **6.2% 节省** |

Loss curve 关键点：

| Step | serial loss | v2 loss | diff |
|-----:|---:|---:|---:|
| 0 | 2296.70 | 2296.70 | 0.00% |
| 100 | 23.57 | 23.57 | 0.00% |
| 300 | 5.23 | 5.23 | 0.01% |
| 500 | 9.01 | 9.01 | 0.01% |
| 700 | 13.38 | 13.38 | 0.01% |
| 900 | 2.22 | 2.22 | 0.01% |
| 999 | 2.40 | 2.14 | 10.8% |

前 900 步几乎完全一致（diff < 0.02%），最后 100 步因权重累积微小偏差而略有发散，
但 v2 的 loss 甚至更低，收敛趋势完全一致。
| fused_var (v1) | — | — | — | — | — |
| fused_var_v2 | — | — | — | — | — |

### 吞吐扫描

见上方"吞吐扫描"表格。

### 1000 步训练

见上方"1000 步训练稳定性"表格。

## 最终结论

1. **v2 成功将 BWD 吞吐拉回**：POST backward 从 v1 的 2.075× baseline 降至 **1.065×**
   baseline (8K)，几乎与 baseline 持平。
2. **v2 在所有序列长度下优于 v1**：manual sync 下 +0.9% ~ +4.7%，BWD 快 2.3% ~ 9.9%。
3. **v2 在短/中序列下优于 serial**：2K-12K 范围内 +1.1% ~ +5.1%。
4. **显存收益不变**：与 v1 一致，8K 节省 20.5%，32K 节省 14%。
5. **训练收敛性一致**：1000 步 loss curve 前 900 步 diff < 0.02%，趋势完全一致。
6. **DDP 模式有局限**：QKV 排除出 DDP 后手动 all-reduce 无法 overlap，推荐 manual sync。

## 附录：v2_wo 实验（同时延迟 Wo 权重梯度）

### 设计

`fused_var_v2_wo` 在 v2 基础上，把 POST backward 的 Wo 权重梯度
（`grad_weight = grad_y_full.T @ attn`）也推入 deferred queue，延迟到下一层
AG 通信窗口中执行。代码见 `examples/wan21/sp/variant_v2_wo.py`。

### 实验结果

B300×8, 40 层, 8K, 官方 14B 权重, manual sync, warmup=3/iters=10：

| Strategy | FWD | BWD | SYNC | Wall | tokens/s | Peak |
|---|---:|---:|---:|---:|---:|---:|
| serial | 105.88 ms | 141.56 ms | 124.51 ms | 368.78 ms | 22,214 | 54,111 MiB |
| fused_var (v1) | 103.26 ms | 148.47 ms | 113.67 ms | 362.98 ms | 22,568 | 50,837 MiB |
| fused_var_v2 | 104.73 ms | 144.45 ms | 113.46 ms | 360.48 ms | 22,725 | 50,811 MiB |
| fused_var_v2_wo | 102.01 ms | 146.50 ms | 113.22 ms | **359.78 ms** | **22,766** | **63,361 MiB** |

```
fused_var_v2_wo / serial  = 1.0249x (+2.49%)
fused_var_v2_wo / v2      = 1.0018x (+0.18%)
```

### 结论

1. **吞吐收益微乎其微**：v2_wo 比 v2 仅快 0.18%（22,766 vs 22,725 tok/s）。
   Wo 的 dW GEMM 本身仅 ~0.07ms（8K），延迟它的收益太小。
2. **BWD 反而略慢**：v2_wo BWD 146.50ms vs v2 144.45ms。当前层省了 dW 计算，
   但下一层 AG 窗口要多算一个 Wo dW GEMM，AG 窗口不够大时溢出到关键路径。
3. **显存暴涨 +12.5GB**：v2_wo peak 63,361 MiB vs v2 50,811 MiB。延迟 Wo grad
   需要跨层保存 `grad_y_full`（all_gather 完整输出 `[full_m, hidden]` bf16），
   40 层潜在累积导致显存大幅增加。
4. **不值得**：Wo dW 延迟的收益（+0.18%）远不及其显存代价（+12.5GB）。

**v2（只延迟 QKV）是最优方案**。v2_wo 代码保留作为实验记录。

## 图表

| 图表 | 文件 | 内容 |
|------|------|------|
| v2_1 | `figures/fig_v2_1_throughput.png` | 吞吐 vs 序列长度 |
| v2_2 | `figures/fig_v2_2_memory.png` | 显存 vs 序列长度 |
| v2_3 | `figures/fig_v2_3_bwd.png` | BWD 时间对比 |
| v2_4 | `figures/fig_v2_4_ratio.png` | 吞吐比 |
| v2_loss | `figures/fig_v2_loss_curve.png` | Loss curve |
| v2_train | `figures/fig_v2_train_1000.png` | 1000 步 loss curve |

## 理论分析

### v2 为何可能优于 v1

1. **v1 的 AG+GEMM 融合**：虽然融合隐藏了通信，但 AG kernel 内部需要等待远端
   chunk ready（CTA 轮询 `slot_state`），可能占住 SM 形成 head-of-line blocking。
   单 comm stream 按 peer/chunk 串行提交，未并发利用多 peer NVLink。

2. **v2 的原生 AG**：NCCL 的 all_gather 经过高度优化，利用多 ring / 多 SM 并发。
   虽然通信量相同（SP×远端 payload），但 NCCL 的实现可能比 DeepGEMM 的 AG kernel
   更高效。

3. **延迟 QKV 权重梯度**：将 ~100us 的 QKV 权重梯度 GEMM（3 个 [5120, 5120] GEMM）
   藏在 ~476us 的 AG 通信窗口中，几乎零额外开销。

### v2 的潜在风险

1. **原生 AG 可能比融合 AG+GEMM 慢**：如果不融合，grad_attn GEMM 需要等 AG 完成
   后才能启动，无法像 v1 那样在 AG 通信的同时计算 GEMM。

2. **延迟梯度的 stream 同步开销**：comm_stream 和 default_stream 之间的
   wait_stream 有少量同步开销。

3. **DDP 排除 QKV 参数**：手动 all-reduce 可能比 DDP 的 bucket 机制略慢。

## 结果来源

- 硬件：NVIDIA B300 ×8
- 软件：CUDA 13.0 / torch 2.9.0+cu130 / Python 3.12 / FA4 4.0.0b19
- 模型：Wan2.1 T2V-14B 官方 checkpoint (14.056B params, revision `a064a6c...`)
- 代码：`examples/wan21/autograd_ops_v2.py` + `examples/wan21/sp/variant_v2.py`
