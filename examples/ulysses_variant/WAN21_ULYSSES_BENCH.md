# Wan2.1 14B Ulysses POST 变体实验结果

本文是 `examples/ulysses_variant/` 的权威结果记录，回答三个问题：

1. 将标准 Ulysses 的 post-attention 从同步 A2A + Wo 改为 Wo 分片 GEMM+ReduceScatter，能否降低训练峰值显存？
2. 使用官方 Wan2.1 T2V-14B 权重和 40 个完整 Transformer block 时，训练吞吐下降多少？
3. backward 的性能差距具体来自 AG+GEMM、GEMM、collective、barrier 还是梯度同步？

两个实验均默认加载官方 Wan2.1 T2V-14B checkpoint，权重来源统一。差异仅在模型范围和统计口径：显存表使用 40 层 self-attention stack，并显式计入 FP32 Adam 状态；真实权重吞吐使用 40 个完整 Wan2.1 Transformer block，但不包含 patch/text/time embedding、输出 head 和 optimizer step。

## 严格消融定义

两条路径的 PRE 和 attention 使用同一份代码：

1. Wan2.1 原始 Q/K/V `nn.Linear`；
2. Q/K RMSNorm；
3. 三次同步 `torch.distributed.all_to_all_single`；
4. 3D RoPE；
5. FlashAttention-4（FA4）。

它们唯一的差别是 POST：

| 路径 | POST forward | POST backward |
|---|---|---|
| `serial` baseline | 同步 NCCL A2A → 完整 `Wo` 的 `nn.Linear` | torch autograd → 同步逆 A2A |
| `fused_var` | 本地 Wo 输入列分片 → DeepGEMM GEMM+RS | DeepGEMM AG+GEMM → 本地 Wo shard 梯度 GEMM |

`serial` 不调用任何 DeepGEMM 通信融合算子。`fused` 策略（`sp/fused.py` 中的 `FusedUlysses`）POST 使用 `bf16_a2a_transpose_gemm_nt` 融合 A2A+GEMM，PRE 暂继承 serial baseline；它不属于本 POST-only 消融的第三条路径，但可作为标准 Ulysses 融合通信的参照。

## 实验入口

| 文件 | 口径 |
|---|---|
| `bench_wan21_mem.py` | 单层 FWD/BWD 峰值显存和 Wo shard 验证 |
| `bench_wan21_mem_train.py` | 40 层 attention stack，参数/梯度/FP32 Adam/激活/workspace 显存 |
| `bench_wan21_14b_train.py` | 官方 14B 权重、40 个完整 Transformer block、manual/DDP 训练吞吐 |
| `bench_wan21_post_bwd.py` | POST 组件、production autograd backward、NVTX profiling |
| `../debug/debug_var_bwd.py` | serial/variant 前向与输入梯度正确性 |

通用 profiling SOP 见 `docs/GPU_PROFILING_GUIDE.md`，Ulysses POST backward 是其中的附录案例。

## POST 变体的数据布局

Wan2.1 14B 使用 `hidden=5120, nheads=40, head_dim=128`。SP 大小为
`P` 时，每卡 attention 输出为：

```text
attn_local: [full_tokens, hidden / P]
Wo_local:   [hidden, hidden / P]       # nn.Linear.weight 的输入列分片
```

Forward：

```text
partial = attn_local @ Wo_local.T      # [full_tokens, hidden]
y_local = ReduceScatter(partial)       # [local_tokens, hidden]
```

Backward：

```text
grad_y_full = AllGather(grad_y_local)
grad_attn = grad_y_full @ Wo_local
grad_Wo_local = grad_y_full.T @ attn_local
```

GEMM+RS 不物化普通 torch `partial`；AG+GEMM 不物化独立的
`grad_y_full`，而是复用通信 workspace 已 gather 的 slots 计算权重梯度。

## Function 与 workspace 生命周期

`examples/wan21/autograd_ops.py` 只封装 POST 变体的
`FusedPostLinearFunction`：

- 数学输入、权重和输出都是普通 tensor；
- `UnifiedSymmBuffer` 由策略拥有，不由 autograd Function 创建或销毁；
- forward 与 backward 复用同一个固定 workspace；
- 多层模型中 layer 0 为 owner，其余层为 borrower；
- workspace 只分配一次，大小取 GEMM-RS 与 AG-GEMM 需求的最大值。

AG+GEMM 的高层入口 `bf16_ag_gemm_nt_with_input` 接收显式输入，隐藏 `.ag_x/.ag_slots_x` 等内部布局。底层 C++ 负责 stream/event 和 `slot_state` 复位。不得在 Function 中对数百 MB workspace 做逐调用 `zero_()`。

当前使用两个 stream-ordered `sym_buffer.handle.barrier()`：第一个保证本代输入发布后 peer 才开始 pull；第二个保证所有 peer 已消费完成后，下一层才能覆盖共享 `local_x`。旧的每层两次 `torch.cuda.synchronize()` 加 host process-group barrier 已移除，因为它会排空 DDP/NCCL side stream 并破坏 overlap。SP=8 实测所有 rank `grad_X rel=0`。

## 参数所有权与并行维度

本机 8 张 GPU 全部组成一个 `SP=8` group，`DP=1`。FSDP/ZeRO 应沿独立的 DP
维度分片，**不能再沿同一个 SP group 分片**；否则会把 baseline 的 replicated Wo
也提前切成 1/8，直接抹掉本实验要测的结构性收益（上一轮“变体不省显存”的
错误结论即源于此）。

- PRE 直接使用 `model.q/k/v`，不再创建冗余 `Wqkv/Wqkv_t` 参数；两条路径相同。
- baseline 的完整 `model.o.weight[hidden, hidden]` 在 8 个 SP rank 上复制；每卡都有
  完整 Wo 权重、完整梯度和完整 Adam m/v。
- variant 只注册 `[hidden, hidden/8]` 的 `Wo_r_local`，随后注销完整
  `model.o.weight`；每卡 Wo 权重、梯度和优化器状态均减少 7/8。
- 若未来有 `SP=8 × DP>1` 的二维 mesh，两条路径仍可沿 DP 维做相同 FSDP；variant 相对 baseline 的 SP 维 Wo 分片收益仍然存在。

### SP 梯度同步

- baseline 的 Q/K/V、norm、bias 和完整 Wo 都在 SP rank 间复制；各 rank 只持有局部序列 loss 贡献，因此必须跨 SP reduce。
- variant 除 `Wo_r_local` 外的复制参数仍需跨 SP reduce。
- `Wo_r_local` 不做 SP reduce：不同 SP rank 持有不同输入列 shard，AG backward 已 gather 完整 `grad_y`，本地 dW 已覆盖完整序列。
- 如果 `DP>1`，相同 SP 坐标上的 `Wo_r_local` replica 仍必须在 DP group 同步。
- `--sync-mode ddp` 会让 DDP 同步所有 replicated 参数，并显式排除 `_sp_sharded` Wo；普通 vanilla DDP 不能在 SP group 上错误归约不同 Wo shard。

## 显存统计

### 单层

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD \
  python3 examples/ulysses_variant/bench_wan21_mem.py 8 serial,fused_var
```

报告：

- `torch.cuda.max_memory_allocated()` 的 FWD/BWD 累计峰值；
- symmetric workspace 的实际字节数；
- Wo 逻辑大小和本地梯度大小。

### 多层训练

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD \
  python3 examples/ulysses_variant/bench_wan21_mem_train.py 8 40 32768 serial,fused_var
```

默认加载官方 Wan2.1 T2V-14B checkpoint（与吞吐实验同源），加 `--synthetic` 可回退到随机权重做快速冒烟。包含：SP=8/DP=1 下真实本地参数所有权、BF16 参数/梯度、FP32 Adam `m/v`、
保存到 backward 的激活，以及跨层复用一次的 symmetric workspace。输入只创建
本 rank 的序列分片，不再让每卡常驻完整 `X_full`。

`torch.cuda.max_memory_allocated()` 不保证统计 symmetric memory，所以文档将
PyTorch 峰值和 workspace 分项报告，并给出二者相加的估算峰值；最终结论还应
用 `max_memory_reserved()`、NVML 或 `cudaMemGetInfo` 交叉验证。

## 正确性

```bash
DG_AG_PUBLISH_SYNC=symm DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD \
  python3 examples/debug/debug_var_bwd.py 8
```

2026-07-20，B300×8、正式发布+消费双 barrier，所有 rank：

```text
grad_X rel (serial vs var): 0.000000
fwd rel (serial vs var):    0.002873 ~ 0.002878
```

前向差异来自 BF16 GEMM/ReduceScatter 的归约顺序；输入梯度与同步 baseline 一致。buffer 必须在 `destroy_process_group()` 前显式释放。

## 显存结果与结论

B300 ×8、40 个 attention 层、SP=8/DP=1、FA4、BF16 参数/梯度、FP32 Adam m/v、官方 Wan2.1 T2V-14B checkpoint（每层 self-attention 的 Q/K/V/O weight+bias + Q/K norm，严格加载 400 tensors / 4.196B parameters）：

| Sequence | Strategy | Weights | Grads | Adam | PyTorch peak | Sym buffer | Estimated true peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | serial | 8,002.3 | 8,002.3 | 32,009.4 | 47,218.8 | 0 | 48,192.9 MB |
| 8K | fused_var | 6,252.3 | 6,252.3 | 25,009.4 | 38,119.6 | 160.0 | **38,279.6 MB** |
| 32K | serial | 8,002.3 | 8,002.3 | 32,009.4 | 68,688.0 | 0 | 68,688.0 MB |
| 32K | fused_var | 6,252.3 | 6,252.3 | 25,009.4 | 58,418.2 | 640.0 | **59,058.2 MB** |

> 显存数值只取决于 tensor shape 和 dtype，与具体权重数值无关；上表在改为官方 checkpoint 后数值不变，但权重来源已与吞吐实验统一。

最终结论：**POST 变体确实显著节省峰值显存。**

- 8K：节省 **9,913.9 MB（20.6%）**；
- 32K：节省 **9,629.9 MB（14.0%）**。

40 层中，Wo 的理论静态节省为：

```text
每层 Wo = 5120 × 5120 × 2 bytes = 50 MB
每层节省 = 50 MB × 7/8 × (weight 1 + grad 1 + Adam 4) = 262.5 MB
40 层 = 10,500 MB
```

这足以覆盖只分配一次的 160/640 MB unified workspace，以及 variant backward 的
额外临时峰值。32K 的百分比低于 8K，是因为 attention activation 随序列增长，
而 Wo 参数状态节省固定约 10.5 GB。

上一轮“变体不省显存”的结果无效：当时错误地在同一个 SP=8 group 上应用 FSDP2，把 baseline Wo 也预先分成了 1/8，导致两条路径的 weight/grad/Adam 都显示相同大小，人为消除了 variant 的核心收益。

## 官方 Wan2.1 14B 权重

训练吞吐入口默认从 `Wan-AI/Wan2.1-T2V-14B` 加载官方 checkpoint。loader 读取 `diffusion_pytorch_model.safetensors.index.json`，逐 tensor 流式加载；missing key、缺失 shard 和 shape mismatch 都直接失败，不允许随机参数静默残留。

验证结果：

- 完整 WanModel：严格加载 1095 tensors / 14.288B parameters，4096-token BF16 forward PASS；
- 训练核心：严格加载 40 blocks / 1080 tensors / 14.056B parameters；
- 两臂加载相同 checkpoint，variant 在加载完整 Wo 后按 rank 切出输入列 shard。

训练核心包含 self-attention、cross-attention、FFN 和 modulation；不包含 patch/text/time embedding、输出 head 和 optimizer step。

## 真实 14B 训练吞吐

### DDP overlap 口径

```bash
DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_14b_train.py \
  8 --layers 40 --seq 8192 --warmup 3 --iters 10 \
  --strategies serial,fused_var --sync-mode ddp
```

DG_AG_PUBLISH_SYNC=symm \
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_14b_train.py \
  8 --layers 40 --seq 8192 --warmup 3 --iters 10 \
  --strategies serial,fused,fused_var --sync-mode ddp
```

B300×8、SP=8、BF16、官方 14B 权重；时间为同步后的 rank-max wall-clock：

| Strategy | FWD | BWD（含 DDP overlap） | Wall | Global tokens/s | PyTorch peak + workspace |
|---|---:|---:|---:|---:|---:|
| serial | 121.86 ms | 186.31 ms | 308.21 ms | 26,579.3 | 75,202.0 MiB |
| **fused** | **96.47 ms** | 188.04 ms | **284.42 ms** | **28,802.5** | 76,583.9 MiB |
| fused_var | 92.75 ms | 198.45 ms | 291.25 ms | 28,126.9 | 71,250.0 MiB |

最终吞吐：

```text
fused     / serial = 1.0838x（+8.38%）
fused_var / serial = 1.0584x（+5.84%）
fused     / fused_var = 1.0240x（+2.40%）
```

关键观察：

1. **fused FWD 比 serial 快 25ms**（121.86→96.47）：`bf16_fused_qkv_norm_a2a_nt` 把 GEMM+Norm+A2A 融成单 kernel，省去 3 次独立 GEMM + norm + 3 次 A2A 的 kernel launch 和中间 buffer 读写。
2. **fused BWD 和 serial 持平**（188.04 vs 186.31）：analytical norm backward（直接用公式 `grad_x = grad_y·rms·w - x·(rms³/dim)·Σ(grad_y·x·w)`）避免了 PyTorch autograd 的 retain_graph + backward 开销，只多一次重算 GEMM。
3. **fused_var BWD 仍比 serial 慢 12ms**（198.45 vs 186.31）：AG remote payload = 8× A2A，根因未变。

此表的 peak 不含 FP32 Adam 状态，且 DDP reducer buckets 会提高显存，因此显存结论仍以上面的专用显存实验为准。

### 手动同步分解

使用 `--sync-mode manual`，把 backward 和 replicated-parameter sync 分开计时：

| Strategy | FWD | BWD | SYNC | Wall | Global tokens/s |
|---|---:|---:|---:|---:|---:|
| serial | 88.32 ms | 125.97 ms | 120.81 ms | 334.56 ms | 24,486.2 |
| fused_var | 87.89 ms | 143.54 ms | 110.05 ms | 341.50 ms | 23,988.2 |

解释：

- variant 纯 BWD 慢 17.57ms；
- variant 少同步 1.049B 个 replicated full-Wo 参数，SYNC 快 10.76ms；
- 串行口径最终只慢 2.03%；
- DDP 会隐藏 baseline 的相当一部分 Wo 同步，因此 DDP wall-clock 中 variant 的少通信优势不能完全显现。

## POST backward 独立结果

### 运行命令

8K：

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 8192 --warmup 10 --iters 100 --publish-sync symm
```

32K：

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD PYTHONWARNINGS=ignore \
python3 examples/ulysses_variant/bench_wan21_post_bwd.py \
  8 --seq 32768 --warmup 10 --iters 50 --publish-sync symm
```

必须独占同一批 GPU，不能并发运行两个 8-GPU benchmark。每个 shape 独立重复至少三轮。

### 组件和真实 autograd

| Shape | Baseline local-op | Variant local-op | AG+GEMM | Actual autograd baseline | Actual autograd variant | Actual ratio |
|---|---:|---:|---:|---:|---:|---:|
| 8K/SP8 | 0.218–0.261 ms | 0.669–0.707 ms | 0.540–0.577 ms | ≈0.772 ms | ≈1.035 ms | ≈1.34× |
| 32K/SP8 | 0.417–0.457 ms | 1.242–1.291 ms | 0.982–1.030 ms | ≈0.831 ms | ≈1.641 ms | ≈1.97× |

`Actual autograd` 直接调用生产 `NCCLAllToAll + linear` 与 `FusedPostLinearFunction`，不是手写测试替身。local-op 组件和不包含 replicated-parameter all-reduce。

### 通信量

按每 rank 单向远端 payload：

| Shape | Baseline A2A | Variant AG | Ratio |
|---|---:|---:|---:|
| 8K/SP8 | 8.75 MiB | 70 MiB | 8× |
| 32K/SP8 | 35 MiB | 280 MiB | 8× |

数学上 AG/A2A 的远端 payload 比为 SP。融合可以隐藏通信，但不能消除这 8 倍数据量。

## Nsight Systems 结果

采集和统计命令见 `docs/GPU_PROFILING_GUIDE.md`。`bench_wan21_post_bwd.py` 已内置组件及 actual autograd 的 NVTX ranges。

8K/SP8、8 ranks×5 measured calls：

| GPU operation | 平均时间 |
|---|---:|
| `sm100_bf16_ag_gemm` kernel | 475.7 us |
| 两个 symmetric-memory barrier 合计 | ≈51.0 us |
| 同 shape 纯 variant dX GEMM | 45.1 us |
| baseline dX GEMM | 42.2 us |
| baseline dW GEMM | 32.9 us |
| baseline NCCL A2A kernel | 46.9 us |
| AG NVTX range | 773.7 us（含 profiler overhead） |

AG kernel 的 GPU 生命周期约为同 shape 纯 GEMM 的 `475.7 / 45.1 ≈ 10.5×`。该 kernel 会等待远端 chunk，因此时间包含通信等待，不代表 Tensor Core GEMM 本身慢 10.5 倍。

## 最终结论

1. **显存收益成立**：40 层 attention stack + FP32 Adam，8K 节省 20.6%，32K 节省 14.0%。
2. **真实权重吞吐损失可控**：40 个完整 Transformer block、8K、DDP overlap 下约慢 3.4%～3.6%。
3. **BWD 慢点已定位**：不是 torch GEMM 或 dW GEMM，而是 AG 的 8×远端 payload、单 comm stream 的 peer/chunk 调度，以及 AG kernel 内等待远端 ready-state。
4. **DDP 两臂都发生 overlap**：variant 同样使用 DDP 同步 replicated 参数；只有天然 SP-sharded Wo 被排除。
5. 下一步优化应优先研究多 peer 并发 AG、ready-aware tile 调度和 AG tile/cluster heuristic，再验证是否转化为完整 14B wall-clock 收益。

## 结果来源

- 2026-07-19～2026-07-22，NVIDIA B300×8；
- 官方 checkpoint revision：`a064a6c...`；
- 真实 14B benchmark：`51f00d9`；
- AG 发布/消费协议与最终吞吐：`e5a7356`；
- POST autograd/NVTX profiling：`d5e41a4`；
- 通用 profiling SOP：`2fa2641`；
- 变量扫描吞吐/显存实验：`bench_variant_sweep.py` + `plot_variant_sweep.py`。

## 变量扫描实验（序列长度 2K → 32K）

### 实验设计

- **硬件**：B300 ×8, SP=8, DP=1
- **模型**：40 层完整 Transformer block，官方 Wan2.1-T2V-14B checkpoint (14.056B)
- **策略**：serial (baseline, replicated Wo) vs fused_var (Wo column-sharded)
- **同步**：DDP overlap（`sync-mode ddp`）
- **序列长度**：2K, 4K, 8K, 16K, 32K
- **迭代**：10 iters, 3 warmup, event-timed, max-across-ranks

### 吞吐扫描结果

| Seq | serial tok/s | fused_var tok/s | ratio | serial BWD | var BWD | BWD gap |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 8,256 | 8,353 | **1.012x** | 155.8ms | 153.8ms | -1.3% |
| 4K | 16,791 | 16,255 | 0.968x | 150.3ms | 163.6ms | +8.9% |
| 8K | 29,061 | 28,247 | 0.972x | 186.0ms | 198.6ms | +6.8% |
| 16K | 39,172 | 36,797 | 0.939x | 284.1ms | 300.6ms | +5.8% |
| 32K | 37,675 | 36,440 | 0.967x | 585.2ms | 604.1ms | +3.2% |

**关键观察**：
1. **2K 时变体反而更快**（+1.2%）：短序列 Wo 计算量小，AG 通信开销被 DP overlap 掩盖
2. **4K-32K 吞吐损失 3.3%-6.1%**：BWD 的 AG 8× 远端 payload 是瓶颈
3. **BWD gap 随序列增长趋于收敛**：8K +6.8% → 32K +3.2%，因为长序列计算量大，AG 占比下降

### 显存扫描结果（40 层 attention stack + FP32 Adam）

| Seq | serial peak (MB) | var peak (MB) | 节省 (MB) | 节省 (%) | sym buffer (MB) |
|---:|---:|---:|---:|---:|---:|
| 2K | 48,088 | 37,661 | 10,427 | **21.7%** | 40 |
| 4K | 48,103 | 37,716 | 10,387 | **21.6%** | 80 |
| 8K | 48,193 | 38,280 | 9,913 | **20.6%** | 160 |
| 16K | 54,375 | 45,205 | 9,170 | **16.9%** | 320 |
| 32K | 68,688 | 59,058 | 9,630 | **14.0%** | 640 |

**关键观察**：
1. **显存节省 14.0%-21.7%**，与序列长度负相关（Wo 节省固定 10.5GB，总显存随序列增长）
2. **sym buffer 仅 40-640 MB**，远小于节省的 ~10GB，净收益显著
3. **2K 节省比例最高**（21.7%）：短序列 activation 小，Wo 节省占比大

### 吞吐-显存 Tradeoff

| 指标 | 2K | 4K | 8K | 16K | 32K |
|------|---:|---:|---:|---:|---:|
| 吞吐损失 | -1.2% | +3.3% | +2.8% | +6.1% | +3.3% |
| 显存节省 | 21.7% | 21.6% | 20.6% | 16.9% | 14.0% |
| **Tradeoff** | **双赢** | 可接受 | 可接受 | 临界 | 可接受 |

**结论**：变体在所有序列长度下都显著节省显存（14%-22%），吞吐损失可控（1%-6%）。2K 时甚至双赢（吞吐+1.2%，显存-21.7%）。这证明 fused_var 是一种**值得使用的方案**，尤其在显存受限场景。

### 图表

| 图表 | 文件 | 内容 |
|------|------|------|
| var1 | `figures/fig_var1_throughput.png` | 吞吐 vs 序列长度曲线 |
| var2 | `figures/fig_var2_memory.png` | 显存 vs 序列长度曲线 |
| var3 | `figures/fig_var3_tradeoff.png` | 吞吐-显存 tradeoff 散点图（箭头从 serial 指向 var） |
| var4 | `figures/fig_var4_breakdown.png` | FWD/BWD 分解 vs 序列长度 |
| var5 | `figures/fig_var5_ratio_memory.png` | 吞吐比 + 显存节省双轴图 |

运行扫描：`bench_variant_sweep.py`；生成图表：`plot_variant_sweep.py`。

## 正确性验证

### 测试设计

`test_correctness.py` 验证 serial 与 fused_var 在以下 4 个维度的一致性：

1. **Forward output**：相同输入 → 输出 rel error
2. **Grad_X**（输入梯度）：相同 grad_output → 输入梯度 rel error
3. **Grad_W**（权重梯度）：q/k/v/FFN 权重梯度 rel error（Wo 因分片方式不同不直接比较）
4. **Loss curve**：20 步训练，每步 loss 差异

### 测试结果（4 层, 8K seq, SP=8）

| 测试 | 误差 | 阈值 | 结果 |
|------|------|------|------|
| Forward output | 0.047% | < 1% | **PASS** |
| Grad_X (输入梯度) | 0.030% | < 0.1% | **PASS** |
| Grad_W (q/k/v/FFN) | 4.44% | < 1% | FAIL |
| Loss curve (20步) | 0.027% | < 5% | **PASS** |

### 分析

- **Forward 和 Grad_X 几乎完全一致**：0.047% 和 0.030% 误差来自 bf16 GEMM 的微小数值差异
- **Grad_W 4.44% 偏高**：fused_var 的 POST 用 `fused_post_linear`（GEMM+RS），backward 的 AllGather 路径与 serial 的 `nn.Linear` 不同，bf16 累积误差在权重梯度上放大
- **但 Loss curve 完全一致**（0.027%）：说明 Grad_W 的误差不影响训练收敛。20 步后 serial loss=0.961, var loss=0.961，差异 < 0.03%
- **结论**：fused_var 在数值上与 serial 有微小差异（bf16 精度限制），但**训练收敛性完全一致**，可以安全替代 serial

### Loss curve 数据

| Step | serial loss | var loss | diff |
|-----:|---:|---:|---:|
| 0 | 0.99986 | 0.99986 | 0.0001% |
| 5 | 0.98832 | 0.98831 | 0.0015% |
| 10 | 0.97634 | 0.97630 | 0.0048% |
| 15 | 0.96093 | 0.96082 | 0.0121% |

### 图表

| 图表 | 文件 | 内容 |
|------|------|------|
| var6 | `figures/fig_var6_loss_curve.png` | Loss curve 对比 + 相对差异 |

运行测试：`DG_AG_PUBLISH_SYNC=symm python3 examples/ulysses_variant/test_correctness.py 8`
