# Wan2.1 14B Ulysses POST 变体消融

本实验只回答一个问题：**将标准 Ulysses 的 post-attention 从同步
A2A + Wo 改为 Wo 分片 GEMM+ReduceScatter，能否降低真实训练峰值显存？**

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

`serial` 不调用任何 DeepGEMM 通信融合算子。`fused_std` 仅保留为旧命令行的
兼容别名，执行内容与 `serial` 完全相同，不属于本消融的第三条路径。

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

AG+GEMM 的高层入口 `bf16_ag_gemm_nt_with_input` 接收显式输入，隐藏
`.ag_x/.ag_slots_x` 等内部布局。底层 C++ 负责 stream/event 和 `slot_state`
复位。不得在 Function 中对数百 MB workspace 做逐调用 `zero_()`；当前只在
AG pull 前执行输入发布同步，防止某 rank 提前读取远端尚未写好的输入。

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
- 若未来有 `SP=8 × DP>1` 的二维 mesh，两条路径仍可沿 DP 维做相同 FSDP；
  variant 相对 baseline 的 SP 维 Wo 分片收益仍然存在。

## 显存统计

### 单层

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD \
  python examples/bench_wan21_mem.py 8 serial,fused_var
```

报告：

- `torch.cuda.max_memory_allocated()` 的 FWD/BWD 累计峰值；
- symmetric workspace 的实际字节数；
- Wo 逻辑大小和本地梯度大小。

### 多层训练

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD \
  python examples/bench_wan21_mem_train.py 8 40 32768 serial,fused_var
```

包含：SP=8/DP=1 下真实本地参数所有权、BF16 参数/梯度、FP32 Adam `m/v`、
保存到 backward 的激活，以及跨层复用一次的 symmetric workspace。输入只创建
本 rank 的序列分片，不再让每卡常驻完整 `X_full`。

`torch.cuda.max_memory_allocated()` 不保证统计 symmetric memory，所以文档将
PyTorch 峰值和 workspace 分项报告，并给出二者相加的估算峰值；最终结论还应
用 `max_memory_reserved()`、NVML 或 `cudaMemGetInfo` 交叉验证。

## 正确性

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python examples/debug_var_bwd.py
```

2026-07-17，B300 2 卡连续 4 次结果：

```text
grad_X rel (serial vs var): 0.000000
fwd rel (serial vs var):    0.002964 ~ 0.002968
```

前向差异来自 BF16 GEMM/ReduceScatter 的归约顺序；输入梯度与同步 baseline
一致。buffer 必须在 `destroy_process_group()` 前显式释放。

## 显存结果与结论

B300 ×8、40 个 attention 层、SP=8/DP=1、FA4、BF16 参数/梯度、FP32 Adam m/v：

| Sequence | Strategy | Weights | Grads | Adam | PyTorch peak | Sym buffer | Estimated true peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | serial | 8,002.3 | 8,002.3 | 32,009.4 | 48,191.3 | 0 | 48,191.3 MB |
| 8K | fused_var | 6,252.3 | 6,252.3 | 25,009.4 | 38,117.4 | 160.0 | **38,277.4 MB** |
| 32K | serial | 8,002.3 | 8,002.3 | 32,009.4 | 68,686.5 | 0 | 68,686.5 MB |
| 32K | fused_var | 6,252.3 | 6,252.3 | 25,009.4 | 58,416.6 | 640.0 | **59,056.6 MB** |

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

上一轮“变体不省显存”的结果无效：当时错误地在同一个 SP=8 group 上应用 FSDP2，
把 baseline Wo 也预先分成了 1/8，导致两条路径的 weight/grad/Adam 都显示相同大小，
人为消除了 variant 的核心收益。
