# Wan2.1 14B Ulysses POST 变体消融

本实验只回答一个问题：**将标准 Ulysses 的 post-attention 从同步
A2A + Wo 改为 Wo 分片 GEMM+ReduceScatter，能否降低真实训练峰值显存？**

## 严格消融定义

两条路径的 PRE 和 attention 使用同一份代码：

1. Wan2.1 原始 Q/K/V `nn.Linear`；
2. Q/K RMSNorm；
3. 三次同步 `torch.distributed.all_to_all_single`；
4. 3D RoPE；
5. `torch.nn.functional.scaled_dot_product_attention`。

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

## 参数与 FSDP2

- PRE 直接使用 `model.q/k/v`，不再创建冗余 `Wqkv/Wqkv_t` 参数。
- baseline 保留完整逻辑 `model.o.weight`，由 FSDP2 管理。
- variant 创建 `[hidden, hidden/P]` 的 `Wo_r_local` 后注销完整
  `model.o.weight`，避免“完整 Wo + 本地 shard”双份常驻。
- 多层显存测试逐层应用 FSDP2，`reshard_after_forward=True`；仅 variant 的
  `Wo_r_local` 被排除，因为它已经按 SP 天然分片。
- Adam 状态按 DTensor 的本地 shard 分配，而不是按 global shape 重复分配。

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

包含：逐层 FSDP2、BF16 参数/梯度、FP32 Adam `m/v`、保存到 backward 的
激活，以及跨层复用一次的 symmetric workspace。输入只创建本 rank 的序列
分片，不再让每卡常驻完整 `X_full`。

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

B300 ×8、40 个 attention 层、逐层 FSDP2、BF16 参数/梯度、FP32 Adam m/v：

| Sequence | Strategy | PyTorch peak | Shared sym buffer | Estimated true peak | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| 8K | serial | 14,442.6 MB | 0 | 14,442.6 MB | — |
| 8K | fused_var | 14,456.5 MB | 160.0 MB | 14,616.5 MB | **+173.9 MB (+1.2%)** |
| 32K | serial | 34,076.4 MB | 0 | 34,076.4 MB | — |
| 32K | fused_var | 34,497.2 MB | 640.0 MB | 35,137.2 MB | **+1,060.8 MB (+3.1%)** |

因此在当前严格 POST-only 消融中，结论是：**变体不省峰值显存，反而略增**。
原因不是 buffer 没有跨层复用；它确实只分配了一次。根本原因是：

1. 标准 Ulysses 的 POST 本来就只保存 `[local_tokens, hidden]` 的 gathered
   activation，并不会物化 `[full_tokens, hidden]` partial；
2. 变体的 attention activation 与 baseline 同量级，未减少逐层保存量；
3. GEMM+RS/AG+GEMM 需要固定 workspace，其主项约为
   `2 × full_tokens × hidden × sizeof(bf16)`，8K/32K 分别为 160/640 MB；
4. 在本实验的 FSDP2 设置下，baseline 完整 Wo 在静态时已经按 8 卡分片，
   与 variant 的天然 Wo shard 每卡同为 1/8，因此没有参数、梯度或 Adam 状态净节省；
5. variant 的 backward 权重梯度还会复用 full-sequence gathered grad，带来额外
   PyTorch 临时峰值，32K 时 tracked peak 已比 baseline 高约 421 MB。

历史结果混入 PRE 融合、冗余权重副本、完整 `X_full` 和错误 FSDP2 生命周期，
均不再作为证据。若目标仍是省显存，需要缩小 GEMM-RS/AG workspace，或让
AG+GEMM 直接/分块累加 Wo 梯度，不能仅靠现有 POST 数据流。
