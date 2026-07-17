# Symmetric Buffer Sharing Analysis: GEMM+RS ↔ AG+GEMM

## 背景

fused_var Ulysses 变体的 POST 阶段使用两个融合算子：
- **Forward**: `bf16_gemm_rs_nt` (GEMM + Reduce-Scatter)，用 `GemmRSSymmBuffer`
- **Backward**: `bf16_ag_gemm_nt` (All-Gather + GEMM)，用 `BF16AGGemmSymmBuffer`

当前各分配独立的 sym_buf。问题：能否让两者共享同一块物理内存？

## 理论分析

### 两个 sym_buf 的结构对比

以 sp=8, local_m=4096, dim=5120, bf16 为例：

#### GEMM+RS (`GemmRSSymmBuffer` / `GemmRSWorkspace`)

```
offset 0:   barrier (32 B)
offset 32:  partial [sp, local_m, dim] = [8, 4096, 5120] bf16 = 320 MB  ← GEMM partial sums
offset 320MB: ready [sp, m_blocks, n_blocks] = ~0.1 MB  ← per-tile ready flags
total: ~320 MB
```

#### AG+GEMM (`BF16AGGemmSymmBuffer` / `BF16AGGemmWorkspace`)

```
offset 0:   barrier (32 B)
offset 32:  local_x [local_m, dim] = [4096, 5120] bf16 = 40 MB  ← 本 rank 的输入
offset 40MB: slots_x [sp, local_m, dim] = [8, 4096, 5120] bf16 = 320 MB  ← 所有 rank 的输入拷贝
offset 360MB: slot_state [sp, 4] = ~0.1 MB  ← per-chunk ready flags
total: ~360 MB
```

### 偏移对比

| 物理偏移 | GEMM+RS | AG+GEMM |
|---|---|---|
| 0 | barrier | barrier |
| 32 | **partial[0]** | **local_x** |
| 32 + 40MB | partial[1] | **slots_x[0]** |
| 32 + 80MB | partial[2] | slots_x[1] |
| ... | partial[r] | slots_x[r-1] |

**关键问题**：RS 的 `partial[0]` 和 AG 的 `local_x` 占用同一块物理内存（offset 32）。
但语义完全不同：
- RS `partial[0]` = rank 0 为目标 rank 0 算的 GEMM partial sum
- AG `local_x` = 本 rank 的输入 (grad_y)

### 偏移不对齐导致的问题

如果共享 buffer：
1. **Forward (GEMM+RS)**: kernel 往 `partial[0..sp-1]` 写 partial sums（offset 32 到 320MB）
2. **Backward (AG+GEMM)**: comm kernel 往 `local_x` (offset 32) 写 grad_y，往 `slots_x[0..sp-1]` (offset 40MB 到 360MB) 写所有 rank 的 grad_y

backward 读 `slots_x[:sp]` 做 weight grad 时，实际读的是物理偏移 40MB 开始的数据。
但 weight grad matmul 需要 `[full_m, dim]` = `slots_x[:sp].reshape(full_m, dim)`。
这在 AG layout 下是正确的。

**但**：AG+GEMM 的 TMA descriptor 是基于 AG layout 创建的，它会从 offset 40MB (`slots_x[0]`) 开始 TMA load。
而 GEMM+RS 的 TMA descriptor 是基于 RS layout 创建的，它从 offset 32 (`partial[0]`) 开始 TMA load。
两者共享 buffer 时，C++ 端各自用自己的 layout 函数计算偏移，所以 TMA 地址是对的。

**实测发现**：backward 正确性 bX_rel=0.2572（FAIL），说明 AG+GEMM kernel 读到了错误的数据。

### 初步（错误）假设

早先怀疑是「两个 layout 数据区起始偏移不同（差 40MB 的 `local_x` 前缀），不能直接共享 raw buffer」。
**该假设已被推翻**：`UnifiedSymmBuffer` 实测证明单物理 buffer 可安全服务 fwd/bwd——fwd/bwd 不并发，各自按自己 layout 算 TMA 偏移，offset 差在单 pass 复用下无害。

### 真正的根因（2026-07-17 复核）

共享 layout 本身没有问题；非确定性 `grad_X rel≈0.44` 的直接原因是 **AG 输入发布竞态**：
每个 rank 把本地 `grad_y` 写进 `local_x` 后，AG 的独立 communication stream 会立即 pull
其他 rank 的 `local_x`。如果远端 rank 尚未完成 copy，就会拉到上一轮数据。

之前在 autograd Function 里执行 `barrier + synchronize + buffer.zero_()` 偶尔能掩盖该竞态，
但它并不是正确协议：

- GEMM-RS 的 NVLink barrier 是 phase/sign 自复位协议，逐调用清零可能擦掉 peer 已到达的 signal；
- AG launcher 自己会在 comm stream 上清零 `slot_state`；
- 对数百 MB data region 做 `zero_()` 没有语义必要，并破坏 overlap。

### 当前修复

新增高层入口 `bf16_ag_gemm_nt_with_input(d, a, b, workspace, num_tokens)`：

1. copy 本地输入到 workspace `local_x`；
2. 等待 copy 完成；
3. 所有 rank 完成设备端 barrier；
4. 才允许 C++ AG comm stream pull 远端输入；
5. 返回已经 gather 的 slots view，供 Wo shard 权重梯度 GEMM 复用。

因此 `FusedPostLinearFunction` 不再访问 `.ag_x/.ag_slots_x` 内部布局，也不再清零整个
workspace。2 卡连续 4 次验证 `grad_X rel=0.0`。

### 结论

**可以共享单块物理 buffer**。必须同步的是“所有 rank 的 AG 输入已经发布”，而不是
粗暴复位整块物理内存。后续可将该输入发布 barrier 下沉为纯 GPU/NVLink 协议，以移除
当前高层入口中的 host synchronize。

## 可能的解决方案

### 方案 1: 修改 RS layout 加入 local_x 前缀
在 `GemmRSWorkspace` 中添加一个 `local_x` 区域，使偏移与 AG layout 对齐。
- 优点：可以共享 buffer，省 320MB
- 缺点：需要改 C++ layout + kernel，改动大

### 方案 2: 让 weight grad 不需要 full grad_y
当前 weight grad 需要 `[full_m, dim]` 的 grad_y_full。如果改成分块计算（每 rank 的 grad_y 分别做 matmul 后累加），就不需要 full buffer。
- 优点：不依赖 sym_buf 共享
- 缺点：多次小 matmul，效率低

### 方案 3: 让 AG+GEMM kernel 同时输出 weight grad
让融合 kernel 在算 grad_attn 的同时，也累加 weight grad。
- 优点：最优雅，无需额外 buffer
- 缺点：需要改 CUDA kernel

### 方案 4: 当前方案（分开 buffer）
保持 GEMM+RS 和 AG+GEMM 各自独立的 sym_buf。
- 优点：简单，已验证
- 缺点：多占 320MB

## 实测数据（分开 buffer）

### 单层 trace（32K seq, 8 GPU）

| 阶段 | serial | fused_var | 差值 |
|---|---|---|---|
| baseline | 608 MB | 937 MB | +329 (sym_buf) |
| fwd peak | 1565 MB | 1878 MB | +313 |
| bwd peak | 1565 MB | 1878 MB | +313 |
| after bwd | 813 MB | 1114 MB | +301 |

### 多层训练（32K seq, 8 GPU, FSDP2 + Adam states）

| Layers | serial | fused_var | diff | per-layer diff |
|---|---|---|---|---|
| 1 | 4,264 | 4,610 | 346 | 346 |
| 4 | 14,874 | 16,146 | 1,272 | 318 |
| 8 | 29,020 | 31,483 | 2,463 | 308 |
| 16 | 57,313 | 62,117 | 4,804 | 300 |

### 每层多吃的 ~300MB 来源

1. **weight grad matmul 临时分配**：`grad_y_full.t() @ attn` 需要 `[full_m, dim]` = 335MB 的 NT GEMM 临时（cuBLAS 固有），serial 只需 `[lm, dim]` = 42MB → 差 293MB/layer
2. **sym_buf 常量**：960MB（跨层复用），不随层增长
3. Wo 切分省的：~-44MB/layer（权重+梯度+Adam）

### 关键发现

- **1 层时 fused_var 反而省显存**（-614MB）：sym_buf 被Wo省的权重+梯度+Adam 摊薄
- **多层时 gap 随层线性增长**：每层 weight grad matmul 临时 293MB 累积
- **根因是 GEMM+RS 架构固有**：Wo 权重梯度需要 full grad_y（`[full_m, dim]`），而 serial 只需 local（`[lm, dim]`）

## 2026-07-17 更新：UnifiedSymmBuffer 已解决单 buffer fwd+bwd 复用

`deep_gemm/unified_buffer/UnifiedSymmBuffer`（`get_unified_symm_buffer`）实现了本文档想要的「单物理 buffer 服务所有算子」。对 GEMM-RS(fwd) 与 AG-GEMM(bwd)：

- **一块** `symm_mem.empty(max(rs_bytes, ag_bytes))` 物理分配，rendezvous **一次**（消除每层每步毫秒级开销）。
- fwd 用 RS 视图（`partial` @ offset 32）；bwd 用 AG 视图（`ag_x` @ 32 / `ag_slots_x` @ 32+sp*M*H*2），两者在**同一块物理内存**上 reinterpret。
- fwd/bwd **不并发**（整轮 forward 跑完才进 backward，每层顺序执行），故物理内存可安全复用，无交叉读——之前「offset 差 40MB」的分析在本方案下不构成问题（kernel 各自按自己 layout 算偏移，且不会读到对方 pass 的数据）。
- 已通过 `tests/comm/test_unified_buffer.py`（2/8 卡 GEMM-RS + AG-GEMM 7/7 PASS）。

### wan21 迁移（2026-07-17）

`examples/wan21` 的 POST 变体已迁移到单个 unified buffer，并重构成
`FusedPostLinearFunction`：

- `fused_variant._create_buffers` 只创建一个 `get_unified_symm_buffer(...)`；
- forward 调 GEMM+RS，backward 经 `bf16_ag_gemm_nt_with_input` 调 AG+GEMM；
- Function 只描述 autograd 数学，不创建、销毁或清零 workspace；
- layer 0 拥有 workspace，其余层借用，跨全部层与前后向只分配一次。

**状态（2026-07-17 已验证）**：B300 2 卡 `examples/debug_var_bwd.py` 连跑 4/4：

- `grad_X rel (serial vs var) = 0.000000`
- `fwd rel (serial vs var) = 0.002964–0.002968`
- teardown 在 `destroy_process_group()` 前显式释放 workspace，无 SIGABRT。

严格 POST-only 显存消融必须区分 SP 与 DP/FSDP 维度。B300×8 全部用于 SP、DP=1 时，
baseline Wo 在 SP rank 间复制，variant 才拥有 1/8 shard；不能错误地沿同一 SP group
再做 FSDP。修正后 40 attention 层实测：8K true peak 38,277.4 vs 48,191.3 MB
（节省 9,913.9 MB/20.6%）；32K 为 59,056.6 vs 68,686.5 MB（节省
9,629.9 MB/14.0%）。160/640 MB unified workspace 远小于 40 层 Wo 权重、梯度和
Adam 状态约 10.5 GB 的理论节省。
