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

### 真正的根因（已定位并修复）

unified buffer 在**所有层的 fwd GEMM-RS 与 bwd AG-GEMM 之间跨 pass 复用**同一块对称内存。
GEMM-RS 与 AG-GEMM 各自依赖 buffer 开头的 **+1/-1 握手信号区 `[0,32)`** 及 AG 的 **`slot_state` 就绪标志**（kernel 用 `ld_acq_sys` 轮询）。
若不复位，前一次 op 留下的脏信号/slot 字节会让后一个 kernel 的 handshake 读到过期值 →
gather/AG 非确定性错乱 → `grad_X rel` 偶发抖动（0.25 / 0.44 / 0.63），且**并非随机垃圾而是稳定偏大**，指向 slot 就绪信号在 comm stream 未 flush 就被 memset 竞态。

### 修复（对齐 `tests/comm/test_unified_buffer.py` Test 6，并加固）

在 `GemmRSFunction.forward`（GEMM-RS 前）与 `.backward`（AG-GEMM 前）都插入：

```
group.barrier()
torch.cuda.synchronize()   # 强制 flush symm_mem comm stream，避免跨流竞态
sym_buffer.buffer.zero_()  # 清 barrier 信号区 + slot_state + 数据区
torch.cuda.synchronize()
group.barrier()
torch.cuda.synchronize()   # ← 关键：二次 sync 确保 comm stream 完全 flush 后再让 kernel 启动
```

**关键点**：仅 `barrier + zero_()`（如 Test 6）仍有 ~1/3 偶发失败；必须额外一道 `synchronize()`
强制 flush comm stream，才能消除 slot_state 的跨流竞态。

### 结论

**可以共享单块物理 buffer**（unified buffer 已落地）。所谓「偏移差 40MB」不是障碍；
真正要守住的是**每次在同一 buffer 上启动新 kernel 前，先 barrier+sync+zero_+sync+barrier+sync 复位握手/就绪区**。

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

`examples/wan21` 的 `GemmRSFunction`（`autograd_ops.py`）+ `fused_variant.py` 已从旧的两独立 buffer
（`GemmRSSymmBuffer` + `BF16AGGemmSymmBuffer`）迁移到单个 unified buffer：

- `fused_variant._create_buffers` 只建 `self.sym_post = get_unified_symm_buffer(group, bs, seq, dim)`（num_max_tokens_per_rank = bs*(seq//sp) == self.local_m，正好等于 gemm_rs 调用传的 local_m）。
- `GemmRSFunction.forward` 用该 buffer 跑 GEMM+RS；`backward` 用 `sym_buffer.ag_x` 写 grad_y、调 `bf16_ag_gemm_nt(sym_buffer, ...)` 跑 AG+GEMM、从 `sym_buffer.ag_slots_x` 读 gathered grad_y 算权重梯度。
- 跨层通过 `share_buffers_from` 共享**同一个** buffer（destroy 时只 destroy 一次）。

**状态（2026-07-17 已验证）**：本机即 **B300 SXM6 ×8**（CUDA 13.0），2 卡 `examples/debug_var_bwd.py` 实跑
（`DG_JIT_USE_NVRTC=1`），连跑 4/4 全部 PASS：

- `grad_X rel (serial vs var)` ≈ **0.000000–0.000002**（迁移前唯一失败模式 0.2572 已消失）
- `fwd rel (serial vs var)` ≈ **0.0026**（≤ 预期 ~0.028）
- 退出码 0，无 SIGABRT（teardown 时在 `destroy_process_group()` 前显式 `destroy_buffers()` 释放 unified buffer，避免 `~CUDASymmetricMemory` 析构时序崩）

注意：`debug_var_bwd.py` 的 rel 在**修复前不稳定**（0.44/0.63 抖动），正是脏 signal/slot_state 竞态
的直接证据；加固同步后稳定归零。上线前建议在其他 seq/bs 配置下再抽测一次。
