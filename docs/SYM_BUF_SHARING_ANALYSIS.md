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

### 根因

1. **RS 的 `partial` 区域在 forward 写入后，没有清零**
2. **AG 的 comm kernel 往 `local_x` (offset 32) 写 grad_y**，覆盖了 RS `partial[0]` 的数据
3. **AG 的 comm kernel 往 `slots_x` (offset 40MB) 写其他 rank 的 grad_y**，这恰好对应 RS 的 `partial[1:]` 区域
4. 但 **AG 的 `slots_x[0]` 在 offset 40MB，而 RS 的 `partial[0]` 在 offset 32** — `slots_x[0]` 实际上落在了 RS `partial[1]` 的位置
5. weight grad 读 `slots_x[:sp]` 得到的是 `partial[1:sp+1]`，但 `partial[sp]` 越界！

### 结论

**不能直接共享 raw buffer**，因为两个 layout 的数据区起始偏移不同（差 40MB 的 `local_x` 前缀）。

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
