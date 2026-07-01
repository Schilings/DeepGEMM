# Fused QKV GEMM + Norm + A2A 融合算子需求

## 背景

Wan2.1 官方模型的 self-attention forward 数据流：
```
x [B, S, dim]
  → q = self.q(x)          [B, S, dim]    (Q projection, has bias)
  → q = self.norm_q(q)     [B, S, dim]    (RMSNorm on full dim)
  → q = q.view(B, S, H, D)  [B, S, H, D]
  → q = rope_apply(q)       [B, S, H, D]   (3D RoPE)
  → attention
```

K 的处理同理。V 不做 norm 也不做 RoPE。

## 问题

当前 DeepGEMM 的 `bf16_gemm_a2a_transpose_nt` 算子做 GEMM + A2A-scatter（rank-major）：
```
x [local_m, K] @ Wqkv[N, K]^T → A2A-scatter(heads) → qkv[bs, seq, local_n]
```

GEMM 和 A2A 在**同一个 kernel** 里完成，**无法在 GEMM 后、A2A 前插入 norm_q/norm_k**。

结果：A2A scatter 后每 rank 只持有 `local_n` 维（640），而 `norm_q` 期望 `dim` 维（5120）。
- 如果跳过 norm：输出错误（fwd_rel ~0.85）
- 如果在 local_n 上做 norm：norm_q weight shape 不匹配（[5120] vs [640]）
- 如果在 A2A 后做 norm 但用 local 切片：数学不等价（RMSNorm 是 per-element scale，但 head 间归约顺序变了）

## 需求：Fused QKV GEMM + Norm + A2A 算子

### 目标
开发一个融合算子，在**单个 kernel** 内完成：
1. Q/K/V GEMM（分开或 fused，带 bias）
2. **norm_q / norm_k**（RMSNorm，在 A2A scatter 之前，作用于 full dim）
3. A2A-transpose scatter（rank-major，把每 rank 的 head 组送达目标 rank）

### 数据流
```
输入: x [local_m, dim]  (bf16, seq-sharded, full hidden)
权重: Wq [dim, dim], Wk [dim, dim], Wv [dim, dim]  (bf16, NT layout, with bias)
     norm_q_weight [dim], norm_k_weight [dim]  (float32, RMSNorm scale)
     eps (float, RMSNorm epsilon)

Kernel 内部:
  1. q = x @ Wq^T + bias_q    [local_m, dim]
  2. k = x @ Wk^T + bias_k    [local_m, dim]
  3. v = x @ Wv^T + bias_v    [local_m, dim]
  4. q = rmsnorm(q, norm_q_weight, eps)   [local_m, dim]  ← 关键：full dim norm
  5. k = rmsnorm(k, norm_k_weight, eps)   [local_m, dim]
  6. [q, k, v] → A2A-transpose-scatter (rank-major head groups)
     每个 rank 收到: qkv [bs, seq, 3*local_n]  (local_n = local_nh * head_dim)

输出: qkv [bs, seq, local_nqkv]  (bf16, ready for RoPE + attention)
```

### 关键设计点

1. **三个 GEMM 可选融合或分开**：
   - 融合方案：三个 GEMM 的 epilogue 都写进同一个 sym buffer 的不同区域，然后统一 A2A。一个 kernel 做三个 GEMM + norm + A2A。
   - 分开方案：三个独立 kernel（每个一个 GEMM + norm），最后统一 A2A。更简单但多一次 kernel launch。

2. **RMSNorm 在 epilogue 中**：
   - GEMM epilogue 原本是写 A2A scatter 的 P2P 写入。改成先写本地 buffer（full dim），做 RMSNorm（per-element scale），再 A2A scatter。
   - RMSNorm 计算量很小（一个 reduce + 一次 elementwise），可以和 GEMM epilogue 融合（在 epilogue 的 store 前加 scale）。
   - 但 RMSNorm 需要先算 `mean(x^2)` 再 `x * rsqrt(mean + eps) * weight`——这是一个两 pass 操作，epilogue 是单 pass。可能需要把 RMSNorm 做成 epilogue 的一部分或加一个轻量 kernel。

3. **Norm 输入 dtype**：
   - GEMM 输出 bf16，RMSNorm 用 float32 计算（官方做法），输出 bf16 再 A2A。
   - 或者 GEMM 输出 fp32 accumulator → RMSNorm（fp32）→ cast bf16 → A2A。

4. **Bias**：
   - 官方 Q/K/V 有 bias，需要在 GEMM 后加 bias。
   - 可以在 epilogue 里加（C = A@B + bias + rmsnorm）。

5. **V 不做 norm**：
   - V 直接 GEMM + bias + A2A，不经过 RMSNorm。

### 性能预期
- 消除 3 次独立 GEMM 的 kernel launch 开销
- norm 在 GEMM epilogue 里 fused，不增加额外 HBM 读写
- A2A 仍然是 rank-major P2P scatter
- 预期比当前分开方案快 ~1.3-1.5x（省 2 次 kernel launch + 2 次 HBM 读写 norm）

### 约束
- `dim % 128 == 0`（GEMM K 对齐）
- `num_heads % sp == 0`（head 组均分）
- `head_dim % 8 == 0`（uint4 向量化）
- `seq % sp == 0`（seq 均分）
- `(seq // sp) % 128 == 0`（M-tile 对齐）

### 对偶 backward 算子
- Forward: GEMM + Norm + A2A-scatter
- Backward: A2A-gather + Norm-inverse + GEMM
  - A2A-gather：收集各 rank 的 head 组
  - Norm-inverse：RMSNorm 的 backward = `grad / rms * weight + grad * (-x * (x·grad) / (rms^3 * dim))`
  - GEMM backward：`grad_X = grad_local @ Wqkv`，`grad_Wqkv = grad_local^T @ x`

### 实现路径建议
1. 先实现 **分开方案**（3 个 GEMM + norm kernel + 统一 A2A），验证正确性
2. 再把 3 个 GEMM + norm 融合成**单 kernel**（epilogue 里加 norm + bias）
3. 最后加 **autograd.Function 包装**（backward = 对偶 A2A+norm-inverse+GEMM）

### 优先级
- **高**：这是让 fused 策略（fused_std/fused_var）恢复使用 fused kernel 的前提
- 当前 fused 策略退化为 serial（分开 Q/K/V + NCCL A2A），失去融合算子的性能优势
- 开发后 fused_std/fused_var 可以切换回 fused kernel，恢复 ~1.4x 加速比
