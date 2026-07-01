# Fused QKV GEMM + RMSNorm + A2A-transpose 设计

> 需求文档：`docs/FUSED_QKV_NORM_A2A.md`
> 基座算子：`bf16_gemm_a2a_transpose_nt`（GEMM + A2A-transpose，单 kernel，SM100）
> 目标硬件：SM100 / B300（单机 NVLink 域 ≤8 卡）
> **性能**：geo **1.61x** vs serial（8 卡 B300）
> **正确性**：8 卡 20/20 PASS（MHA/GQA × norm on/off）

---

## 0. 核心设计：Norm 推迟

RMSNorm 的 `y = x * rsqrt(mean(x²) + eps) * weight` 是两 pass，但可以拆成：
1. **Pass 1（GEMM kernel 内）**：算 `sum(x²)` → `rms = rsqrt(sum/dim + eps)`
2. **Pass 2（对端）**：`out = x * rms * weight`（纯 elementwise，可融进后续算子）

这样 GEMM kernel 和基座几乎一样——epilogue 多了 `pre_cast` hook 算 x²sum + scatter rms，
**不需要 grid sync、不需要两阶段、不需要 norm kernel**。

**RMSNorm 设计说明**：
- **V 不做 norm**（只 Q/K 做，符合 Wan2.1 官方实现）
- **Q 和 K 分开做 RMSNorm**（各自独立算 `sum(x²)` 和 `rms`，不合并）
- RMSNorm 跨 full dim（如 Q 段的 `q_dim=5120` 所有 head 一起算 mean(x²)）
- norm 可选（`norm_q_weight=None` 跳过 Q 的 norm，`norm_k_weight=None` 跳过 K 的 norm）
- 返回 `(out, rms)`：out 是未 norm 的 QKV，rms 是 `[bs, seq, 2]` 标量，对端自己乘

---

## 1. 问题

现有 `bf16_gemm_a2a_transpose_nt` 把 GEMM 和 A2A scatter 融在**同一个 epilogue** 里：
TMEM → SMEM → TMA store（直推 peer HBM）。**无法在 GEMM 后、A2A 前插入 RMSNorm**。

RMSNorm 公式：`y = x * rsqrt(mean(x²) + eps) * weight`，是**两 pass**：
1. Pass 1：对每行（full dim=5120）做 `sum(x²)` → `rms = rsqrt(sum/dim + eps)`
2. Pass 2：`y = x * rms * weight`

而 GEMM epilogue 是**单 pass tile-wise**：每个 CTA 只看到 128×128 tile，无法访问同一行的其他 N-tile。

---

## 2. 方案：两个版本并行对比

同时实现两个版本，用 benchmark 数据决定哪个更有潜力：

- **v2a（双 kernel）**：Kernel1(GEMM+本地写+x²sum) → Kernel2(norm+scatter)
  - 已有 CUDA 骨架：`sm100_bf16_rmsnorm_a2a_scatter.cuh`
  - 优点：每个 kernel 职责单一，易调试/调优；Kernel2 可独立调 tile/SM 配置
  - 缺点：1 次额外 kernel launch（~5us）+ D 在 HBM 的写→读（两 kernel 间）

- **v2b（单 kernel）**：persistent 两阶段 + kernel 内 grid_sync
  - 复用 `comm::grid_sync` / `comm::nvlink_barrier`
  - 优点：省 launch 开销；与基座 `bf16_gemm_a2a_transpose_nt` 同构
  - 缺点：warp 角色切换（GEMM→norm/scatter）实现复杂；SMEM/TMEM 生命周期管理更难

两版本共用：layout / host TMA descriptor / Python API / test / bench

### 2.1 为什么单 kernel 可行（v2b）

基座 `bf16_gemm_a2a_transpose_nt` 是单 kernel——GEMM epilogue 直接 P2P TMA scatter，无需跨 tile 信息。
本算子加了 RMSNorm，是**两 pass**（先 `sum(x²)` 全行 reduce，再 `x * rsqrt * weight`），单个 tile 看不到整行，
所以需要在 kernel 内部做 **grid sync**（复用现有 `comm::grid_sync` / `comm::nvlink_barrier`）。

单 kernel 方案省掉：
- kernel launch 开销（~5us）
- D 的 HBM 写→读（虽然单 kernel 仍需写本地 HBM 再读，因为 TMEM/SMEM 放不下整行，但省了 kernel 间的额外同步）

### 2.2 整体架构（单 kernel，两阶段 persistent）

```
sm100_bf16_fused_qkv_norm_a2a_impl（单 kernel，256T = 8 warp，与 GEMM-RS/A2A 同构）:

  ── 阶段 1: GEMM + 写本地 HBM + x² sum ──
  起始 nvlink_barrier（tag 71: 保证上轮 peer 读完我的 buffer）
  W0: TMA Load A+B
  W1: MMA Issue
  W2: Reserved / TMEM Allocator
  W4-W7: Epilogue —— TMEM→SMEM→TMA store 到**本地 HBM**
                        + x² partial sum → atomic add 到 sum_buffer[row, q_or_k_idx]
                        （仅 Q/K 段，V 段不做 sum）

  ── grid_sync（kernel 内部，所有 tile 完成）──
  comm::grid_sync / comm::nvlink_barrier（保证所有 CTA 的本地写 + x² sum 完成）

  ── 阶段 2: Norm + A2A scatter ──
  W0-W7: 重读本地 HBM tile → SMEM
         RMSNorm elementwise（if Q/K 段 && norm enabled）
         TMA store scatter → peer HBM（scatter_maps[dst_rank]）
  ptx::tma_store_wait<0>  ── drain 所有 push
  结束 nvlink_barrier（tag 72: 保证所有 scatter 全局可见）
```

### 2.3 为什么不能在 GEMM epilogue 里直接 scatter（像基座那样）

基座的 epilogue 是 `TMEM → SMEM → TMA store to peer`，一步到位。
本算子需要在 `TMEM → SMEM` 后先做 RMSNorm（需要全行 x² sum），才能 scatter。
而 RMSNorm 的 x² sum 需要**所有 N-tile 完成**后才有——所以必须有 grid sync 把两阶段隔开。

### 2.4 x² sum 的计算（参考 sm100_tf32_hc_prenorm_gemm）

`sm100_tf32_hc_prenorm_gemm` 用专门的 cast-and-reduce warps（W4-W7 之外的额外 warps）
在 GEMM 流水线中同时做 bf16→fp32 cast + x² reduce。我们的方案：
- epilogue warps（W4-W7）在 STSM 阶段（TMEM→SMEM）拿到 fp32 的 GEMM 结果
- 在写 SMEM 的同时，计算 `values[i] * values[i]` 的 partial sum
- 对 Q 段的 tile，atomic add 到 `sum_buffer[global_m, 0]`
- 对 K 段的 tile，atomic add 到 `sum_buffer[global_m, 1]`
- V 段不做 sum

### 2.5 与现有算子的关系

| 维度 | `bf16_gemm_a2a_transpose_nt` | 本 Fused QKV+Norm+A2A |
|------|------|------|
| kernel 数 | 1 | 1（两阶段 persistent + grid sync） |
| GEMM | ✓ | ✓ |
| Epilogue 目标 | peer HBM (P2P scatter) | **本地 HBM**（阶段1）→ peer HBM（阶段2） |
| x² sum | ✗ | **✓ (atomic add in 阶段1 epilogue)** |
| RMSNorm | ✗ | **✓ (阶段2 elementwise)** |
| A2A scatter | ✓ (in epilogue) | ✓ (阶段2，grid sync 后) |
| Grid sync | ✗ | **✓ (阶段1→2 之间)** |
| NVLink barrier | init + final | init + grid_sync + final |

---

## 3. 数据流

### 3.1 通用 API（norm 可选 + GQA）

```
输入: x [bs*local_seq, K]  (bf16, seq-sharded, full hidden)
权重: b [N, K]  (bf16, NT layout, N = (q_nheads + 2*kv_nheads) * head_dim)
     # b 的布局: [Wq(q_nheads*hd, K); Wk(kv_nheads*hd, K); Wv(kv_nheads*hd, K)]
     # GQA: q_nheads > kv_nheads (如 q=40, kv=8, gqa=5)
     # MHA: q_nheads == kv_nheads

norm_q_weight [q_dim]  (fp32, RMSNorm scale for Q; None = 不做 norm)
norm_k_weight [k_dim]  (fp32, RMSNorm scale for K; None = 不做 norm)
eps (fp32, RMSNorm epsilon)
bias [N]  (bf16, optional; None = 无 bias)

N 结构:
  Q range: [0,            q_nheads*hd)           # q_dim = q_nheads * head_dim
  K range: [q_dim,         q_dim + kv_dim)        # kv_dim = kv_nheads * head_dim
  V range: [q_dim+kv_dim,  q_dim + 2*kv_dim)     # V 不做 norm
```

### 3.2 A2A scatter 逻辑（GQA 支持）

```
输出: qkv [bs, seq, local_n_total]  (bf16)
  local_n_total = (local_q_nheads + 2*local_kv_nheads) * head_dim
  local_q_nheads  = q_nheads  / num_ranks
  local_kv_nheads = kv_nheads / num_ranks

dst 输出布局: [local_q_n | local_kv_n | local_kv_n]
  local_q_n  = local_q_nheads  * head_dim
  local_kv_n = local_kv_nheads * head_dim

scatter 索引:
  对 GEMM 输出 D[global_m, n]:
    if n < q_dim:                              # Q 段
      rel = n
      dst_rank = rel / local_q_n
      base_n = rel % local_q_n                 # 写到 dst [0, local_q_n)
    elif n < q_dim + kv_dim:                  # K 段
      rel = n - q_dim
      dst_rank = rel / local_kv_n
      base_n = rel % local_kv_n + local_q_n   # 写到 dst [local_q_n, local_q_n+local_kv_n)
    else:                                      # V 段
      rel = n - q_dim - kv_dim
      dst_rank = rel / local_kv_n
      base_n = rel % local_kv_n + local_q_n + local_kv_n  # 写到 dst [local_q_n+local_kv_n, ...)

  base_m_idx = b*seq + rank_idx*local_seq + s_local  (同现有, pre-attn seq 分片)
```

### 3.3 norm 可选

- `norm_q_weight is None` → Q 段不做 RMSNorm（直接 GEMM + bias + scatter）
- `norm_k_weight is None` → K 段不做 RMSNorm
- V 段永远不做 norm
- 当两者都为 None 时，退化为带 bias 的 `bf16_gemm_a2a_transpose_nt`

Kernel 1:
  D = x @ Wqkv^T   [bs*local_seq, N]
  → write D to local_buffer [bs*local_seq, N] (bf16)
  → sum_q[r] = Σ D[r, 0:dim]²   (fp32, per-row atomic)
  → sum_k[r] = Σ D[r, dim:2*dim]²  (fp32, per-row atomic)
  → V range: no sum needed

Kernel 2:
  for each tile of local_buffer:
    load tile → SMEM
    if Q range: rms = rsqrt(sum_q[row]/dim + eps); tile = tile * rms * norm_q_weight[col]
    if K range: rms = rsqrt(sum_k[row]/dim + eps); tile = tile * rms * norm_k_weight[col]
    if V range: identity
    → TMA store scatter to peer (scatter_maps[dst_rank])

输出: qkv [bs, seq, 3*local_n]  (bf16, BSHD ready for RoPE + attention)
  local_n = N / num_ranks = 3 * local_nheads * head_dim
```

---

## 4. Symm buffer 布局

```
[0 .. 32)            barrier/signal 区（kNumBarrierSignalBytes=32）
[32 .. 32+OUT)       输出区 out: [bs*seq, local_n]，row-major, stride=local_n
                      OUT = bs*seq*local_n*elem_size
                      local_n = (3*dim) / num_ranks  (含 Q/K/V)
总字节 align(32+OUT, 16)
```

**本地 buffer（非 symm）**：
- `local_buffer [bs*local_seq, N]` (bf16) — Kernel 1 的 GEMM 输出
- `sum_buffer [bs*local_seq, 2]` (fp32) — per-row x² sum for Q(0) and K(1)

---

## 5. Kernel 1 详细设计：`sm100_bf16_gemm_rmsnorm_local`

### 5.1 复用 GEMM-A2A-transpose 的 GEMM 主体

Warp 布局不变（256T = 8 warp）：
- W0: TMA Load A+B
- W1: MMA Issue
- W2: Reserved / TMEM Allocator
- W4-W7: Epilogue

### 5.2 Epilogue 改动

现有 epilogue（`sm100_store_cd`）：
```
TMEM → SMEM (cast FP32→BF16) → TMA store to scatter_maps[dst_rank]
```

改为：
```
TMEM → SMEM (cast FP32→BF16) → TMA store to LOCAL buffer
                            + x² partial sum → atomic add to sum_buffer
```

**x² sum 计算**：在 STSM 阶段，数据已在 FP32 寄存器中（`values[0..7]`）：
```cuda
// 现有：cast FP32→BF16 并写 SMEM
ptx::st_shared(smem_ptr, cast_into_bf16_and_pack(values[0], values[1]), ...);

// 新增：同时计算 x² partial sum（仅 Q/K 范围）
if (n_col < 2 * dim) {  // Q or K range
    float sq_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sq_sum += values[i] * values[i];
    // atomic add to sum_buffer[row, q_or_k_index]
    int sum_idx = (n_col < dim) ? 0 : 1;  // Q=0, K=1
    atomicAdd(&sum_buffer[global_m + lane_row, sum_idx], sq_sum);
}
```

**注意**：V 范围（`n_col >= 2*dim`）不做 sum。

**TMA store 目标**：从 `scatter_maps[dst_rank]` 改为**本地 buffer 的 2D TMA descriptor**。

### 5.3 为什么用 atomic add

每行（dim=5120）有 5120/128=40 个 N-tile，分属 40 个 CTA。每个 CTA 计算自己 128 列的 x² partial sum，需要合并到 per-row total。atomic add 到 fp32 sum_buffer 是最简单的方式。

对于 8 卡 × num_sms(148) × 40 tiles/row，竞争每行的 atomic 只有 40 个 CTA，且分散到不同 SM，竞争可控。

---

## 6. Kernel 2 详细设计：`sm100_bf16_rmsnorm_a2a_scatter`

### 6.1 整体结构

```cuda
__global__ void sm100_bf16_rmsnorm_a2a_scatter_impl(
    const bf16* local_buffer,      // [bs*local_seq, N] Kernel 1 的输出
    const float* sum_buffer,       // [bs*local_seq, 2] per-row x² sum (Q, K)
    const float* norm_q_weight,    // [dim]
    const float* norm_k_weight,    // [dim]
    float eps, uint32_t dim,
    SymBuffer sym_buffer,
    GemmA2ATransposeScatterMaps scatter_maps,
    uint32_t bs, uint32_t local_seq, uint32_t seq,
    uint32_t n, uint32_t local_n, uint32_t num_ranks, uint32_t rank_idx
) {
    // nvlink_barrier (init: 保证 Kernel 1 的本地写完成)
    
    // persistent tile loop over [num_m_blocks × num_n_blocks]:
    for each tile (m_block, n_block):
        global_m = m_block * BLOCK_M
        n_col = n_block * BLOCK_N
        
        // 1. TMA load from local_buffer[global_m, n_col] → SMEM
        // 2. RMSNorm (if Q/K range)
        // 3. TMA store scatter to scatter_maps[dst_rank]
        
        dst_rank = n_col / local_n
        base_n_idx = n_col % local_n
        b = global_m / local_seq
        s_local = global_m % local_seq
        base_m_idx = b * seq + rank_idx * local_seq + s_local
        
        // RMSNorm:
        if (n_col < dim):  // Q range
            rms = rsqrt(sum_buffer[row, 0] / dim + eps)
            smem_tile = smem_tile * rms * norm_q_weight[col_in_dim]
        elif (n_col < 2*dim):  // K range
            rms = rsqrt(sum_buffer[row, 1] / dim + eps)
            smem_tile = smem_tile * rms * norm_k_weight[col_in_dim - dim]
        // V range: identity
    
    // drain scatter stores
    tma_store_wait<0>();
    
    // nvlink_barrier (final: 保证所有 scatter 全局可见)
}
```

### 6.2 Warp 布局

Kernel 2 更简单（无 GEMM），只有 data movement + elementwise：
- W0: TMA load from local buffer → SMEM
- W1-W3: RMSNorm elementwise (read SMEM → compute → write SMEM)
- W4-W7: TMA store scatter to peer

或更简单：所有 warp 协作 load → norm → store。

### 6.3 RMSNorm 实现

在 SMEM 中做 elementwise RMSNorm：
1. 读 SMEM 中的 bf16 tile
2. 转 fp32
3. 乘 `rms * weight`（rms 从 sum_buffer 读，weight 从全局内存读）
4. 转 bf16 写回 SMEM
5. TMA store 从 SMEM scatter 到 peer

---

## 7. 约束

- `dim % 128 == 0`（GEMM K 对齐 + RMSNorm tile 对齐）
- `3*dim % num_ranks == 0`（head 组均分，含 Q/K/V 三段）
- `head_dim % 8 == 0`
- `seq % num_ranks == 0`
- `(seq / num_ranks) % 128 == 0`（M-tile 对齐）
- `dim % BLOCK_N == 0`（RMSNorm 的 N-tile 不跨 Q/K/V 边界）
  - 当 dim=5120, BLOCK_N=128: 5120/128=40, 恰好整除 ✓

---

## 8. 实现路径

### Phase 1: 分开方案验证正确性（P0）
1. `bf16_gemm_nt(x, Wqkv_t)` → 本地 buffer `d[local_m, 3*dim]`
2. PyTorch RMSNorm: `d[:, :dim] = rmsnorm(...)`, `d[:, dim:2*dim] = rmsnorm(...)`
3. 独立 `bf16_a2a_scatter` kernel → scatter to peer
4. 验证 vs torch reference (matmul + rmsnorm + all_to_all)

### Phase 2: 两 kernel 融合（P2）
1. Kernel 1: `sm100_bf16_gemm_rmsnorm_local`（GEMM + 写本地 + x² sum）
2. Kernel 2: `sm100_bf16_rmsnorm_a2a_scatter`（Norm + Scatter）
3. 替换 Phase 1 的独立组件

### Phase 3: autograd.Function（P1）
- forward = Kernel1 + Kernel2
- backward = 对偶 A2A-gather + Norm-inverse + GEMM

---

## 9. 涉及文件

| 文件 | 改动 |
|------|------|
| `deep_gemm/include/deep_gemm/layout/fused_qkv_norm_a2a.cuh` | 新建：workspace layout |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rmsnorm_local.cuh` | 新建：Kernel 1 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_rmsnorm_a2a_scatter.cuh` | 新建：Kernel 2 |
| `csrc/jit_kernels/impls/sm100_bf16_fused_qkv_norm_a2a.hpp` | 新建：host + JIT |
| `csrc/apis/fused_qkv_norm_a2a.hpp` | 新建：API + register |
| `deep_gemm/fused_qkv_norm_a2a/__init__.py` | 新建：Python 入口 |
| `csrc/python_api.cpp` | 改：include + register |
| `deep_gemm/__init__.py` | 改：暴露符号 |
| `tests/comm/test_fused_qkv_norm_a2a.py` | 新建：正确性测试 |
| `benchmarks/bench_fused_qkv_norm_a2a.py` | 新建：性能 benchmark |

---

## 10. 正确性参考

- **ground truth**：`all_gather(x)` → `D_global = X_global @ Wqkv^T` → split Q/K/V → RMSNorm(Q), RMSNorm(K) → A2A scatter by head → BSHD
- **torch baseline**：`torch.matmul + WanRMSNorm + dist.all_to_all_single`
- **fused**：Kernel1 + Kernel2，输出须同时匹配 ground truth 与 torch baseline
- **容差**：RMSNorm 引入 fp32 计算 + rsqrt，rel_err 应 < 0.02（vs bf16 纯 GEMM 的 0.0）
