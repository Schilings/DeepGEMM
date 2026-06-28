# GEMM-A2A-Transpose 设计（Ulysses SP pre-attn：QKV/Q proj GEMM + All2All-transpose）

> 目标算子：Ulysses SP **pre-attention** 步骤的 **GEMM + All2All-transpose 融合**。
> 入口符号 `bf16_gemm_a2a_transpose_nt`。它是 post-attn `a2a_transpose_gemm`（A2A→Wo GEMM）的
> **对偶/逆向**：pre-attn 先做投影 GEMM，再 A2A-transpose 把「seq 分片 / full head」重排成
> 「head 分片 / full seq」喂给 attention（FlashAttention 的 BSHD/THD 输入）。
>
> **一句话**：本算子 = **GEMM-RS 去掉 reduce、把 dst 切分轴从 M(token) 换成 N(head)、push 目标
> 从「dst 的 slot[my_rank]」换成「dst 输出 buffer 的本 rank seq 偏移」**。GEMM-RS 的 epilogue
> push-scatter（`scatter_maps` + `sm100_store_cd` 的 2D TMA store 直推 peer）原样复用。

---

## 1. 场景与数据流（Ulysses SP，pre-attention）

```
输入（每 rank: 自己的 seq 分片、full hidden）
   x : [bs, local_seq, hidden]                local_seq = seq / sp_size,  hidden = 模型维(K)
        │  QKV/Q proj GEMM  (A=x[M,K], B=Wqkv[N,K], D=[M,N])
        │     M = bs*local_seq,  K = hidden,  N = nheads*head_dim
        ▼
   D : [bs*local_seq, nheads, head_dim]        (本 rank seq 分片的「全部 head」投影结果)
        │  All2All-transpose（按 head 散射 + seq↔head 转置）
        ▼
   y : [bs, seq, local_nheads, head_dim]       (head 分片、full seq；BSHD，喂 FlashAttention)
        local_nheads = nheads / sp_size,  local_N = local_nheads*head_dim = N/sp_size
```

**核心**：本 rank `r` 只投影自己的 seq 分片（M=bs*local_seq），得到「这些 token 的全部 head」。
A2A-transpose 把 head 维按 dst_rank 散射（dst_rank 拿 head 组 `[dst*local_nheads:(dst+1)*local_nheads]`），
同时把本 rank 的 seq 分片写到 dst 输出的 seq 偏移 `r*local_seq`。每个目标位置**恰好被一个 src 写一次**
（纯排列），所以**不需要 reduce、不需要清零**。

### 转置索引契约（实现必须严格遵守）

本 rank `r` 的 GEMM 输出 `D[global_m, n]`，`global_m ∈ [0, bs*local_seq)`，`n ∈ [0, N)`：
- `b = global_m / local_seq`，`s_local = global_m % local_seq`（要求 `local_seq % BLOCK_M == 0`，
  保证一个 M-tile 不跨 batch）。
- `dst_rank = n / local_N`，`n_local = n % local_N`（要求 `local_N % BLOCK_N == 0`，
  保证一个 N-tile 的列只属于一个 dst）。
- 写入 dst_rank 输出 buffer（视为 2D `[bs*seq, local_N]`，行主、stride=local_N）的位置：

```
out_dst[ b*seq + r*local_seq + s_local , n_local ] = D[global_m, n]
```

即「本 rank 的 seq 分片」落在 dst 输出的行偏移 `r*local_seq`（per-batch），
「dst 的 head 组」落在 dst 输出的列段 `n_local`。out_dst 即 BSHD `[bs, seq, local_nheads, head_dim]`。

---

## 2. 架构（单 kernel，抄 GEMM-RS）

GEMM-RS 是 dual-kernel（GEMM push-scatter + flagless RS reduce），因为 RS 需要把各 rank 的 partial
**累加**。pre-attn 的 A2A 是**纯排列、无累加**，所以 **reduce kernel 直接删掉** → **单 kernel**：

```
单 kernel sm100_bf16_gemm_a2a_transpose_impl（256T = 8 warp，与 GEMM-RS 同构）:
  起始 nvlink_barrier  ── 保证上一轮所有 peer 对「我的 buffer」的读取已完成，本轮才覆盖写
  W0: TMA Load A+B
  W1: MMA Issue
  W2: Reserved / TMEM Allocator
  W4-W7: Epilogue —— 按 (m_block,n_block) 算 dst_rank/base_m_dst/base_n_dst，
                      用 scatter_maps.maps[dst_rank] 经 2D TMA store 直推 dst 输出 buffer
  ptx::tma_store_wait<0>  ── drain 所有 push
  结束 nvlink_barrier  ── 保证所有 peer 对「我的 buffer」的 push 已全局可见，之后我才读
```

- **A = x**（本 rank 输入，普通 tensor，非 symm）：`[M=bs*local_seq, K=hidden]`，TMA 2D load。
- **B = Wqkv**（普通 tensor，NT）：`[N, K]`。
- **scatter 目标 = dst 的 symm 输出 buffer**（无 per-rank slot；各 rank 写不同 seq 行，互不重叠）。
- GEMM 的 tile 调度用**标准 persistent raster**（[M×N] 网格，支持 mc=2），不需要 GEMM-RS 的
  「self-chunk-last 环形」调度（那是为了 reduce 的本地化收尾）。

### 与 GEMM-RS 的差异速查

| 维度 | GEMM-RS | 本 GEMM-A2A-Transpose |
|------|---------|----------------------|
| dst 切分轴 | M（token chunk）：`dst = m_block / num_m_blocks_per_rank` | N（head 组）：`dst = (n_block*BLOCK_N) / local_N` |
| GEMM 的 M | `total_m`（每 rank 算完整 M） | `bs*local_seq`（每 rank 只算自己 seq 分片） |
| push 目标 | dst 的 `slot[my_rank]`（2D `[m_per_rank, n]`） | dst 输出 buffer（2D `[bs*seq, local_N]`） |
| push 坐标 | `(local_m, n)` | `(b*seq+r*local_seq+s_local, n%local_N)` |
| 第二 kernel | RS reduce（FP32 累加 num_ranks 份） | **无**（A2A 纯排列，不累加、不清零） |
| 输出 | reduce 写 user `y[m_per_rank, n]` | symm buffer 本身即输出 `[bs,seq,local_nheads,head_dim]` |
| tile 调度 | self-chunk-last 环形 | 标准 raster |

---

## 3. Symm buffer / barrier 布局（`layout/gemm_a2a_transpose.cuh`）

```
[0 .. 32)            barrier/signal 区（kNumBarrierSignalBytes=32，与 GemmRSWorkspace 同结构）
                       - grid_sync_count (idx0)
                       - nvl_barrier_counter (idx4)
                       - nvl_barrier_signal[2] (idx5,6)
[32 .. 32+OUT)       输出区 out: [bs*seq, local_N]，elem_size (bf16=2/fp32=4)
                       OUT = bs*seq*local_N*elem_size
总字节 align(32+OUT, 16)
```

- scatter_maps.maps[d]：2D TMA descriptor，base = `sym_buffer_ptrs[d] + 32`，
  gmem_inner=local_N，gmem_outer=bs*seq，stride=local_N，swizzle=128（与 epilogue STSM 一致）。
  d==rank → 本地写；d!=rank → P2P 直推 peer HBM。
- 输出区**无需清零**（每位置恰被写一次）。barrier 区在 buffer 创建时 `zero_()` 一次即可，
  靠 nvlink_barrier 的 +1/-1 自复位协议跨调用平衡（与 GEMM-RS 一致，禁止 per-call memset）。

---

## 4. 约束（host assert）

- `nheads % num_ranks == 0`，`seq % num_ranks == 0`。
- `local_seq = seq/num_ranks`，`local_seq % BLOCK_M(128) == 0`（tile 不跨 batch / seq 边界）。
- `local_N = (nheads/num_ranks)*head_dim`，`local_N % BLOCK_N(128) == 0`
  （head_dim=128 时 local_nheads≥1 即满足；head_dim=64 需 local_nheads 偶数）。
- `N = nheads*head_dim`，`N % 128 == 0`，`K = hidden`，`K % 64 == 0`。
- `bs >= 1`（THD = bs=1 的 BSHD，天然覆盖；varlen packed 同 post-attn 结论：uniform 切即可）。

---

## 5. 涉及文件

| 文件 | 改动 |
|------|------|
| `deep_gemm/include/deep_gemm/layout/gemm_a2a_transpose.cuh` | 新建：输出区 `[bs*seq, local_N]` + barrier 区 |
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_a2a_transpose.cuh` | 新建：GEMM + epilogue 转置 push-scatter（抄 gemm_rs kernel）|
| `csrc/jit_kernels/impls/sm100_bf16_gemm_a2a_transpose.hpp` | 新建：单 kernel host + per-dst scatter map（复用 gemm_rs_compute 启发式）|
| `csrc/apis/gemm_a2a_transpose.hpp` | 新建：get_symm_buffer_size + 入口 + register_apis |
| `deep_gemm/gemm_a2a_transpose/__init__.py` | 新建：SymmBuffer 类 + 入口（输出为 buffer 视图）|
| `csrc/python_api.cpp` | 改：include + register |
| `deep_gemm/__init__.py` | 改：暴露符号 |
| `tests/test_gemm_a2a_transpose.py` | 新建：all_gather ground truth + torch all_to_all baseline 交叉 |
| `benchmarks/bench_gemm_a2a_transpose.py` | 新建：torch / sep(gemm+all_to_all) / fused 三路 |
| `docs/GEMM_A2A_TRANSPOSE_ITERATION.md` | 新建：迭代记录 + 当前状态 |
| `docs/PROGRESS.md` / `docs/RULE.md` | 改：算子表新增一行 |

---

## 6. 正确性参考（test）

- **ground truth（非循环）**：`all_gather(x)` 沿 seq 拼出 `X_global[bs, seq, hidden]` →
  `D_global = X_global.reshape(bs*seq,K) @ Wqkv.t()` → reshape `[bs, seq, nheads, head_dim]` →
  取本 rank head 组 `[:, :, r*local_nheads:(r+1)*local_nheads, :]` → BSHD `[bs, seq, local_nheads, head_dim]`。
- **torch 原生 baseline（交叉对照）**：本地 GEMM + `dist.all_to_all`（按 dst 切 head、按 src 拼 seq）。
- fused kernel 输出（symm buffer 视图）须同时匹配 ground truth 与 torch baseline。

## 7. 里程碑

1. M0 正确性：单 kernel 跑通 transpose 语义 `{2,4,8}` 卡全 PASS（rel≈1e-6 ~ bf16 量级）。
2. bench：vs torch-native（matmul+all_to_all）/ vs deepgemm-separate（bf16_gemm_nt + all_to_all），
   记录 4/8 卡 focus shape。
3. 调优（可选）：通信旋转、mc=2、raster——但 push 直接吸收在 GEMM epilogue，单 kernel 已无 comm
   kernel 抠 SM 的代价，预期天然优于 separate。
