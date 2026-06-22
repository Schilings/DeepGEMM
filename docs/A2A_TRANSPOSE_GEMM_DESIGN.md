# A2A-Transpose-GEMM 设计（Ulysses SP post-attn：All2All-transpose + Wo GEMM）

> 目标算子：把当前语义错位的 `bf16_a2a_gemm_nt`（token(M)-A2A）重写为 **Ulysses SP post-attention**
> 真正需要的 **All2All-transpose（沿 hidden/K 收集）+ Wo GEMM**，方法学对齐 flux
> `src/a2a_transpose_gemm`，在 SM100（B 卡）上落地。
> 背景与「为什么旧实现不对」见 `A2A_GEMM_ITERATION.md` 顶部「当前状态」。

## 实现进度（分支 `a2a-transpose-gemm`）

- **M0（正确性，已完成）**：转置 scatter comm kernel（`impls/sm100_a2a_transpose_comm.cuh`，uint4 P2P，
  写 dst 的 hidden 列偏移 `rank*local_hidden`）+ 新 layout（`layout/bf16_a2a_transpose_gemm.cuh`：
  barrier/signal + input + gathered 三区）+ host/api/python（`a2a_transpose_gemm`）。M0 用
  `comm → SP-group barrier → 标准 bf16_gemm_nt`（无 overlap）。
  **正确性 `tests/test_a2a_transpose_gemm.py {2,4,8}` 全 4/4 PASS**（vs all_gather 重建的非循环
  ground-truth，rel~1e-6；torch 候选 rel=0）。入口：`deep_gemm.bf16_a2a_transpose_gemm_nt(d, Wo, sym)`。
- **M1（overlap，进行中）**：per-M-tile barrier（comm 计数到 world_size）+ GEMM 消费者按 tile-ready
  消费（替换 M0 的 barrier+标准 GEMM）；bench vs separate。
- **M2（调优）**：comm SM 数 / TILE_M / raster / mc=2。

---

## 1. 场景与数据流（Ulysses SP，post-attention）

```
Attention 输出（每 rank: 本 rank 的 head 子集、全 seq）
   x : [bs, local_nheads, seq, head_dim]          local_nheads = nheads / sp_size
        │  All2All-transpose（沿 hidden 收集 + seq↔head 转置）
        ▼
   x': [bs, local_seq, nheads, head_dim] = [bs, local_seq, hidden]   local_seq = seq / sp_size
        │  Wo GEMM   (A=x' [M, K],  B=Wo [N, K],  D=[M, N])
        ▼
   y : [bs, local_seq, N]                          M = bs*local_seq,  K = hidden = nheads*head_dim
```

**核心**：x' 中本 rank 自己 seq 分片的每个 token 行，其完整 hidden(K) 由**所有 sp_size 个 rank** 各贡献
一段 `local_hidden = local_nheads*head_dim` 拼成。**GEMM 的一个 M-tile 必须等所有 rank 的 hidden 切片
到齐才能算完整-K**——这与旧 token-A2A（每行来自单个 rank、K 已完整）本质不同。

### 转置索引契约（实现必须严格遵守，来自 flux `push_tile_to_dst`）

对本 rank（=`r`）输入 `x[bs, local_nheads, seq, head_dim]`：对每个全局 token `global_seq ∈ [0, seq)`：
- `dst_rank = global_seq / local_seq`，`dst_seq = global_seq % local_seq`
- 把本 rank 的全部 `local_nheads` 个 head 写到 `dst_rank` 输出 buffer 的 head 段 `[r*local_nheads : (r+1)*local_nheads]`：

```
out_dst[bs, dst_seq, (r*local_nheads + nh), hd] = x[bs, nh, global_seq, hd]
        ( nh ∈ [0,local_nheads), hd ∈ [0,head_dim) )
```

即「本 rank 的 hidden 切片」落在 dst 的 hidden 列偏移 `r*local_hidden`。out 视为 `[bs, local_seq, hidden]`。

---

## 2. 架构（对齐 flux，映射到 SM100；与本仓 AG-GEMM 同构）

flux 的 post-attn-a2a-transpose-GEMM = **comm kernel（产 A）+ 独立 GEMM kernel（按 M-tile-ready 消费 A）**，
这与本仓 **AG-GEMM**（`sm100_bf16_ag_gemm`：comm 产 `slot_x` + GEMM poll 每 chunk flag 后 load A）结构一致。
因此本设计**复用 AG/a2a 的 SM100 GEMM 消费者骨架**，只改通信与 flag 语义。

### 2.1 通信：A2A-transpose comm kernel（comm stream）

- 端口移植 flux `post_attn_all2all_transpose_kernel` / `push_tile_to_dst`（cuda core、`uint4` 向量化、按
  `TILE_M` 分块）。每个 comm CTA：把本 rank 的某 dst 的 seq 分片，按上面的转置索引写入 **dst 的 symm
  comm buffer 的 hidden 列偏移 `r*local_hidden`**（self 即本地写）。
- **per-M-tile barrier**：每写完一个 M-tile，`atomicAdd_system(barrier[dst][tile], -1)`；当计数到
  `-world_size*block_per_tile` 时 `st.release.sys` 置该 tile barrier=1（= 所有 rank 的 hidden 切片到齐）。
- comm kernel 启动后由 bid0/thread0 置 `a2a_signal=1`（供 GEMM stream 的 `cuStreamWaitValue` 解除）。

### 2.2 计算：SM100 GEMM 消费者（compute stream）

- 复用现有 `sm100_bf16_a2a_gemm.cuh`（或新建 `sm100_bf16_a2a_transpose_gemm.cuh`）的 GEMM 骨架：
  Load A（poll flag → TMA load A）/ Load B / MMA / epilogue 2D store。
- **改 flag 语义**：A 的一个 M-tile 在 load 前 `wait_eq(barrier[m_tile]==1)`（**单 flag/​M-tile、计数到
  world_size**），而不是旧的「per-(src_rank,chunk) 等单个 src」。
- **A buffer = 本地 symm comm buffer**，in-place 读，视为 `[M=bs*local_seq, K=hidden]`（行主、K 连续）。
- **输出 M = bs*local_seq**（本 rank 只算自己 seq 分片），不再是 `num_ranks*M_per_rank`。
- B = Wo `[N, K]`（NT），D = `[M, N]`。

### 2.3 编排（host）

- 主 stream 记 `ready_event`；`compute_stream` 等它。
- comm kernel 发到主 stream（写 peer buffer + barrier + a2a_signal）。
- GEMM 发到 `compute_stream`：先 `cuStreamWaitValue(compute_stream, a2a_signal, 1)`，再 launch（内部 per-tile
  barrier wait）。
- GEMM 后记 `compute_event`，主 stream 等它；reset barrier。
- 也可用 PDL/单 stream 变体，但首版对齐 flux 双 stream + 高优先级 compute_stream 最稳。

---

## 3. Symm buffer / barrier 布局

- **comm 输出 buffer**（symm，per rank）：`[bs, local_seq, hidden]`（= flux 的 comm_output_buffer，已含全
  hidden）。GEMM 直接把它 reshape 成 `[M, K]` 作为 A。
- **barrier buffer**（symm，int32）：`bs * ceil(local_seq / TILE_M) + 1` 个（每 M-tile 一个 + 末位 a2a_signal）。
  初值 0；comm 写完置 1；每次调用后 reset。
- 复用 `layout/sym_buffer.cuh` 的 P2P 映射；新建 `layout/bf16_a2a_transpose_gemm.cuh`（替代沿 M 的
  `bf16_a2a_gemm.cuh` 布局：去掉 `slots[num_ranks]` 沿 M 拼接，改为单个 `[bs,local_seq,hidden]` 收集 buffer +
  per-M-tile barrier）。

---

## 4. 正确性参考（test 必须改）

旧 test 用 `dist.all_to_all(行块)+cat(dim=0)+@b.t()`（token-A2A）——**必须换成 transpose 语义**：

```python
# x: [bs, local_nheads, seq, head_dim]（每 rank 不同）
# 参考 A2A-transpose：用 all_to_all 在 hidden 维收集 + 转置，得到本 rank 的 [bs, local_seq, nheads, head_dim]
#   等价实现：把 x 按 dst_rank 切 seq → all_to_all → 在 head 维按 src_rank 拼接 → reshape[bs,local_seq,hidden]
xt = ulysses_post_attn_a2a_transpose(x, sp_group)        # [bs, local_seq, hidden]
d_ref = xt.reshape(bs*local_seq, hidden) @ Wo.t()        # [bs*local_seq, N]
```

并保留跨 rank 一致性校验。shape 集合用 Ulysses 真实量级（`local_seq×N×hidden`，hidden=nheads*head_dim）。

---

## 5. 涉及文件（预估）

| 文件 | 改动 |
|------|------|
| `deep_gemm/include/deep_gemm/impls/sm100_bf16_a2a_transpose_gemm.cuh` | 新建：GEMM 消费者（复用 a2a/ag 骨架，改 per-M-tile full-K barrier、A=comm buffer、M=local_seq）|
| `deep_gemm/include/deep_gemm/impls/a2a_transpose_comm.cuh`（或 host CE+小kernel）| 新建：转置 scatter comm kernel（移植 flux push_tile_to_dst）|
| `deep_gemm/include/deep_gemm/layout/bf16_a2a_transpose_gemm.cuh` | 新建：comm buffer `[bs,local_seq,hidden]` + per-M-tile barrier 布局 |
| `csrc/jit_kernels/impls/sm100_bf16_a2a_transpose_gemm.hpp` | 新建：双 stream 编排（comm + GEMM + a2a_signal + event）|
| `deep_gemm/a2a_gemm/__init__.py`（或新 `a2a_transpose_gemm/`）| 新接口：输入 `[bs,local_nheads,seq,hd]`、Wo、sp_group |
| `tests/test_a2a_transpose_gemm.py` | 新建：transpose 语义参考 + 跨 rank 一致性 |
| `benchmarks/bench_a2a_transpose_gemm.py` | 新建：vs (a2a-transpose + 标准 GEMM) separate 基线 |

> 是否保留旧 `bf16_a2a_gemm_nt`（token-A2A）：建议旧实现以 tag 存档或标注「非 Ulysses」，主推 transpose 版。

---

## 6. 风险与里程碑

- **风险**：转置索引/列偏移写错（用单测逐元素校验先行）；per-M-tile barrier 的 system-scope 原子在 B 卡
  P2P 的可见性/性能；GEMM 等 full-K 削弱 overlap（只能靠不同 M-tile 先后 ready 的流水）。
- **里程碑**：
  1. M0 正确性优先：comm kernel + GEMM（先可不追 overlap，甚至 comm 完成后再 GEMM）跑通 transpose 语义 `{2,4,8}` 卡 6/6。
  2. M1 overlap：双 stream + per-M-tile barrier，bench vs separate。
  3. M2 调优：comm SM 数、TILE_M、raster、mc=2。

---

## 7. 关键差异速查（旧 token-A2A vs 本设计）

| 维度 | 旧 `bf16_a2a_gemm_nt` | 本 A2A-transpose-GEMM |
|------|------|------|
| 通信轴 | M（token 整块）| K（hidden 切片）+ seq↔head 转置 |
| A 来源 | num_ranks 个 slot 沿 M 拼 | 单 comm buffer `[bs,local_seq,hidden]` |
| flag | per-(src,chunk)，等单个 src | per-M-tile，等所有 rank（full-K）|
| 输出 M | num_ranks*M_per_rank | bs*local_seq |
| 适配 Ulysses post-attn | 否 | 是 |
