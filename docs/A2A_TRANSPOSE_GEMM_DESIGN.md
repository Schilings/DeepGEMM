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
- **M1（overlap，已实现 + 已评测 + 已对齐 flux 调优）**：
  - 实现：comm 改 tile 粒度 + per-M-tile barrier（计数到 world_size，置 1）；融合 GEMM 消费者
    （`impls/sm100_bf16_a2a_transpose_gemm.cuh`）Load-A warp 等 `barrier[m_block]==1` 后再 load；
    host（`csrc/sm100_bf16_a2a_transpose_gemm.hpp`）comm 走 comm_stream + GEMM 走 compute_stream，
    **SM carveout**（`DG_A2AT_COMM_SMS`，默认 24）防 GEMM 饿死 comm 死锁。
  - **正确性**：`tests/test_a2a_transpose_gemm.py {2,4,8}` 全 4/4 PASS（融合路径）。

  - ⚠️ **首版评测曾误判为「0.4~0.5x，单节点零和墙」——那是两个实现 bug 造成的假象，已修正：**
    1. **comm block 只有 256 线程**：低 SM carveout 下，单 SM 的 NVLink 带宽由「在途 P2P store 数」决定，
       256 线程远喂不满（要数千个 uint4 在途才能隐藏 NVLink 延迟）。实测大 shape `comm@16SM`：
       **256 线程=491us，512=267us，1024=152us（3.2×）**。→ comm 线程数提到 **1024**（host 默认 + env）。
    2. **GEMM 自旋用 `ld.acquire.sys` 死循环**：所有 GEMM CTA（comm_sms=16 时达 116 个）每轮都发
       system-scope acquire load，**轰炸内存 fabric、反过来饿死 comm 的 P2P store**。→ 改为
       **relaxed.sys 轮询 + `__nanosleep` 指数退避 + 命中后单次 `fence.acquire.sys`**（`ptx/ld_st.cuh`）。
    - 两项修复后大 shape `(8,56,4096,128,7168)` fused **731→450us（1.6×）**，多数 shape 从 0.4~0.5x 升到 ~1.0x。

  - **comm 带宽 vs SM 饱和曲线（8 卡，大 shape，1024 线程）**：comm 要 ~64 SM 才接近打满 NVLink；
    `8SM=447us / 16=152 / 32~130 / 64=125 / 132=116`（≈440GB/s 上限）。

  - **性能（8 卡，公平基线 = comm@全SM + gemm@全SM 串行；fused = 1024线程 + relaxed自旋 + comm_sms=24）**：

    | shape (bs,nh,seq,hd,N) | comm us | gemm us | serial us | fused us | fused/serial |
    |------|------|------|------|------|------|
    | (1,32,2048,128,4096) | 20 | 30 | 50 | 49 | 1.01x |
    | (1,56,2048,128,7168) | 22 | 45 | 67 | 72 | 0.93x |
    | (8,32,2048,128,4096) | 40 | 64 | 103 | 113 | 0.92x |
    | (4,32,8192,128,4096) | 70 | 100 | 170 | 173 | 0.98x |
    | (1,32,16384,128,4096) | 39 | 62 | 101 | 114 | 0.89x |
    | (8,56,4096,128,7168) | 112 | 251 | 363 | 452 | 0.80x |

  - **修正后的 Verdict（诚实）**：**不是零和墙、也不是 0.4x 灾难——是「~parity」**。对齐 flux 调优后，
    融合 overlap 机制确实生效（comm 被实质藏住），但在**本单节点**上 fused 仍 **0.8~1.0x（偶尔持平）、
    尚未稳定超过串行**。物理本质：comm 要带宽=要 SM，从 GEMM 抠走 ~24 SM 的代价 ≈ 它藏住的 comm，
    所以单节点接近持平。**结论：单节点 M0（串行、两段各占满 SM）仍是最快、最稳的选择；M1 现在是一个
    正确且经过 flux 式调优的 overlap 实现，真正的净收益要在 comm«gemm（大算力）或多节点场景体现。**

### 为什么 flux 报告「有加速」而我们说「~parity」（基线口径差异，关键）

> flux 的加速口径见 `flux/test/python/gemm_a2a_transpose/test_gemm_a2a_transpose.py`：

- flux 的对照基线 `perf_torch` = **`torch.matmul`（cublas）+ `torch_pre_attn_all_to_all_transpose`**，
  而后者（`flux/python/flux/testing/ulysses_sp_utils.py`）做通信的方式是
  **`.permute(...).contiguous()`（一趟完整 HBM 转置）+ NCCL `all_to_all_single` + 再 `.permute().reshape()`**，
  整体**串行**。flux 拿「fused 总时间」去比这个弱基线，自然大幅领先。
- **本机实测对照（8 卡）**，证明 flux 的领先主要来自「单趟融合 comm kernel 远快于 NCCL+torch 转置」，
  而不是 overlap 本身：

  | shape | torch comm(NCCL+2转置) | 我们 comm(单趟) | torch 总(串行) | **我们 M0 串行** |
  |------|------|------|------|------|
  | (8,32,2048,128,4096) | 106.7us | 25.1us（**4.3×**）| 149.9us | **73.7us（1.6×）** |
  | (4,32,8192,128,4096) | 165.9us | 47.5us（**3.5×**）| 248.7us | **136.4us（1.8×）** |
  | (8,56,4096,128,7168) | 268.2us | 79.4us（**3.4×**）| 510.4us | **318.3us（1.6×）** |

- **结论**：flux 的「主要收益」= 用单趟 GPU kernel 同时完成转置+all2all（避开 NCCL + 两趟 torch 转置），
  这一点**我们的 M0 已经完全做到**（comm 比 torch 基线快 3~4×，M0 总时间比 torch 基线快 1.6~1.8×）。
  flux 在此之上再叠加 overlap；而 overlap 单节点 ~parity（见上）。所以**「为什么 flux 能」的答案不是
  它有魔法 overlap，而是它的基线是 NCCL+torch；换成强基线（我们的 M0/优化 comm），结论同样回到 ~parity。**
  我们没有落后 flux——M0 即达到 flux 级的 comm 质量。

### 资源利用率分析（B300 SXM6，本机）

> 针对「B 卡 SM 更多、NVLink 更快，是不是资源没吃满」做的核查。

- **硬件**：B300 SXM6，**148 SM**，287GB，8 卡全 NVSwitch（NV18，NVLink5 ≈ 900 GB/s/方向/卡）。
- **SM 利用**：comm 与 GEMM 都已用满可用 SM（fused 用全部 148：comm carveout 24 + GEMM 124）。**不存在
  空闲 SM**，瓶颈不在「SM 没用上」。
- **通信旋转（ring/shifted all-to-all，已采用）**：原 dst-major 无旋转时**所有 rank 同时先打 dst0、再 dst1…**
  → 任意时刻只有一个 GPU 的 NVLink ingress 在收（~1/R bisection）= 热点。改为**按 rank 旋转
  `dst=(rank+step)%R`**（每个 step 是一个 permutation，rank r→dst r+s）后所有 ingress 链路同时忙，
  **comm 大 shape 112→94us（≈459→545 GB/s，+16%），M0/serial 363→343us**。
  - ⚠️ **旋转与 fused overlap 冲突**：旋转把「推给我的」贡献分散到所有 step（peer p 在 step=(my_rank−p)%R
    才推我）→ 我的 tile 整段 comm 末尾才补齐 → GEMM 无法 overlap（fused 451→533）。**带宽最优（打散）与
    overlap 最优（早集中到齐）天然冲突**。故**只在 M0/standalone comm 旋转（带宽优先、无消费者），
    fused（`kSetBarrier`）保持不旋转**（早集中到齐、利于 overlap）。
- **NVLink 利用（关键）**：comm 大 shape egress = 51.4MB/rank，旋转后 `comm@全SM=94us → ≈545 GB/s`
  （未旋转 459=50%峰值）。剩余差距是**散射+转置 P2P 写的访问模式天花板，不是 MLP/线程不足**：
  - 全 SM 下 comm 线程 256/512/1024 时间几乎不变（122/111/111us）→ 已饱和该模式下的写带宽。
  - 手动 4× unroll（load/store 解耦提 MLP）对带宽**中性**。
  - 对照：copy-engine **连续** P2P 单链路能到 **~667 GB/s** → 差距来自「每行仅 local_hidden≈1792B 连续、
    跨行 stride=hidden」的散射写 + all-to-all，而非硬件没力气。
- **能不能更快**：理论上 pull（远端读）或把转置折进 GEMM 的 TMA A-load（让网络传输变连续）可能逼近峰值，
  但前者需重写 comm、后者 TMA 无法任意 gather，均为独立大工程，**未做**。
- **结论**：单节点的限制是 **comm 的 NVLink 写带宽（且 comm 带宽=SM 数）**，不是闲置算力。因此 fused
  在单节点 ≈ parity 是该模式 + 单节点拓扑的固有结果，B300 更多 SM 改变不了「comm 抠走的 SM ≈ 它藏住的量」。

- **试过但回退的（记录避免重复踩坑）**：
  - comm work 改 **tile-major 排序**（tile 外层、dst 内层，对齐 flux `get_tile_info`）：让所有 rank 的 tile
    渐进 ready，**只救活 comm-heavy 大 shape（fused 452→407us，0.80→0.85x）**，却让中等 shape 回退
    （4,32,8192：0.98→0.72x），**平均 0.84x < dst-major 0.92x**，故**保留 dst-major**。根因：bench 同步取
    最慢 rank，dst-major 下高 rank 的 tile 全在 comm 末尾才 ready → GEMM 几乎不 overlap（先空转再算）；
    tile-major 修了这点但牺牲了其它 shape 的 comm 局部性。
  - comm copy 循环 **4× unroll**（提 MLP）：带宽中性，回退保持简洁。
  - **pull 模式 comm**（远端读代替远端写：每 rank 读所有 peer 的 input 填自己的 gathered，barrier 退化为
    纯本地 gpu-scope，无跨卡原子）：本想远端读更易打满 NVLink，但**实测 B300 上 pull 全面慢于 push**
    （大 shape 114→152us、(1,56,2048) 22→43us），即 remote write > remote read，故回退。
- **M2（可选，未做）**：仅在 comm « gemm 的大算力 shape 或多节点场景下重启 overlap 调优；否则保持 M0。

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
