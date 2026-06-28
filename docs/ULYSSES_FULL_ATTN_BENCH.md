# Ulysses 完整 Attention 链路 Benchmark 与 bf16 误差分析

机器：NVIDIA B300 SXM6 AC ×8。脚本：`benchmarks/bench_ulysses_full_attn_flow.py`。
端到端测试见 `tests/ulysses/test_ulysses_full_attn_flow.py`。
attention 统一使用 **FlashAttention-4**（安装见 `docs/INSTALL_FA4.md`，封装于 `tests/ulysses/fa4_attn.py`）。

---

## 1. 测什么

Ulysses SP 单 rank 的完整 attention 链路（序列并行输入）：

```
X_local[bs, local_seq, hidden]
  --[PRE]  融合 QKV-proj GEMM + A2A-transpose  --> q,k,v [bs, seq, local_nh, hd]   (OUR op #1)
  --[ATTN] attention (FlashAttention-4)         --> attn  [bs, local_nh, seq, hd]
  --[POST] A2A-transpose + Wo GEMM（comm/计算重叠）--> y    [bs*local_seq, N]         (OUR op #2)
```

每个 shape、每种输入布局都对比三条链路：

| 链路 | PRE | ATTN | POST |
|---|---|---|---|
| **fused（ours）** | `bf16_gemm_a2a_transpose_nt`（单 kernel epilogue scatter 重叠） | FlashAttention-4 | `bf16_a2a_transpose_gemm_nt_fused`（单 kernel 重叠） |
| **torch-native（串行）** | `torch.matmul` + `all_to_all`（不重叠） | FlashAttention-4 | `all_to_all` + `torch.matmul`（不重叠） |
| **async-Ulysses（手工重叠）** | 拆 Q/K/V → 3×GEMM + 3×A2A，多 stream 流水线重叠 | FlashAttention-4 | token 分块 → 逐块 (scatter+A2A) 与 Wo GEMM 多 stream 重叠 |

**async-Ulysses** 是比串行 torch-native 更强的对照：用 ≥2 条 CUDA stream 在**算子外**手工编排计算-通信重叠
（PRE：Q 的 GEMM 算完即在 comm stream 发 Q 的 A2A，同时 comp stream 算 K，依此类推；POST：token 切块，
块 A2A 与已到达块的 Wo GEMM 重叠）。我们的融合算子则是在**单 kernel 内**用 epilogue scatter 做重叠，
因此 `async → ours` 的差距单独刻画了**相对手工多 stream 重叠的额外收益**。

ATTN 三条链路完全相同（本工作不优化 attention），所以对两个 baseline 各报告两个口径：

- **e2e speedup** = `(torch | async)链路 / fused链路`（含 attention 的诚实全链路加速）
- **comm+GEMM speedup** = `(PRE+POST)_(torch | async) / (PRE+POST)_ours`（融合算子真正起作用的部分）

> 表格列约定：时间 `us = ours/torch/async`；加速比 `= vs_torch/vs_async (x)`。

---

## 2. 输入 shape 与权重 shape（正常训练场景：权重均为方阵）

约定 `hidden = nheads * head_dim`，且 **Wo 输出宽度 N = hidden**（方阵）。

### 权重 shape（方阵）

| 权重 | shape | 说明 |
|---|---|---|
| `Wq` / `Wk` / `Wv` | `[hidden, hidden]` | 各自方阵 |
| 融合 `Wqkv` | `[3*hidden, hidden]` | 3 个方阵块按 rank-major 堆叠 |
| `Wo` | `[hidden, hidden]` | 方阵，`N = hidden` |

### 输入 shape

| `(bs, nheads, seq, head_dim)` | `hidden = N` | 权重（方阵） |
|---|---:|---|
| `(1, 32, 4096, 128)` | 4096 | `[4096,4096]`，`Wqkv [12288,4096]` |
| `(1, 56, 4096, 128)` | 7168 | `[7168,7168]`，`Wqkv [21504,7168]` |
| `(2, 32, 4096, 128)` | 4096 | `[4096,4096]`，`Wqkv [12288,4096]` |
| `(2, 56, 2048, 128)` | 7168 | `[7168,7168]`，`Wqkv [21504,7168]` |
| `(1, 64, 8192, 128)` | 8192 | `[8192,8192]`，`Wqkv [24576,8192]` |

---

## 3. BSHD vs THD —— 等价前提

同一个 shape `(bs, nheads, seq, hd)` 按两种布局各跑一遍：

- **BSHD**：`bs` 条长度 `seq` 的序列，批量排布 → tokens = `bs*seq`。
- **THD**：把**同样**的 `bs` 条序列打包成单一 token 流 `T = bs*seq`（`bs'=1, seq'=T`）。

融合 comm/GEMM 算子两种布局处理的 token 总数完全相同（`bs*seq`），调用方式也相同（仅 symm-buffer
的 `(bs, seq)` 描述符不同），因此**加速比必然一致** —— 这正是“等价前提”要验证的：算子对 BSHD / THD
一视同仁。等长序列下 attention FLOPs 也完全相同，故 ATTN 每个 shape 只计时一次、两布局共用。

> THD 真正的额外收益（变长序列免 padding）属于 attention 侧，不是这两个 comm/GEMM 算子的加速来源，
> 故等价测试用等长序列即可证明算子层面的等价性。

---

## 4. 结果

时间单位 us（每算子 20 iters、逐 iter event 计时 + 跨 rank barrier）。

> 注：下方表格为**加入 async-Ulysses 基线之前**的实测，仅含 `ours/torch` 与 vs_torch 加速比。
> 脚本现已额外输出 async-Ulysses 一列（`ours/torch/async` + `vs_torch/vs_async`），待在 B300×8 重跑后补全
> async 对照数字与 geo_mean。预期 async 比串行 torch 更快，因此 `vs_async` 加速比会略低于 `vs_torch`，
> 用以体现融合算子相对**手工多 stream 重叠**的净增益。

### 8 GPUs（ATTN = FlashAttention-4）

| `(bs,nh,seq,hd)` hidden | 布局 | PRE ours/torch | ATTN | POST ours/torch | e2e ours/torch | **e2e** | **c+g** |
|---|---|---|---:|---|---|---:|---:|
| (1,32,4096,128) 4096 | BSHD | 94/146 | 115 | 72/125 | 281/386 | **1.37x** | 1.63x |
| (1,32,4096,128) 4096 | THD | 93/137 | 115 | 63/141 | 271/393 | **1.45x** | 1.78x |
| (1,56,4096,128) 7168 | BSHD | 186/233 | 116 | 96/130 | 397/478 | **1.21x** | 1.29x |
| (1,56,4096,128) 7168 | THD | 183/243 | 116 | 98/132 | 396/490 | **1.24x** | 1.33x |
| (2,32,4096,128) 4096 | BSHD | 132/195 | 115 | 89/117 | 337/427 | **1.27x** | 1.41x |
| (2,32,4096,128) 4096 | THD | 157/193 | 115 | 82/116 | 354/424 | **1.20x** | 1.29x |
| (2,56,2048,128) 7168 | BSHD | 180/238 | 93 | 96/162 | 369/493 | **1.34x** | 1.45x |
| (2,56,2048,128) 7168 | THD | 181/232 | 93 | 99/130 | 372/455 | **1.22x** | 1.29x |
| (1,64,8192,128) 8192 | BSHD | 374/452 | 227 | 197/200 | 798/879 | **1.10x** | 1.14x |
| (1,64,8192,128) 8192 | THD | 374/440 | 227 | 198/240 | 800/907 | **1.13x** | 1.19x |

**geo_mean**：BSHD e2e **1.253x** / comm+GEMM **1.374x**；THD e2e **1.243x** / comm+GEMM **1.363x**。

### 4 GPUs（ATTN = FlashAttention-4）

| `(bs,nh,seq,hd)` hidden | 布局 | PRE ours/torch | ATTN | POST ours/torch | e2e ours/torch | **e2e** | **c+g** |
|---|---|---|---:|---|---|---:|---:|
| (1,32,4096,128) 4096 | BSHD | 122/196 | 119 | 75/112 | 317/427 | **1.35x** | 1.56x |
| (1,32,4096,128) 4096 | THD | 124/182 | 119 | 75/117 | 318/418 | **1.31x** | 1.50x |
| (1,56,4096,128) 7168 | BSHD | 293/349 | 152 | 149/174 | 593/674 | **1.14x** | 1.18x |
| (1,56,4096,128) 7168 | THD | 292/351 | 152 | 148/177 | 592/679 | **1.15x** | 1.20x |
| (2,32,4096,128) 4096 | BSHD | 188/303 | 156 | 110/154 | 454/613 | **1.35x** | 1.53x |
| (2,32,4096,128) 4096 | THD | 190/412 | 156 | 110/156 | 455/723 | **1.59x** | 1.89x |
| (2,56,2048,128) 7168 | BSHD | 293/351 | 116 | 148/183 | 557/650 | **1.17x** | 1.21x |
| (2,56,2048,128) 7168 | THD | 292/349 | 116 | 176/175 | 585/640 | **1.10x** | 1.12x |
| (1,64,8192,128) 8192 | BSHD | 684/778 | 388 | 331/327 | 1404/1493 | **1.06x** | 1.09x |
| (1,64,8192,128) 8192 | THD | 674/818 | 388 | 337/383 | 1399/1589 | **1.14x** | 1.19x |

**geo_mean**：BSHD e2e **1.207x** / comm+GEMM **1.300x**；THD e2e **1.244x** / comm+GEMM **1.353x**。

### 结论

1. **comm+GEMM（融合算子真正发力的部分）8 卡 ~1.37x、4 卡 ~1.30x 加速**；含 attention 的全链路 e2e
   8 卡 ~1.25x、4 卡 ~1.22x（attention 占比越大，e2e 稀释越多，符合预期）。
2. **换用 FlashAttention-4 后 attention 大幅变快**（例如 8192 shape 的 8 卡 attn 从 SDPA 的 796us 降到
   227us），attention 在 e2e 中的占比下降，因此 e2e 加速比比 SDPA 版本更高（8 卡从 ~1.20x 提升到 ~1.25x）；
   而 comm+GEMM 口径基本不变（融合算子本身未改），仍为 ~1.37x。
3. **BSHD 与 THD 数值几乎一致**（同 shape 两行差异多在测量抖动内），实证算子对两种输入布局**等价**。
4. **小 K / 大 batch 通信占比高 → 融合收益最大**（如 `(1,32,4096)` comm+GEMM 1.6~1.8x）；
   大 hidden（8192）计算 bound，e2e 收敛到 ~1.1x，但每个 shape 仍稳定快于 torch-native。

---

## 5. `test_ulysses_full_attn_flow.py` 中 rel ~1e-3 误差分析

**结论先行：这不是 bug，1e-3 是全 bf16 链路对 FP32 真值的正常精度量级；两个融合算子本身数值精确。**

### 5.1 误差分解（8 卡，shape `(1,32,2048,128,4096)`，逐项实测）

| 环节 | 测量 | rel | 含义 |
|---|---|---:|---|
| PRE op | `rel(q, ref)` | **0.00e+00** | 融合 QKV-proj + A2A **逐元素精确** |
| ATTN | dist_attn vs **全 head** attention 切片 | 1.11e-03 | bf16 attention 在不同 head 数下的归约顺序差异 |
| ATTN | dist_attn vs **本 rank head 组** attention | **0.00e+00** | 同 head 数 → **精确相等**（证明上一行非真误差） |
| 输出 GEMM | bf16 输出 vs FP32 输出 | 1.41e-03 | bf16 输出 GEMM 的量化地板 |

> 说明：上表的逐项分解最初用 SDPA 实测；改用 FlashAttention-4 后整链路 `test_ulysses_full_attn_flow.py`
> 实测 rel 仍稳定为 **1.41e-3**（3/3 PASS），与「纯 bf16 输出 GEMM 地板」一致，结论不变。
> pre-attn 链路（`test_ulysses_pre_attn_flow.py`）FA4 分布式 vs FA4 参考逐组计算时 q/k/v/attn rel 恒 **0.0**。

### 5.2 两个独立的 bf16 来源

旧测试 rel ≈ 2.6e-3，由两个**相互独立、且都合理**的 bf16 误差叠加而成：

1. **Attention 参考构造伪差（~1.1e-3，非真实误差）**：旧参考用**一次覆盖全部 32 个 head 的 attention**，
   而分布式路径每个 rank 只跑自己的 `local_nh=4` 个 head。bf16 attention 在不同 head 数下选择不同的
   kernel tiling / 归约顺序，导致 ~1e-3 的位级差异。**这不是算子误差** —— 一旦参考也按「本 rank head 组」
   计算 attention，rel 立刻变成 **0.0**（见上表第 3 行）。

2. **输出投影 bf16 vs FP32（~1.4e-3，预期地板）**：参考的 `Wo` 投影用 FP32（`ag.float() @ Wo.float().t()`）
   即真值，而算子是 bf16（fp32 累加、bf16 输入/输出）。bf16 尾数 8 bit，单元素相对舍入 ~2⁻⁸≈3.9e-3，
   经 `hidden` 维点积部分平均后，输出平均相对误差 ~1.4e-3。这是 **bf16 输出 GEMM 的精度地板**，无法再降
   （除非把算子也改成 FP32 输出）。

### 5.3 已做的修正

把测试参考的 attention 改为**按 rank head 组逐组计算**（与分布式执行一致），消除第 1 项伪差，同时
保留 FP32 输出投影（对真值比较）。修正后：

```
rel = 1.41e-03   （三个 shape 完全一致，3/3 PASS）
```

跨 shape 高度稳定的 1.41e-3 恰好等于「纯 bf16 输出 GEMM 量化地板」（5.1 表最后一行），进一步佐证残差是
**系统性 bf16 舍入**而非数值 bug。阈值 `0.03` 仍然合适。

### 5.4 要点

- 融合算子精确：PRE op q/k/v 逐元素 rel = 0；POST op 在输入/dtype 一致时同样精确。
- 残差 = bf16 attention + bf16 输出 GEMM 对 FP32 真值的固有舍入，**1e-3 级别对全 bf16 链路完全正常，不偏大**。
- 旧的 2.6e-3 偏高部分来自参考端「全 head 一次 attention」的构造方式，与算子无关，已通过 head 组一致化消除。
