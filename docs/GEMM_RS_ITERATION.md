# GEMM-RS 迭代记录

## 背景

本文件用于记录 `bf16_gemm_rs_nt` 的**连续迭代过程**（改动、验证、结论），口径对齐 `A2A/AG` 迭代文档。

目标：

1. 面向 `SM100` 的单机（intra-node）`Reduce-Scatter + GEMM` 持续优化；
2. 以 `flux` 的设计思路作为参考，但不被既有实现绑定；
3. 每轮迭代必须包含：**改动点 -> 正确性/性能结果 -> 去留结论**。

---

## 当前基线（承接 `PROGRESS.md`）

- 主线：`bf16_gemm_rs_nt`
- 正确性：`tests/test_gemm_rs.py 2` 通过（6/6）
- 最近基线（13 shape）：geo mean 约 `1.10x`（fused vs separate）
- 最近重点 5 shape：geo mean 约 `1.17x`（vs torch-native）

> 说明：详细命令与最新统一口径以 `docs/PROGRESS.md` 为准；本文件只记录“迭代动作与结论”。

---

## 2026-06-18：Iteration 1 — 对齐 AG 风格，尝试 A/B 分离加载（失败回退）

### Change

文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`

- 参考 `AG GEMM`，尝试把 RS kernel 的 load 结构由
  - `Warp4: A+B unified load`
  改为
  - `Warp4: Load A`
  - `Warp5: Load B`
- 对应修改 stage barrier producer 计数（从单 producer 调整为双 producer）。

### Result

- 编译可过，但在实测中出现 launch failure / 不稳定行为。
- 该版本未形成稳定可复现收益。

### Verdict

- **回退**到稳定的单 load warp 版本（`A+B unified`）。
- 结论：当前 RS 的 A/B 分离加载不能直接照搬 AG，需要结合 RS 通信阶段重新设计同步协议。

---

## 2026-06-18：Iteration 2 — 通信轮询 scope 优化（保留）

### Change

文件：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`

- 在 ready flag 轮询中按来源 rank 分流：
  - self rank：`ld_acq`（GPU scope）
  - remote rank：`ld_acq_sys`（system scope）
- 保留 2-rank hot path 的并行轮询结构（warp0/warp1 分摊）。

### Result

- 内核恢复稳定可运行。
- 单点与 focus-only 复测显示该改动为**低风险小收益/中性偏正**，无正确性回退。

### Verdict

- **保留**。
- 结论：在不改变主 pipeline 的前提下，先吃 memory scope 精简收益是性价比最高的路径。

---

## 2026-06-18：Flux 参考结论（用于后续迭代方向）

- `flux` 的 RS 路径中，TMA 主要用于**本地可见缓冲区搬运/读取**；
- 跨 rank 通信仍主要依赖 `nvshmem put/get + barrier/signal`（不是“跨 GPU 直接 TMA 通信”）。

这意味着后续在 `DeepGEMM` 上可优先尝试：

1. 保持跨 rank 通信路径稳定；
2. 把“本地 reduce/fetch 阶段”做成更强的 TMA 化与流水解耦；
3. 再评估是否值得重构成更接近 flux 的双核或职责拆分模型。

---

## 2026-06-18：Iteration 3 — 重构为真·Flux pull 式 dual-kernel（进行中）

### Change

把主线 `sm100_bf16_gemm_rs.cuh` 从「单 kernel push + in-kernel reduce」彻底重构为
**真·Flux pull 式 dual-kernel**，对齐 Flux 的
`Sm90AuxStoreReduceScatter`(GEMM 写本地 scatter buffer) + `Sm90ReduceScatterDma`(独立 kernel 从远端 pull)：

- `impls/sm100_bf16_gemm_rs.cuh`（Kernel 1, GEMM compute）：
  - 256T(8 warps)，**移除 comm warps**（旧版 384T → 寄存器 spilling 的根因被消除）；
  - epilogue **纯本地 scatter write**：每个 tile 按 `dst_rank` 写到本地 `slot[dst_rank]`，
    置**本地** per-tile ready flag（`st_rel_sys`，system scope 可被 peer 读到）；
  - GEMM epilogue **不再跨 NVLink**（这是相对 push 式 v3 的核心收益）。
- `impls/sm100_rs_reduce.cuh`：新增 `kPullBased` 模板分支：
  - rank R 对自身 chunk 每个 tile，poll 各 src rank 的**远端** `flag[R][m][n]`（`sym_buffer.map` 到 src）；
  - FP32 累加各 src 的**远端** `slot[R]`（map P2P，self 映射回本地）→ 写 output；
  - 读完后远端 reset flag（Flux `wait_eq_reset` 语义）。
- `jit_kernels/impls/sm100_bf16_gemm_rs.hpp`：改为 dual-kernel pull 编排
  （compute_stream 跑 GEMM + comm_stream 跑 pull reduce + event 同步），复用 `GemmRSComputeConfig`。
- `jit_kernels/impls/sm100_rs_reduce.hpp`：把 `pull_based` 透传到 Args/generate_impl（push v3 路径不受影响）。
- `deep_gemm/gemm_rs/__init__.py`：默认路由切到 pull；保留 `DG_GEMM_RS_IMPL=v3/push` 旧 push 路径可选。

跨迭代正确性：GEMM 起始 `nvlink_barrier` + host 端 `cudaStreamSynchronize(comm_stream)`/event 门控，
保证「上一轮所有 rank 的 reduce(含远端 reset) 完成 → 本轮才 set flag」，杜绝 stale flag 读。

### Result

- C++ 扩展编译通过，代码已 commit & push。
- **Bug 修复**：首轮 2-GPU 测试死锁（nvlink_barrier timeout）。根因是 host 端 per-call
  `cudaMemsetAsync` 清零 barrier 区与对端 GEMM 的 in-flight NVLink 信号写竞争
  （`cudaStreamSynchronize(comm_stream)` 延迟某 rank 的 memset → 把对端已送达的信号清掉 → 死锁）。
  `comm::nvlink_barrier` 本身是 phase/sign 自复位协议，buffer 创建时已 `zero_()`，
  故 per-call memset 既不必要又有害 → **移除 memset**。
- **正确性**：`tests/test_gemm_rs.py 2` → **6/6 PASS，max_diff=0.0**（与参考逐元素精确一致）。
- **性能（2 GPU，3 iter，指定 13 shape）**：geo_mean **0.584x vs torch / 0.582x vs sep**，
  fused 平均 **628.5T** vs separate 1065.2T。**明显慢于旧 push v3（~1.10x）**。

### Verdict

- **正确性达标，性能回退（需优化）**。
- 根因（性能）：真·Flux pull 的速度优势依赖 Flux `Sm90ReduceScatterDma` 的 **TMA 流水线 fetch**；
  当前 RS reduce 用**朴素标量 P2P 读**（每元素串行读 num_ranks 个远端 slot），且 reduce 与 GEMM
  双流**抢占 SM/带宽**。而 push v3 把跨卡传输放在 GEMM epilogue 的 TMA async store 中被计算掩盖、
  reduce 只读本地，所以更快。
- 下一步（perf）：把 pull reduce 改造为 **TMA 流水线 fetch+reduce**（对齐 Flux `Sm90ReduceScatterDma`：
  远端 → smem 的 producer/consumer 流水），或改进 GEMM/reduce 的 SM 划分与重叠；
  在此之前，主线高性能路径仍可用 `DG_GEMM_RS_IMPL=v3`（push）回退。

---

## 2026-06-18：Iteration 4 — 收敛到 pull，删除 push 路径

### Change

既然 Flux 单机 RS = pull（`Sm90ReduceScatterDma` 为 TMA 远端 fetch；push 仅用于跨节点 ring），
主线收敛为唯一 pull 实现，**删除 push 路径**：

- 删除 `impls/sm100_bf16_gemm_rs_compute.cuh`、`jit_kernels/impls/sm100_bf16_gemm_rs_compute.hpp`、
  `apis/gemm_rs_compute.hpp`、孤立的 `heuristics/gemm_rs.hpp`、`*.cuh.bak`；
- 删除 v3 test/bench：`tests/test_gemm_rs_v3.py`、`quick_bench_v3.py`、
  `benchmarks/bench_gemm_rs_v3.py`、`benchmarks/bench_gemm_a2a_pdl.py`；
- `sm100_rs_reduce.cuh/.hpp` 收敛为 **pull-only**（移除 `kPullBased`/`pull_based` 开关）；
- `python_api.cpp` 移除 `gemm_rs_compute` 注册；`gemm_rs/__init__.py` 移除 `bf16_gemm_rs_nt_v3` 与
  `DG_GEMM_RS_IMPL` 开关，`deep_gemm/__init__.py` 移除其导出。

### Result

- 编译通过；`tests/test_gemm_rs.py 2` → **6/6 PASS, max_diff=0.0**（清理后正确性不变）。

### Verdict

- **保留**。主线唯一 = 真·Flux pull。性能优化转入下一步（TMA 流水线 fetch）。

---

## 2026-06-18：Iteration 5 — pull RS reduce 高 MLP 重写（大幅提速，保留）

### 背景诊断（AKO4ALL profile 思路）

先用 SM carveout 实验探因：把 GEMM 限制在 `num_sms-C` 个 SM、reduce 用 `C` 个 SM。
结果 carveout **越切越慢**（C=8 时 fused 飙到 2949us）。由此测得 reduce 吞吐随 SM 数**近似线性**
（C=8→2782us、16→1439、24→1032、32→775、全 SM→158us，约 24000 SM·us 恒定），
说明 reduce 是**延迟/occupancy-bound（MLP 太低）而非带宽-bound**。
推算「最优 carveout 切分」最优点 C≈73 也只有 ~329us，与串行 325us 持平 → **carveout 在 reduce 低效时无意义**。
结论：真正瓶颈是 **reduce kernel 自身 MLP 太低**，必须先把它做成带宽-bound。

### Change

文件：`deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh`

- **预计算固定基址**：slot[rank_idx] 的远端基址 `slot_base[s]` 与 flag 基址 `flag_base[s]` 对整个
  kernel 恒定 → 提到 hot loop 外只算一次，消除 inner 循环里每元素的 `get_partial_ptr`（64 位乘法）+ `map`。
- **高 MLP 批处理**：Phase 2 每线程一次性处理 `kUnroll=4` 个 128-bit 向量，先把全部
  `kUnroll × kNumRanks` 个 P2P load 发射出去、再统一消费 → MLP 从 ≈kNumRanks(2) 提升到 8，
  少量 SM 即可逼近 P2P 带宽。
- `__launch_bounds__(kNumThreads, 2)`（给 unroll 让出寄存器，避免 spill）。
- 同时在 host `sm100_bf16_gemm_rs.hpp` 加入 SM carveout 机制（`DG_RS_REDUCE_SMS`，**默认 0 关闭**），
  留作 reduce 变成带宽-bound 后再评估 overlap。

### Result

- 正确性：`tests/test_gemm_rs.py 2` → **6/6 PASS, max_diff=0.0**。
- 性能（2 GPU，13 shape）：geo_mean **0.733x vs torch / 0.739x vs sep**（此前 0.606x / 0.611x），
  avg fused **814T**（此前 660T）。单 shape 普遍升到 **0.77~0.83x**。
  （注：全量顺序 bench 里 `4096x7168x4096` 偶发 0.33x，单独复测稳定 0.81x，系顺序跑的 L2/NCCL 干扰噪声。）

### Verdict

- **保留**。MLP 重写是迄今最大单步收益（+0.13x geo / +150T）。
- 下一步：reduce 已接近带宽-bound，可重新评估 **SM carveout + tile 级 overlap**；
  并尝试 self-rank 走本地快路径、`kUnroll` 调参、cp.async 预取。

---

## 2026-06-18：Iteration 6 — carveout 死胡同确认 + kUnroll 调参（关键结论）

### Change / 实验

1. **重测 SM carveout**（reduce 已高 MLP 后）：`DG_RS_REDUCE_SMS ∈ {16,32,48,64}`，3 个代表 shape。
2. **kUnroll 扫参**：4 → 8（BLOCK 128×128 bf16 时 vecs_per_tile=2048=`kNumThreads×8`，单趟无尾部浪费）。

### Result

- **carveout 仍全面更差**（C=16 时 fused 是 C=0 的 ~3x）。reduce 吞吐**仍随 SM 数近似线性**
  （即使 MLP=8），且 full-SM 下远端读仅 ~360–510 GB/s（NVLink5 峰值 ~900GB/s）。
- **kUnroll 4→8 仅 ~1%**（如 16384x7168x7168：3175→3149us）。MLP 已到顶，保留 8（更干净）。
- 正确性 `tests/test_gemm_rs.py 2` 6/6 PASS。

### Verdict（关键结论，指导后续）

- **SM-load reduce 已触及自身天花板**（既非 MLP 不足，也非带宽峰值，而是 SM 走 LSU 跨 NVLink
  load 的有效并发上限）。
- **SM carveout 对 SM-based reduce 是零和**：从 GEMM 切 C 个 SM → GEMM tensor 吞吐降 C/148，
  而 reduce 延迟-bound 需要很多 SM 才能跟上；代数上「最优切分」点的 fused ≈ 当前串行值甚至更差
  （大/小 shape 均验证）。**放弃 carveout**（`DG_RS_REDUCE_SMS` 保留但默认 0）。
- 要突破 ~0.80x → **必须把跨卡搬运移出 SM 算力路径**，用 **TMA/copy 引擎** 把远端 partial 异步
  fetch 进 smem（SM 只做加法），并与 GEMM **共驻同一批 SM**（不抢 tensor core）——即 Flux
  `Sm90ReduceScatterDma` / push-v3 TMA-epilogue 的本质。这是下一个大迭代。

---

## 2026-06-18：Iteration 7 — reduce grid 过订阅（oversubscription，大幅提速，保留）

### 洞察

Iter 6 说 SM-load reduce「触顶」，但其实是**每 SM 只 1 个 reduce block**（grid=`min(total_tiles, num_sms)`）。
reduce 在 GEMM 之后跑（SM 全空），那就**过订阅**：每 SM 放多个 reduce block → 更多 warp 同时发射
P2P load → 更多 outstanding 请求 → 更高 NVLink 有效带宽。这是 latency/concurrency-bound 的正解。

### Change

文件：`csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`

- reduce grid 由 `num_sms` 改为 `num_sms × reduce_grid_mult`（kernel 本就 grid-stride，正确性不变）。
- `reduce_grid_mult` 默认 **2**（env `DG_RS_REDUCE_MULT` 可调）。扫参 {1,2,3,4}：2 全面优于 1，
  与 4 接近且更稳（启动开销小）。

### Result

- 正确性：`tests/test_gemm_rs.py 2` → **6/6 PASS**。
- 性能（2 GPU，6 iter，13 shape）：geo_mean **0.835x vs torch / 0.836x vs sep**
  （Iter 6 的 ~0.73x → 0.835x），avg fused **906T**（814→906T）。最差 shape 0.77x（baseline 0.51x）。
  分组：K=7168 0.85x、focus 5 shape 0.85x、M/rank≥8192 0.86x(vs sep)。

### Verdict

- **保留**。零风险纯 host 改动，单步 +0.10x geo / +90T。
- 本会话累计：**0.61x → 0.835x（vs torch），660T → 906T**（Iter 5 高 MLP + Iter 7 过订阅）。

---

## 本会话进度小结（2026-06-18）

| 阶段 | geo vs torch | geo vs sep | avg fused | 关键改动 |
|------|------|------|------|------|
| 起点（pull 初版） | 0.606x | 0.611x | 660T | 朴素标量 P2P reduce |
| Iter 5 | 0.733x | 0.739x | 814T | reduce 高 MLP（预计算基址 + kUnroll 批量发射）|
| Iter 6 | ~0.73x | ~0.73x | ~820T | kUnroll=8；确认 carveout 死胡同 |
| Iter 7 | **0.835x** | **0.836x** | **906T** | reduce grid 过订阅 ×2 |

---

## 2026-06-18：Iteration 8 — 跨卡传输 fused 进 epilogue（push-scatter），冲 >1.0x（保留）

### 决策依据（实测物理结论）

Iter 6/7 后又做了两组实验，**证明分离 reduce kernel 在本硬件结构性无法 overlap**：
- **SM carveout**（切 SM 给 reduce）：零和——GEMM tensor 吞吐降 C/148，延迟-bound 的 reduce 又需很多 SM，
  代数最优点 fused ≈ 串行甚至更差（大小 shape 均验证）。
- **smem reserve 让 reduce 共驻**：单调更差。根因——GEMM 用 `__launch_bounds__(256,1)` 独占寄存器，
  独立 reduce 要共驻须把 GEMM 寄存器砍半 → spilling（正是要避免的）。
- 每种配置实测 fused ≈ GEMM + reduce_tail，**零 overlap**。

结论：`separate = GEMM + NCCL_RS`，要 >1.0x 必须让 RS **藏进 GEMM**。唯一可行 = **把跨卡 NVLink 传输
放进 GEMM epilogue 用 TMA async store，与后续 tile 的 MMA 重叠**；reduce 只读「本地」已汇聚 partial。
这正是 Flux 融合 GemmRS 让 comm 藏进 compute 的本质（也是本仓库历史 push-v3 实测 1.10x 的架构）。

### Change（仅改 epilogue 目标指针 + reduce 读本地，非大重写）

- `impls/sm100_bf16_gemm_rs.cuh` epilogue：每个 tile（我对 chunk dst_rank 的 partial）经 `tma_store_1d`
  **push 到 dst_rank 的 buffer slot[rank_idx]**（`sym_buffer.map` 到 dst；self 即本地），
  并置 dst_rank 的 `flag[rank_idx][m][n]`（`st_rel_sys`）。跨卡传输由 TMA 引擎发射、与 MMA 重叠。
- `impls/sm100_rs_reduce.cuh`：reduce 改为**纯本地**累加——各 src 已把对本 rank chunk 的 partial
  push 进本地 slot[s]，故 `slot_base[s]/flag_base[s]` 直接取本地（去掉 `map`），poll 本地 flag
  （release.sys 由 peer 写入）→ 本地 FP32 求和 → output → reset 本地 flag。本地 HBM，极快。

### Result

- 正确性：`tests/test_gemm_rs.py 2` → **6/6 PASS, max_diff=0.0**。
- 性能（2 GPU，6 iter，13 shape）：geo_mean **0.964x vs torch / 0.973x vs sep**，avg fused **1054T**
  （Iter 7 的 0.835x/906T → 0.964x/1054T，**+0.13x**）。
  **多个 shape 已 >1.0x**：16384x7168x4096 **1.09x**、8192x4096x4096 1.06x、8192x7168x4096 1.05x、
  4096x7168x4096 1.04x、16384x7168x4096... (vs sep)。
- 弱势点：小 K（2048x7168x2048 0.83x、4096x7168x2048 0.89x，GEMM 短→本地 reduce tail 占比大）；
  超大 N·K（16384x7168x7168 0.95x，push 体量大、NVLink 接近饱和）。

### Verdict

- **保留**。这是冲过 1.0x 的关键架构步（单步 +0.13x，多 shape 破 1.0x）。
- 本会话累计：**0.61x → 0.973x（vs sep），660T → 1054T**。

---

## 2026-06-18：Iteration 9 — 去掉冗余 CPU 同步（保留）

### Change

`csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`：删除 step-0 的 `cudaStreamSynchronize(comm_stream)`。
它是冗余的——step-3 已让 compute_stream 通过 `comm_done_event` 等待本轮 reduce，且 comm_stream FIFO 有序，
下轮 GEMM 在 GPU 上天然排在本轮 reduce 之后；CPU 阻塞只会串行化 launch、拖累小 shape。

### Result

- 正确性：`tests/test_gemm_rs.py 2` → **6/6 PASS**。
- 性能（2 GPU，8 iter，13 shape）：geo_mean **0.989x vs torch / 0.995x vs sep**，avg fused **1082T**
  （Iter 8 的 0.964x/1054T → 0.989x/1082T）。
- **分组（Megatron SP 主目标）**：
  - **User focus 中大 5 shape：1.043x vs torch / 1.049x vs sep** ✅
  - **M/rank≥8192：1.030x vs sep** ✅
  - 单 shape >1.0x：8192x4096x4096 1.09x、4096x7168x4096 1.07x、8192x7168x4096 1.07x、16384x7168x4096 1.06x、
    4096x4096x4096 1.05x 等。

### Verdict

- **保留**。中大 shape（主目标）已稳超 1.0x；整体 0.995x（与 separate 基本持平）。
- 残余拖累：小 K（2048x7168x2048 0.85x、4096x7168x2048 0.91x，separate≈纯 GEMM、NCCL RS≈0，结构性难超）；
  16384x7168x7168 0.95x（push 体量大）。

---

## 本会话总进度（2026-06-18）

| 阶段 | vs torch | vs sep | avg fused | 关键改动 |
|------|------|------|------|------|
| 起点 | 0.606x | 0.611x | 660T | 朴素标量 P2P pull reduce |
| Iter 5 | 0.733x | 0.739x | 814T | reduce 高 MLP |
| Iter 7 | 0.835x | 0.836x | 906T | reduce grid 过订阅 ×2 |
| Iter 8 | 0.964x | 0.973x | 1054T | **跨卡传输 fused 进 epilogue（push-scatter，与 MMA 重叠）** |
| Iter 9 | **0.989x** | **0.995x** | **1082T** | 去冗余 CPU 同步 |

**focus 中大 shape：1.049x vs sep ✅（主目标已达 >1.0x）**

---

## 2026-06-18：Iteration 10 — 去 flag + 纯 1D 连续流式 reduce（决定性突破，保留）

### 洞察

reduce 在 stream 序上**总是等 GEMM 完成才跑**（host 的 `gemm_launched_event` 在 compute_stream 中
排在 GEMM 之后 → 等于 GEMM 完成），且 GEMM 末尾的 system-scope `nvlink_barrier`(tag42, `red_add_rel_sys`
/`ld_acq_sys`) 已保证所有 push 全局可见。所以 **per-tile flag 轮询完全冗余**。
而每 rank 的 chunk 在 buffer 内是 `[token][hidden]` **完全连续**的。

### Change

- `impls/sm100_rs_reduce.cuh`：**去掉 flag/poll/reset 与全部 `__syncthreads`**，重写为
  **纯 1D 连续流式**累加（`output[i]=Σ_s slot[s][i]`，i 遍历整个连续 chunk），kUnroll=8 高 MLP、
  完美 coalesce → 打满本地 HBM 带宽。
- `impls/sm100_bf16_gemm_rs.cuh`：epilogue 去掉 per-tile flag set（map+`st_rel_sys`），可见性交给末尾 barrier。

正确性依据：reduce 等 GEMM 完成 + tag42 system-scope barrier 保证 push 可见；跨迭代 slot 覆盖安全由
host event 序（下轮 GEMM 等本轮 reduce）+ 起始 tag41 barrier 保证。

### Result

- 正确性：`tests/test_gemm_rs.py 2` → **6/6 PASS, max_diff=0.0**（连跑 2 次稳定）。
- 性能（2 GPU，8 iter，13 shape）：geo_mean **1.148x vs torch / 1.142x vs sep**，avg fused **1232T**
  （Iter 9 的 0.989x/1082T → **1.148x/1232T**，单步 +0.15x）。
- **全部 13 shape ≥ 1.02x vs sep**（最差 16384x7168x7168 1.02x）。旧最大短板小 K 反成最强：
  **2048x7168x2048 1.31x、4096x7168x2048 1.38x**；focus 中大 5 shape **1.143x vs sep**。

### Verdict

- **保留**。决定性突破——整体稳超 1.0x 且超过历史 push-v3(1.10x)。
- 根因：旧 flag 版每 tile 有 `ld_acq_sys` 轮询 + 2×`__syncthreads` + strided 寻址，对小 shape 开销占比极大；
  连续流式全部消除。

---

## 本会话总进度（2026-06-18）— 0.61x → 1.14x

| 阶段 | vs torch | vs sep | avg fused | 关键改动 |
|------|------|------|------|------|
| 起点 | 0.606x | 0.611x | 660T | 朴素标量 P2P pull reduce |
| Iter 5 | 0.733x | 0.739x | 814T | reduce 高 MLP |
| Iter 7 | 0.835x | 0.836x | 906T | reduce grid 过订阅 ×2 |
| Iter 8 | 0.964x | 0.973x | 1054T | 跨卡传输 fused 进 epilogue（push-scatter，与 MMA 重叠）|
| Iter 9 | 0.989x | 0.995x | 1082T | 去冗余 CPU 同步 |
| **Iter 10** | **1.148x** | **1.142x** | **1232T** | **去 flag + 纯 1D 连续流式 reduce** |

**全部 13 shape ≥1.02x vs sep；focus 中大 1.143x；峰值 1.40x。**

---

## 2026-06-18：Iteration 11 — 4/8-GPU 验证 + reduce kUnroll 自适应 rank 数（保留）

### 背景

4/8 卡才是真实场景。首轮多卡 bench：4 卡 geo 1.140x vs sep（全 ≥1.02x）很好；
但 **8 卡部分超大 N·K 跌到 0.93–0.98x**。根因：reduce 的寄存器缓冲 `reg[kUnroll][kNumRanks]`
= `kUnroll×kNumRanks×4` 寄存器；kUnroll=8、kNumRanks=8 → **256 regs/线程，严重 spilling**。

### Change

`impls/sm100_rs_reduce.cuh`：kUnroll 随 rank 数自适应，保持 in-flight loads(≈kUnroll×kNumRanks)
与寄存器压力恒定：`kUnroll = kNumRanks>=8 ? 2 : (kNumRanks>=4 ? 4 : 8)`。

### Result（2 GPU，8 iter；4/8 GPU，8 iter）

- 正确性：`test_gemm_rs.py {2,4,8}` 全 **6/6 PASS, max_diff=0.0**。
- **跨规模 geo_mean（vs sep）高度一致**：

  | GPUs | vs torch | vs sep | focus 中大 (vs sep) | avg fused | 全部 ≥1.0x? |
  |------|------|------|------|------|------|
  | 2 | 1.148x | **1.142x** | 1.143x | 1232T | 是（最差 1.02x）|
  | 4 | 1.129x | **1.148x** | 1.199x | 1225T | 是（最差 1.00x）|
  | 8 | 1.103x | **1.140x** | **1.202x** | 1193T | 基本是（仅 16384x7168x7168 0.98x）|

- 8 卡修复效果：geo 1.079→**1.140x** vs sep；小 K 0.97→1.06x；focus 1.142→1.202x。

### Verdict

- **保留**。自适应 kUnroll 在 4/8 卡都是净赢，2 卡不变。多卡（4/8）主线稳定 **~1.14x vs sep**、
  focus 中大 **1.20x**，已全面超越 separate 基线与历史 push-v3(1.10x)。
- 唯一残余：8 卡 `16384x7168x7168`（M/rank=16384，N=K=7168，reduce 需读 8 个 235MB slot）0.98x vs sep。

---

## 2026-06-18：Iteration 12 — 调度改为 chunk-sequential ring「self 最后」（保留）

### Change

`impls/sm100_bf16_gemm_rs.cuh` `get_next_block`：把 dst_rank 的遍历由「per-tile 交错」改为
**phase 外层 / tile 内层**——先把其它 rank 的 chunk（远端 push）整段算完（ring 顺序 r→r+1,r+2,…，
各 rank 同 phase 目标互不相同 = NVLink 负载均衡、无热点），**自己的 chunk 放最后**（epilogue 映射本地、
末尾零跨卡通信，且远端 push 全部前置 → 更好被 MMA 掩盖、并可与上一次调用的 reduce/通信重叠）。

### Result（focus 中大 5 shape，8 iter，vs sep）

| GPUs | 调度前(interleaved) | 调度后(self-last ring) |
|------|------|------|
| 2 | ~1.14x | 1.130x（2 rank 收益最小，噪声内）|
| 4 | 1.199x | 1.199x（持平）|
| 8 | 1.202x | **1.233x（+0.03x）** |

- 8 卡 focus 单 shape（vs sep）：4096x4096x4096 1.28x、4096x7168x4096 1.26x、8192x4096x4096 1.29x、
  8192x7168x4096 1.21x、4096x4096x7168 1.14x；avg fused 1337T。
- 正确性：`test_gemm_rs.py {4,8}` 6/6 PASS。

### Verdict

- **保留**。rank 越多收益越明显（8 卡 +0.03x，4 卡持平，2 卡噪声内），无回退。符合「self 最后 + ring 均衡」预期。

---

## 2026-06-22：Iteration 13 — 冲 >1.3x 的诊断（多条路径实测，全部回退；定位真正瓶颈）

本轮目标是把 8 卡 focus 从 1.22x 推到 >1.3x。系统性实测了多条路径，**全部中性或回退、已回退**，
但精确定位了瓶颈结构（数据驱动），为后续指明唯一有效方向。

### 各路径实测（8 卡 focus，baseline 3efe39d = 1.224x vs sep；正确性均 6/6 PASS）

| 路径 | 结果 | 结论 |
|------|------|------|
| reduce 合并到 compute_stream（去 comm_stream/event）| ~1.22x（中性）| tail 不是 event/跨 stream 开销 |
| reduce grid 解除 total_tiles 上限 + mult 扫参（2/3/4/5/6）| 全在 1.22~1.23x（噪声）| reduce 不是 occupancy/grid 限制 |
| **PDL**（gemm trigger + reduce grid-sync，单 stream）| **1.200x（回退）**| GEMM 占满 SM 时 reduce 无法共驻，PDL 预调度机制净负 |
| reduce slot 读用 `__ldg`（只读缓存）| 1.217x（中性）| 不是缓存绕过问题 |
| epilogue per-tile `tma_store_wait<0>` 放松为 `<stages-1>` + 末尾排空 | 1.212x（中性）| push 暴露不是 per-tile 排空导致 |
| 跨调用 overlap（reduce[i] ‖ gemm[i+1]）| 未实现 | **破坏 API 契约**（输出对 compute_stream 不 ready）；Megatron SP 输出有依赖链不可跨调用 overlap，纯属虚高 benchmark，放弃 |

### 关键诊断（实测，数据驱动）

用 event 实测 reduce kernel 真实耗时 + 纯 GEMM 对比，得到 8 卡 focus 的精确分解：

| shape | 纯 deepgemm GEMM | gemm+push(SKIP_REDUCE) | +reduce(full) | sep | 当前 vs sep |
|------|------|------|------|------|------|
| 4096x4096x4096 | 626us | 755us | 819us | 1030us | 1.26x |
| 8192x7168x4096 | 2617us | 2870us | 3034us | 3653us | 1.19x |

两大开销：
1. **push 开销（gemm+push − 纯GEMM ≈ 129~253us）> reduce tail（49~164us）**。
   - reduce tail：实测 reduce kernel 49us(4096³)/162us(8192x7168)，**读 R 个 slot + 写 1 = (R+1)=9× 输出
     ≈ 302MB，@6.16TB/s ≈ 已打满 HBM 带宽**。→ reduce **不是低效，是不可压缩的内存流量**（除非算法级降流量）。
   - push 开销：自定义 gemm_rs kernel 比标准 gemm 慢 ~20%。根因疑为 **epilogue 用逐行 `tma_store_1d`
     推 push（每行仅 ~128B 的小 TMA 传输）**，而标准 gemm 用整块 2D TMA store。

### Verdict

- 本轮所有快速改动**全部回退**（无一净正）。工作树保持在 `3efe39d`。
- **唯一有效方向（高价值、大改）**：把 epilogue 的逐行 1D TMA push 改成**整块 2D TMA store**
  （为每个目标 rank 建 TMA descriptor，远端 P2P TMA store——push-v3 历史验证可用）。这针对最大的
  ~129~253us push 开销，理论可把 gemm+push 拉近纯 GEMM → 冲 ~1.4x。
- 次选（大改）：reduce 算法级降流量（树形/分层 RS），把 (R+1)× 降到 ~2×。

---

## 下一步计划（冲 >1.3x，聚焦中大 shape）

1. **[最高价值] epilogue push 改 2D TMA store**：host 为 R 个目标 rank 的 scatter slot 建 `CUtensorMap`，
   kernel epilogue 用整块 2D TMA store 替代逐行 1D，消除小传输低效 → 缩小 push 开销。
2. 次选：reduce 算法级降流量（树形/分层）。
3. 每轮迭代必须记录 4/8 卡 × focus shape 的 benchmark 数据（见 RULE.md §7.1）。

---

## 本轮涉及文件

- `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
- `deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh`
- `csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`
- `csrc/jit_kernels/impls/sm100_rs_reduce.hpp`
- `deep_gemm/gemm_rs/__init__.py`
- `docs/PROGRESS.md`
- `docs/GEMM_RS_DESIGN.md`
- `docs/SESSION_MEMORY.md`
- `docs/FLUX_GEMM_RS_STUDY.md`
- `docs/GEMM_RS_ITERATION.md`
