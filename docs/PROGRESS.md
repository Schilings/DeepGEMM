# 算子开发进度总览（索引）

> 最后更新：2026-06-30
> 用途：**新会话据此挑选/确认目标算子**，再读该算子的 `*_DESIGN` / `*_ITERATION` 文档接班继续开发。
> 规则见 `docs/RULE.md`；接班流程见 `docs/SESSION_MEMORY.md`。

---

## 各算子状态一览

| 算子 | 入口符号 | 当前最佳（focus 中大 / geo vs sep） | 正确性 | 分支 / tag | 详细文档 |
|------|---------|------|------|------|------|
| **GEMM-RS** | `bf16_gemm_rs_nt` | focus vs sep：8卡 **1.20x** / 4卡 **1.20x**；vs torch-native：8卡 1.16x / 4卡 1.18x；全13-shape geo **1.14x** | 6/6 PASS @ {2,4,8}（test 已加 torch 原生 gemm+RS 交叉对照）| `main`（2D TMA push）；1D 版存档 tag `gemm-rs-1d-stable` | `GEMM_RS_DESIGN.md` / `GEMM_RS_ITERATION.md` / `FLUX_GEMM_RS_STUDY.md` |
| **A2A-transpose-GEMM**（Ulysses post-attn，正确版）| `bf16_a2a_transpose_gemm_nt`=**默认 M0 串行** / `_fused`=M1 overlap(opt-in)；`seq_major=True` 接 FA 原生 BSHD/THD | **M0 单节点最优**：比 torch(NCCL+2转置)基线快 **1.6~2.4×**，comm 快 3~4×（只转 1 次、融进 comm）；M1 overlap 单节点 **~parity（净负）**；`seq_major` 是 layout 匹配(非加速)，**直接覆盖 THD/varlen（uniform 切，comm 不需要 cu_seqlens）** | **{2,4,8} 4/4 PASS**（M0/fused/seq_major 三路）；`test_ulysses_attn_flow` {8}3/3；`test_ulysses_varlen_thd`(FA varlen){8} PASS | **已并入 `main`**（merge `5516d8d`） | `A2A_TRANSPOSE_GEMM_DESIGN.md`（flux 双stream/基线口径/资源/seq_major/THD 全记录）|
| **GEMM-A2A-transpose**（Ulysses **pre-attn**，GEMM+A2A）| `bf16_gemm_a2a_transpose_nt` | 8卡 geo vs sep **1.42x** / vs torch-native **1.54x**；4卡 vs sep **1.44x** / vs torch **1.55x**；fused 平均 ~1190 TFLOPS | **{2,4,8} 全 PASS**，max_diff/rel/consist **恒 0.0**（纯排列，逐元素等于精确参考；torch `matmul+all_to_all` 亦 0.0）| `main`（单 kernel，抄 GEMM-RS 改 N 切分+删 reduce+转置散射）| `GEMM_A2A_TRANSPOSE_DESIGN.md` / `GEMM_A2A_TRANSPOSE_ITERATION.md` |
| **A2A-GEMM**（旧 token-A2A，**语义错误**）| `bf16_a2a_gemm_nt` | — | ⚠️ **3/6 FAIL** + 语义错位（token(M)-A2A，非 Ulysses）→ 已被上面的 transpose 版取代 | `main`（保留未删）| `A2A_GEMM_ITERATION.md` / `A2A_GEMM_DESIGN.md`（旧）|
| **AG-GEMM** | `bf16_ag_gemm_nt` | geo **~1.13x**（8 卡）| PASS | `main`（仅 PDL 默认开启被保留）| `AG_GEMM_ITERATION.md` / `AG_GEMM_FLUX_REFERENCE.md` |

---

## Wan2.1 14B Ulysses SP Attention Benchmark（2026-06-30 新增）

> 详见 `docs/WAN21_ULYSSES_BENCH.md`。代码在 `benchmarks/wan21/`。

真实 Wan2.1 14B 训练场景（THD/PackedSequence），三条策略对比 FWD+BWD：

| 策略 | PRE | POST | BWD POST | BWD PRE |
|------|-----|------|----------|---------|
| **serial** (baseline) | matmul + NCCL A2A | NCCL A2A + matmul | serial A2A-inv + matmul | serial A2A-inv + matmul |
| **fused_std** | `bf16_gemm_a2a_transpose_nt` | `bf16_a2a_transpose_gemm_nt_fused` | serial A2A-inv + matmul | **`bf16_a2a_transpose_gemm_nt`** (M0, Wqkv_t) |
| **fused_var** | `bf16_gemm_a2a_transpose_nt` | `bf16_gemm_rs_nt` (Wo 行拆分) | **`bf16_ag_gemm_nt`** (AG+GEMM) | **`bf16_a2a_transpose_gemm_nt`** (M0, Wqkv_t) |

8 GPU B300 结果（FWD+BWD, us）：

| Shape | serial | fused_std | fused_var | 加速比 |
|-------|--------|-----------|-----------|--------|
| 1x8K | 4821 | 6470 | **4335** | 1.11x |
| 1x32K | 18261 | 19131 | **17008** | 1.07x |
| 1x74K | 74746 | 74163 | **71972** | 1.04x |
| 1x168K | 341607 | 337642 | **334064** | 1.02x |
| 1x64K | 58657 | 57613 | **56456** | 1.04x |
| 1x148K | 268541 | 266519 | **263839** | 1.02x |

- **fused_var 全面最优**：所有 shape FWD+BWD 都最快
- **正确性**：2卡 1x8K serial verify → FWD rel=0.0019, BWD grad_X=0.0016, grad_W=0.0034 → PASS
- **后续优化**：BWD PRE 已改用融合 A2A+GEMM (M0, Wqkv_t NT)，正确性验证通过(gX_rel=0.0016)但加速不显著(comm 数据量小)；attention 是长序列瓶颈可考虑 SP+CP

---

> 每个算子的「最新当前状态 / 接班信息」见对应 `*_ITERATION.md` 顶部「当前状态」节，
> 本表只做一行总览，避免与各算子文档重复维护细节。

---

## benchmark 统一口径（所有通信类融合算子通用）

- `separate` 基线 = 标准 GEMM + 对应通信原语（如 RS：`bf16_gemm_nt + reduce_scatter_tensor`）。
- `fused` = 该算子的融合入口（如 `bf16_gemm_rs_nt`）。
- 迭代时必须先保证 correctness 不退化，再比较 `fused` vs `separate` 的增益。
- Megatron SP 导向：优先看中大 shape 分组（`N=7168` / `K=7168` / `M/rank>=2048`）的几何均值与短板 shape。
- benchmark 脚本通用开关：
  - `DG_BENCH_FOCUS_ONLY=1`（只跑重点中大 shape 子集）
  - `DG_BENCH_SHAPES="M,N,K;..."` / `DG_BENCH_SINGLE_SHAPE=M,N,K`（显式 shape）
  - `DG_JIT_USE_NVRTC=1`（无 nvcc 时）

---

## 接班即用命令（以 GEMM-RS 为例，其它算子替换脚本名）

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 8 8
```
