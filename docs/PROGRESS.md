# 算子开发进度总览（索引）

> 最后更新：2026-06-22
> 用途：**新会话据此挑选/确认目标算子**，再读该算子的 `*_DESIGN` / `*_ITERATION` 文档接班继续开发。
> 规则见 `docs/RULE.md`；接班流程见 `docs/SESSION_MEMORY.md`。

---

## 各算子状态一览

| 算子 | 入口符号 | 当前最佳（focus 中大 / geo vs sep） | 正确性 | 分支 / tag | 详细文档 |
|------|---------|------|------|------|------|
| **GEMM-RS** | `bf16_gemm_rs_nt` | 8卡 focus **1.22~1.23x** / geo **1.14x** | 6/6 PASS @ {2,4,8} | `main`（2D TMA push）；1D 版存档 tag `gemm-rs-1d-stable` | `GEMM_RS_DESIGN.md` / `GEMM_RS_ITERATION.md` / `FLUX_GEMM_RS_STUDY.md` |
| **A2A-transpose-GEMM**（Ulysses post-attn，正确版）| `bf16_a2a_transpose_gemm_nt`(M1融合) / `_m0`(M0串行) | **M0 单节点最优**：比 torch(NCCL+转置)基线快 **1.6~1.8×**，comm 快 3~4×；M1 overlap 单节点 **~parity（净负）** | **6/6 PASS @ {2,4,8}**（vs all_gather 非循环 ground-truth）| 分支 `a2a-transpose-gemm`（未合 main）| `A2A_TRANSPOSE_GEMM_DESIGN.md`（含 flux 口径复核/资源分析/已试已弃实验）|
| **A2A-GEMM**（旧 token-A2A，**语义错误**）| `bf16_a2a_gemm_nt` | — | ⚠️ **3/6 FAIL** + 语义错位（token(M)-A2A，非 Ulysses）→ 已被上面的 transpose 版取代 | `main`（保留未删）| `A2A_GEMM_ITERATION.md` / `A2A_GEMM_DESIGN.md`（旧）|
| **AG-GEMM** | `bf16_ag_gemm_nt` | geo **~1.13x**（8 卡）| PASS | `main`（仅 PDL 默认开启被保留）| `AG_GEMM_ITERATION.md` / `AG_GEMM_FLUX_REFERENCE.md` |

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
