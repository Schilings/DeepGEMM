# DeepGEMM 开发规则（通用主线规则 · 算子无关）

> 本文件是**算子无关**的开发规则权威版本，约束 AI 在本仓库的所有迭代行为。
> 每次会话由**用户指定目标算子**（如 GEMM-RS / AG-GEMM / A2A-GEMM），AI 基于该算子的
> 文档（见 §3 算子文档地图）无缝接班并继续开发。
> 冲突时优先级：本文件 > 指定算子的 `*_ITERATION.md` 顶部「当前状态」> 其它文档。

---

## 1. 核心原则（所有算子通用）

1. **阶段性改动必须立即 commit + push**（防止会话/机器中断导致成果丢失）。
2. 所有判断以**当前机器可复现结果**为准，不用历史结论替代实测。
3. **单会话聚焦单一目标算子**：由用户指定，所有迭代围绕该算子的入口符号与文档进行，
   不擅自切换或并行改动其它算子。
4. 开发优先级：
   - P0：`tests/` 多卡正确性稳定（绝不退化）
   - P1：`benchmarks/` 可复现基线
   - P2：性能迭代（含 AKO4ALL）
5. 关键结论必须沉淀到该算子的 `*_ITERATION.md`，并在 `docs/PROGRESS.md`（进度总览）更新一行状态。
6. **优化目标紧贴 Megatron SP**：通信类融合算子优先围绕中大 shape（重点 `M/rank>=1024` 且
   `N/K` 为 `4096/7168` 组合）做主线迭代，以中大 shape 的几何均值为主要决策依据。
7. **学习方向明确**：以 `flux`（H 卡稳定上线）作为方法学参考；本仓库在 B 卡（SM100）上做等价
   策略适配与验证，不做生硬照搬。
8. **改动隔离**：高风险大改在独立分支开发，`main` 始终保留已验证可用版本；被取代的稳定版用
   `git tag` 存档（零信息损失），不在 `main` 上并行维护多套命名实现。

---

## 2. 当前仓库与环境

- 仓库：`https://github.com/Schilings/DeepGEMM.git`
- 主分支：`main`
- 当前环境可能无 `nvcc`：优先用 `nvcc`，没有则安装；安装不了再默认走 NVRTC：`DG_JIT_USE_NVRTC=1`
- 目标硬件：SM100（B 系列），单机 NVLink 域（≤8 卡）。

---

## 3. 算子文档地图（用户指定算子 → 读对应文档）

每个算子是一组「设计 + 迭代 + 参考」文档。**会话开局：用户指定算子后，按下表读取该算子全部文档**
（顺序：`*_DESIGN` → `*_ITERATION` 顶部「当前状态」→ 参考文档）。

| 算子 | 入口符号 | 设计 | 迭代记录（含当前状态 / 接班） | 参考 / 学习 | test / bench |
|------|---------|------|------|------|------|
| **GEMM-RS** | `bf16_gemm_rs_nt` | `GEMM_RS_DESIGN.md` | `GEMM_RS_ITERATION.md` | `FLUX_GEMM_RS_STUDY.md` | `tests/test_gemm_rs.py` / `benchmarks/bench_gemm_rs.py` |
| **AG-GEMM** | `bf16_ag_gemm_nt` | （见 ITERATION 背景节） | `AG_GEMM_ITERATION.md` | `AG_GEMM_FLUX_REFERENCE.md` | `tests/test_ag_gemm.py` / `benchmarks/bench_ag_gemm.py` |
| **A2A-GEMM** | `bf16_a2a_gemm_nt` | `A2A_GEMM_DESIGN.md` | `A2A_GEMM_ITERATION.md` | （见 DESIGN 节） | `tests/test_a2a_gemm.py` / `benchmarks/bench_a2a_gemm.py` |

**通用知识（所有算子都应吸收）**：
- `docs/PROGRESS.md`：**算子进度总览 / 索引** —— 先看这里挑选/确认目标算子的当前状态与分支/tag。
- `docs/SESSION_MEMORY.md`：**通用接班 SOP**（与算子无关的开局流程）。
- `docs/SM100_2CTA_CLUSTER.md`：SM100 2-CTA cluster / multicast / scheduler 必读知识（见 §3.1）。

### 3.1 SM100 2-CTA cluster 关键知识（通用，写 cluster/multicast 相关代码前必读）

- 2-CTA cluster（`cluster_m=2`）中两个 CTA 应满足：**相邻 M-tile、相同 N-tile**。
- 正确调度前提：两个 CTA 必须走不同 `blockIdx.x`，从而拿到不同 `m_block_idx`。
- `kIsMulticastOnA=false` 场景下：
  - A：各 CTA 加载不同 M 行（不需要额外 m 偏移）；
  - B：各 CTA 按 `block_rank_in_cluster()` 加载一半 N 列。
- 2SM UMMA 由 leader CTA 发射，Epilogue 仍由两个 CTA 各自独立写回自己的 128 行。
- 若出现 `multicast=2` 错误 / hang，优先排查 scheduler 是否错误复用了 `cluster_idx` 导致双 CTA 拿到同一 `m_block_idx`。

---

## 4. 新会话启动 SOP（必须执行）

1. **确认目标算子**（用户指定；若未指定，先读 `docs/PROGRESS.md` 与用户确认再开工）。
2. 读取通用规则与该算子文档：
   - `docs/RULE.md`（本文件）→ `docs/PROGRESS.md`（总览）→ `docs/SESSION_MEMORY.md`
   - 目标算子的 `*_DESIGN.md` → `*_ITERATION.md`（顶部「当前状态」）→ 参考文档
   - 涉及 cluster/multicast：`docs/SM100_2CTA_CLUSTER.md`
3. 加载技能（见 §5）。
4. 编译 `_C`：`python3 setup.py build_ext --inplace --force`
5. 跑目标算子正确性：`DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python tests/test_<op>.py 2`
6. 跑目标算子 benchmark 建立基线，再进入优化迭代。

---

## 5. CodeBuddy 技能规范（必须记录并执行）

### 5.1 必需技能

- `cuda-skill`
- `ako4all`

### 5.2 会话开始即加载

任务第一步执行：`use_skill("cuda-skill")` 与 `use_skill("ako4all")`。

### 5.3 `ako4all` 缺失时自动安装（兜底）

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后重启会话，再次执行 `use_skill`。

### 5.4 AKO4ALL 在本项目中的约束

- 不使用 `solution/` 隔离路径，直接在原文件迭代（**高风险大改例外**：按 §1.8 用独立分支）。
- 每轮有效改动后都要 `commit + push`。
- 使用**目标算子对应的** test/bench 验证（见 §3 表）。

---

## 6. 常用命令（以 GEMM-RS 为例；其它算子替换为对应脚本名）

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force

# 目标算子正确性（脚本名按 §3 表替换：test_gemm_rs / test_ag_gemm / test_a2a_gemm）
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

# Megatron SP 中大 shape 基线（focus 子集）
DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 8 8

# 指定单 shape 诊断
DG_BENCH_SINGLE_SHAPE=4096,7168,4096 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 8 8
```

---

## 7. 文档维护要求

任一阶段完成后必须更新：

- 目标算子的 `*_ITERATION.md`：新增 Iteration 条目 + 更新顶部「当前状态」摘要。
- `docs/PROGRESS.md`：该算子的一行状态（最新结果 / 分支 / tag）。
- 必要时更新该算子的 `*_DESIGN.md`（设计变化）/ `docs/SESSION_MEMORY.md`（通用流程变化）。

### 7.1 迭代必须记录 benchmark 数据（强制）

每轮迭代在该算子的 `*_ITERATION.md` 对应 Iteration 条目里，**必须记录实测 benchmark 数据**，覆盖：

- **不同 GPU 数**：至少 4 卡与 8 卡（中大 shape 真实场景），有条件再附 2 卡；
- **不同 shape**：至少给出 focus 中大 shape 集合的逐 shape speedup（vs torch / vs sep），
  以及整体 geo_mean 与分组（N=7168 / K=7168 / M/rank≥2048 等）；
- 记录格式：表格（GPUs × shape × {vs torch, vs sep, fused TFLOPS}）或等价清单，必须能与上一轮直接对比；
- **出现性能回退必须回退该改动**（git）并在条目中写明回退原因。

> 口径：以 4/8 卡中大 shape 的几何均值为主要决策依据；极端 shape（如 M/rank=16384 且 N=K=7168）暂不作为主目标。

---

## 8. 重要提醒

- 历史/归档文档（`docs/archive/`）仅作参考，不作为当前实现判断依据。
- 不在主文档中并行维护「多版本命名叙事」；被取代的实现用 `git tag` 存档（见 §1.8）。
- 若出现阻塞，优先构造最小可复现案例，再做结构性调整。
- 涉及 SM100 cluster / multicast / scheduler 改动时，必须回看 `docs/SM100_2CTA_CLUSTER.md`（§3.1）再下改动。
- 涉及通信类融合算子的主线优化策略时，优先对齐 `flux` 的稳定实践，再做 B 卡可行性映射验证。
