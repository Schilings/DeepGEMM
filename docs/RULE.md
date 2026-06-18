# DeepGEMM 开发规则（详细主线版）

> 本文件是当前开发规则的权威版本；若与其他文档冲突，以本文件和 `docs/PROGRESS.md` 为准。

---

## 1. 核心原则

1. **阶段性改动必须立即 commit + push**（防止会话/机器中断导致成果丢失）。
2. 所有判断以**当前机器可复现结果**为准，不用历史结论替代实测。
3. 仅维护**唯一主线口径**：`bf16_gemm_rs_nt`。
4. 开发优先级：
   - P0：`tests/` 多卡正确性稳定
   - P1：`benchmarks/` 可复现基线
   - P2：性能迭代（含 AKO4ALL）
5. 关键结论必须沉淀到 `docs/PROGRESS.md`。
6. **优化目标紧贴 Megatron SP**：优先围绕中大 shape（重点 `M/rank>=1024` 且 `N/K` 为 `4096/7168` 组合）做主线迭代。
7. **学习方向明确**：以 `flux` 的 GEMM-RS（H 卡稳定上线）作为方法学参考；本仓库在 B 卡上做等价策略适配与验证，不做生硬照搬。

---

## 2. 当前仓库与环境

- 仓库：`https://github.com/Schilings/DeepGEMM.git`
- 主分支：`main`
- 当前环境可能无 `nvcc`，用的话优先用`nvcc`，没有就安装，安装不了再默认走 NVRTC：`DG_JIT_USE_NVRTC=1`

---

## 3. 主线知识必读与吸收（必须执行）

### 3.1 必读文档（会话开局强制）

按以下顺序阅读并吸收：

1. `docs/RULE.md`
2. `docs/PROGRESS.md`
3. `docs/SESSION_MEMORY.md`
4. `docs/GEMM_RS_DESIGN.md`
5. `docs/SM100_2CTA_CLUSTER.md`

### 3.2 必须吸收的关键知识点（写代码前先对齐）

#### A) 来自 `SM100_2CTA_CLUSTER.md`

- 2-CTA cluster（`cluster_m=2`）中，两个 CTA 应满足：**相邻 M-tile、相同 N-tile**。
- 正确调度前提：两个 CTA 必须走不同 `blockIdx.x`，从而拿到不同 `m_block_idx`。
- `kIsMulticastOnA=false` 场景下：
  - A：各 CTA 加载不同 M 行（不需要额外 m 偏移）
  - B：各 CTA 按 `block_rank_in_cluster()` 加载一半 N 列。
- 2SM UMMA 由 leader CTA 发射，Epilogue 仍由两个 CTA 各自独立写回自己的 128 行。
- 若出现 `multicast=2` 错误/hang，优先排查 scheduler 是否错误复用了 `cluster_idx` 导致双 CTA 拿到同一 `m_block_idx`。

#### B) 来自 `SESSION_MEMORY.md`

- 当前唯一口径：`bf16_gemm_rs_nt`。
- 新会话标准开局：读文档 → 加载 `cuda-skill` + `ako4all` → build `_C` → 跑 `test_gemm_rs.py 2` → 跑 `bench_gemm_rs.py`。
- 其它算子健康性已验证（A2A/AG），主线 GEMM-RS 正确性已通过，benchmark 已收敛到主线对比路径。

#### C) 来自 `PROGRESS.md`

- 当前基线与结论是运行入口：先看“已验证通过”和“主线 benchmark”小结，再决定下一轮优化方向。
- benchmark 口径固定为：
  - `separate = bf16_gemm_nt + reduce_scatter_tensor`
  - `main fused = bf16_gemm_rs_nt`
- 迭代时必须先保证 correctness 不退化，再比较 fused vs separate 的增益。
- Megatron SP 导向下优先看中大 shape 分组（`N=7168` / `K=7168` / `M/rank>=2048`）的几何均值和短板 shape。

---

## 4. 新会话启动 SOP（必须执行）

1. 按第 3 节顺序读取文档并吸收关键知识点。
2. 加载技能（见第 5 节）。
3. 编译 `_C`：
   - `python3 setup.py build_ext --inplace --force`
4. 先跑主线正确性：
   - `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
5. 再进入 benchmark 与优化迭代。

---

## 5. CodeBuddy 技能规范（必须记录并执行）

### 5.1 必需技能

- `cuda-skill`
- `ako4all`

### 5.2 会话开始即加载

任务第一步执行：
- `use_skill("cuda-skill")`
- `use_skill("ako4all")`

### 5.3 `ako4all` 缺失时自动安装（兜底）

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后重启会话，再次执行 `use_skill`。

### 5.4 AKO4ALL 在本项目中的约束

- 不使用 `solution/` 隔离路径，直接在原文件迭代。
- 不新建优化分支，直接在 `main` 上迭代。
- 每轮有效改动后都要 `commit + push`。
- 使用项目内脚本验证：
  - `tests/test_gemm_rs.py` / `tests/test_gemm_rs_quick.py`
  - `benchmarks/bench_gemm_rs.py`

---

## 6. 常用命令（当前推荐）

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force

# 主线正确性
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

# Megatron SP中大shape基线（主线推荐）
DG_BENCH_MAX_SHAPES=10 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5

# 指定单shape诊断
DG_BENCH_SINGLE_SHAPE=2048,7168,2048 DG_BENCH_SYNC_EACH_ITER=1 \
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 8
```

---

## 7. 文档维护要求

任一阶段完成后必须更新：

- `docs/PROGRESS.md`（状态、数据、结论）
- 必要时更新 `docs/GEMM_RS_DESIGN.md`（主线设计变化）
- `docs/SESSION_MEMORY.md`（新会话接班信息）

---

## 8. 重要提醒

- 历史文档仅作参考，不作为当前实现判断依据。
- 不在主文档中并行维护“多版本命名叙事”，避免新会话误解。
- 若出现阻塞，优先构造最小可复现案例，再做结构性调整。
- 涉及 SM100 cluster / multicast / scheduler 改动时，必须回看 `docs/SM100_2CTA_CLUSTER.md` 再下改动。
- 涉及 Megatron SP 主线优化策略时，优先对齐 `flux` GEMM-RS 的稳定实践，再做 B 卡可行性映射验证。