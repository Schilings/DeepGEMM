# DeepGEMM 开发规则（2026-06-18）

> 本文件是当前开发规则的权威版本；旧规则若冲突，以本文件为准。

---

## 1. 核心原则

1. **阶段性改动必须立即 commit + push**（服务器可能中断）。
2. 每次开发前先读：`docs/RULE.md` + `docs/PROGRESS.md`。
3. 开发优先级：
   - P0：`tests/` 多卡正确性
   - P1：`benchmarks/` 可复现性能基线
   - P2：AKO4ALL 性能迭代
4. 所有关键结论必须沉淀到 `docs/PROGRESS.md`。

---

## 2. 当前仓库与环境

- 仓库：`https://github.com/Schilings/DeepGEMM.git`
- 当前主分支：`main`
- 当前机器现实：可能无 `nvcc`，优先 `NVRTC` 路径。

---

## 3. 新会话启动 SOP（必须执行）

1. 读取文档：
   - `docs/RULE.md`
   - `docs/PROGRESS.md`
   - `docs/GEMM_RS_DESIGN.md`
   - `docs/SESSION_MEMORY.md`
2. 加载技能（见第 4 节）
3. 构建 `_C`：
   - `python3 setup.py build_ext --inplace --force`
4. 跑 quick correctness：
   - `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs_quick.py 2`
5. 再进入 benchmark/优化

---

## 4. CodeBuddy 技能规范（必须记录并执行）

### 4.1 必须技能

- `cuda-skill`
- `ako4all`

### 4.2 会话内必须先加载

在任务开始第一步调用：
- `use_skill("cuda-skill")`
- `use_skill("ako4all")`

### 4.3 若 `ako4all` 缺失时自动安装

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后重启会话并重新 `use_skill`。

### 4.4 AKO4ALL 使用约束（本项目）

- 不使用 `solution/` 隔离路径，直接在原文件迭代。
- 不新建优化分支，直接在 `main` 迭代。
- 每轮迭代后必须 `commit + push`。
- 使用项目内测试/基准脚本：
  - `tests/test_gemm_rs.py` / `tests/test_gemm_rs_quick.py`
  - `benchmarks/bench_gemm_rs.py`

---

## 5. 常用命令（当前可用）

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs_quick.py 2

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## 6. 文档维护要求

- 任一阶段完成后，必须更新：
  - `docs/PROGRESS.md`（状态与结论）
  - 必要时更新 `docs/GEMM_RS_DESIGN.md`（方案变化）
  - `docs/SESSION_MEMORY.md`（新会话接班信息）

---

## 7. 重要提醒

- 不要被历史文档中的旧性能结论束缚。
- 以“当前机器可复现结果”作为唯一决策依据。