# DeepGEMM 开发规则（主线版）

> 本文件为当前规则权威来源；如有冲突，以本文件和 `docs/PROGRESS.md` 为准。

---

## 1. 核心纪律

1. 阶段性成果必须立即 **commit + push**。
2. 所有行动先以可复现结果为依据，不依赖历史结论。
3. 仅维护当前唯一主线（不在主流程中并行维护历史版本叙事）。
4. 关键结论必须同步到 `docs/PROGRESS.md`。

---

## 2. 新会话启动 SOP（必须）

1. 先读：
   - `docs/RULE.md`
   - `docs/PROGRESS.md`
   - `docs/GEMM_RS_DESIGN.md`
   - `docs/SESSION_MEMORY.md`
2. 加载技能：
   - `use_skill("cuda-skill")`
   - `use_skill("ako4all")`
3. 构建扩展：
   - `python3 setup.py build_ext --inplace --force`
4. 跑主线正确性：
   - `DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM python tests/test_gemm_rs.py 2`
5. 进入 benchmark 与迭代。

---

## 3. 技能要求（CodeBuddy）

### 必需技能

- `cuda-skill`
- `ako4all`

### 若 `ako4all` 缺失（兜底安装）

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后重启会话并重新 `use_skill`。

---

## 4. 推荐命令

```bash
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

DG_BENCH_MAX_SHAPES=3 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3
```

---

## 5. 文档维护要求

每次推进后必须更新：

- `docs/PROGRESS.md`（状态、数据、结论）
- `docs/GEMM_RS_DESIGN.md`（主线设计变化）
- `docs/SESSION_MEMORY.md`（新会话接班信息）
