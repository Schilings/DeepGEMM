# GEMM-RS 会话接班记忆（新会话必读）

> 最后更新：2026-06-18 04:16
> 目的：让新会话 5 分钟内无缝接手继续开发。

---

## A. 先做什么（严格顺序）

1. 读取：`docs/RULE.md`、`docs/PROGRESS.md`、`docs/GEMM_RS_DESIGN.md`
2. 加载技能：`cuda-skill` + `ako4all`
3. 构建：`python3 setup.py build_ext --inplace --force`
4. 跑 quick correctness（2 GPU）
5. 进入 benchmark 稳定化与性能迭代

---

## B. 当前关键事实

- 代码已推送到 `origin/main`，关键新提交：`05a4716`。
- 无 `nvcc` 环境下需使用：`DG_JIT_USE_NVRTC=1`。
- `tests/test_gemm_rs_quick.py 2` 已 PASS。
- `benchmarks/bench_gemm_rs.py` 仍有 `unspecified launch failure`（多出现在 barrier/析构阶段）。

---

## C. 新会话直接可执行命令

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs_quick.py 2

DG_BENCH_SINGLE_SHAPE=256,512,1024 DG_BENCH_SYNC_EACH_ITER=1 \
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## D. CodeBuddy 技能要求（必须记录）

### 必需技能

- `cuda-skill`：CUDA/PTX/SM100 诊断与优化
- `ako4all`：自动化性能迭代

### 在 CodeBuddy 中的使用要求

新会话开始后，先让 agent 执行：
- `use_skill("cuda-skill")`
- `use_skill("ako4all")`

### 若 `ako4all` 未安装（兜底）

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后重启会话再加载技能。

---

## E. 当前最短路径任务

1. 稳定单 shape benchmark（2 GPU）
2. 扩到多 shape 小样本
3. 记录稳定基线
4. 再开启 AKO4ALL 迭代

---

## F. 开发纪律

- 每个阶段性改动立即 `commit + push`
- 每次推进同步更新 `docs/PROGRESS.md`
- 出现阻塞先定位可复现最小案例，不盲目大改