# GEMM-RS 会话接班记忆（主线）

> 最后更新：2026-06-18 04:56
> 目标：新会话 5 分钟内无缝接手。

---

## A. 开局顺序（严格执行）

1. 读文档：`RULE.md` → `PROGRESS.md` → `SESSION_MEMORY.md` → `GEMM_RS_DESIGN.md`
2. 加载技能：`cuda-skill` + `ako4all`
3. 构建：`python3 setup.py build_ext --inplace --force`
4. 正确性：`tests/test_gemm_rs.py 2`
5. 性能：`benchmarks/bench_gemm_rs.py`

---

## B. 当前关键事实

- 当前口径是唯一主线：`bf16_gemm_rs_nt`。
- shape 集合已固定为用户指定 13 个，重点 5 个 shape 单独追踪。
- 学习方向：参考 `flux` GEMM-RS（H 卡稳定上线），在 B 卡做策略适配。
- benchmark 脚本支持：
  - `DG_BENCH_FOCUS_ONLY=1`
  - `DG_BENCH_SHAPES="M,N,K;..."`

---

## C. 直接可运行命令

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

MASTER_PORT=29685 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3

MASTER_PORT=29684 DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## D. 当前基线摘要

- 指定 13 shape（2 GPU，3 iter）：**geo mean ≈ 1.102x**
- 重点 5 shape（2 GPU，5 iter）：**geo mean ≈ 1.155x**
- 当前短板：`2048x7168x2048`（约 `0.97x`）
- 重点集最弱点：`4096x4096x7168`（约 `1.04x`，已从约 `1.04x`→`1.06x` 单点改善）

---

## E. 下一步最短路径

1. 继续针对 `K=7168` 场景做定向优化。
2. 每次改动后先跑重点 5 shape，确认主目标集合不退化。
3. 阶段性立即 `commit + push`，避免服务器回收导致进度丢失。