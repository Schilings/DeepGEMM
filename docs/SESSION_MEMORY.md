# GEMM-RS 会话接班记忆（主线）

> 最后更新：2026-06-18 04:38
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
- 项目目标已明确对齐 **Megatron SP 中大 shape**（`1024/2048/4096` + `N/K=4096/7168` 组合）。
- 学习方向：参考 `flux` GEMM-RS（H 卡稳定上线）的成熟策略，在 B 卡上做可复现适配。
- 其它算子健康：
  - `test_a2a_gemm.py 2` 通过（6/6）
  - `test_ag_gemm.py 2`（限 1 shape）通过（1/1）
- 主线 GEMM-RS 正确性通过：`test_gemm_rs.py 2`（6/6）
- benchmark 脚本已收敛为主线对比（separate vs main fused）。

---

## C. 直接可运行命令

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

DG_BENCH_MAX_SHAPES=10 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 5
```

---

## D. 当前中大 shape 基线摘要

- 中大 shape（前 10 shapes，2 GPU，5 iter）：
  - **geo mean ≈ 1.076x**
  - **best ≈ 1.18x**（`4096x7168x4096`）
  - **worst ≈ 0.96x**（`2048x7168x2048`）
  - 平均 TFLOPS：fused `1102.5T` vs separate `1018.5T`

---

## E. 下一步最短路径

1. 先盯 `2048x7168x2048`（当前短板）做单点优化。
2. 再回归中大 shape 集合复测，观察几何均值是否持续抬升。
3. 每轮结束后立即更新 `PROGRESS.md` 并 `commit + push`。
4. 优先验证可从 `flux` 迁移的稳定优化思想（尤其调度与通信重叠）。