# GEMM-RS 会话接班记忆（主线）

> 最后更新：2026-06-18 04:28
> 目标：新会话 5 分钟内无缝接手。

---

## A. 开局顺序（严格执行）

1. 读文档：`RULE.md` → `PROGRESS.md` → `GEMM_RS_DESIGN.md`
2. 加载技能：`cuda-skill` + `ako4all`
3. 构建：`python3 setup.py build_ext --inplace --force`
4. 正确性：`tests/test_gemm_rs.py 2`
5. 性能：`benchmarks/bench_gemm_rs.py`

---

## B. 当前关键事实

- 当前口径是唯一主线：`bf16_gemm_rs_nt`。
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

DG_BENCH_MAX_SHAPES=3 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3
```

---

## D. 当前基线摘要

- 单 shape `256,512,1024`：fused ≈ **1.53x**
- 前 3 shape：geo mean ≈ **1.038x**

---

## E. 下一步最短路径

1. 扩展到 5~8 shape，固定稳定小样本基线。
2. 针对主线算子做参数/调度迭代。
3. 每轮结束后立即更新 `PROGRESS.md` 并 `commit + push`。
