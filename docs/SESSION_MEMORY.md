# 会话接班记忆（通用 SOP · 算子无关）

> 最后更新：2026-06-22
> 目标：新会话 5 分钟内无缝接手**用户指定的目标算子**。
> 本文件只写**与算子无关的开局流程**；各算子的具体当前状态见对应 `*_ITERATION.md` 顶部「当前状态」。

---

## A. 开局顺序（严格执行）

1. **确认目标算子**（用户指定，如 GEMM-RS / AG-GEMM / A2A-GEMM）。若未指定，先读 `docs/PROGRESS.md` 与用户确认。
2. 读文档（顺序）：
   - `RULE.md`（通用规则）→ `PROGRESS.md`（进度总览，定位目标算子状态/分支/tag）→ 本文件
   - 目标算子：`*_DESIGN.md` → `*_ITERATION.md`（顶部「当前状态」）→ 参考文档（见 `RULE.md` §3 算子文档地图）
   - 涉及 cluster/multicast：`SM100_2CTA_CLUSTER.md`
3. 加载技能：`cuda-skill` + `ako4all`
4. 构建：`python3 setup.py build_ext --inplace --force`
5. 正确性：`DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python tests/test_<op>.py 2`
6. 性能：`benchmarks/bench_<op>.py` 建立基线，再进入迭代。

---

## B. 通用关键事实

- **单会话聚焦单一算子**（见 `RULE.md` §1.3），不擅自切换/并行改动其它算子。
- 入口符号、test/bench 脚本、设计/迭代文档的对应关系见 `RULE.md` §3「算子文档地图」。
- benchmark 三路基线惯例：`torch.matmul + 通信原语` / `deep_gemm 标准 GEMM + 通信原语` / 该算子融合入口。
- 学习方向：参考 `flux`（H 卡稳定上线），在 B 卡（SM100）做策略适配，不生硬照搬。
- 主线策略：按 `SM100_2CTA_CLUSTER`，中大 shape 优先 `mc=2`（2-CTA cluster）。
- 高风险大改在独立分支开发，main 保留已验证版；被取代的稳定版用 `git tag` 存档。

---

## C. 直接可运行命令（以 GEMM-RS 为例，替换脚本名即可用于其它算子）

```bash
cd /root/.local/codebuddy/DeepGEMM
git pull
python3 setup.py build_ext --inplace --force

DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

DG_BENCH_FOCUS_ONLY=1 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 8 8
```

---

## D. 收尾（每轮迭代后）

1. 跑该算子 `test_<op>.py`（正确性，绝不退化）+ `bench_<op>.py`（性能，记录 4/8 卡 focus 数据）。
2. 更新该算子 `*_ITERATION.md`（新 Iteration 条目 + 顶部「当前状态」）+ `PROGRESS.md` 一行状态。
3. 阶段性立即 `commit + push`，避免服务器回收导致进度丢失。
