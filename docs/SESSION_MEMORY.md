# GEMM-RS 会话接班记忆（主线）

> 最后更新：2026-06-18 07:18
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
- **【最新】主线已重构为真·Flux pull 式 dual-kernel**（2026-06-18, Iteration 3）：
  - Kernel 1 `sm100_bf16_gemm_rs.cuh`：256T 无 comm warps，epilogue 纯本地 scatter write `slot[dst_rank]` + 本地 flag；
  - Kernel 2 `sm100_rs_reduce.cuh`（`kPullBased=true`）：从各远端 rank pull `slot[R]` 做 FP32 reduce → output，读后远端 reset flag；
  - host `sm100_bf16_gemm_rs.hpp` 双流编排（compute_stream + comm_stream + event）。
  - 旧 push dual-kernel 仍可 `DG_GEMM_RS_IMPL=v3/push` 回退。
- **当前状态**：正确性达标——`test_gemm_rs.py 2` **6/6 PASS, max_diff=0.0**（已修复 nvlink_barrier 死锁：
  移除与对端信号竞争的 per-call barrier memset）。**性能回退**：2-GPU geo_mean ≈ 0.58x（fused 628T vs sep 1065T），
  慢于 push v3(~1.10x)；待把 pull reduce 改成 TMA 流水线 fetch。高性能回退用 `DG_GEMM_RS_IMPL=v3`。
- shape 集合已固定为用户指定 13 个，重点 5 个 shape 单独追踪。
- 学习方向：参考 `flux` GEMM-RS（H 卡稳定上线），在 B 卡做策略适配。
- 主线策略：按 `SM100_2CTA_CLUSTER`，中大 shape 优先 `mc=2`（2-CTA cluster）。
- benchmark 已升级三路基线：
  - `torch.matmul + RS`
  - `deep_gemm.bf16_gemm_nt + RS`
  - `bf16_gemm_rs_nt`
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

- **历史基线为旧 push 路径**（`DG_GEMM_RS_IMPL=v3` 可复现）：
  - 指定 13 shape（2 GPU，3 iter）：geo mean ≈ 1.110x（vs torch） / 1.100x（vs sep）
  - 重点 5 shape（2 GPU，4 iter）：geo mean ≈ 1.174x（vs torch） / 1.161x（vs sep）
  - 短板：`2048x7168x2048` ≈ 0.98x；重点集最弱 `4096x4096x7168` ≈ 1.05x
- **新 pull 路径基线：待跑通正确性后重测**（预期 GEMM 吞吐因消除 384T 寄存器 spilling 而提升）。

---

## E. 下一步最短路径

1. **排查 pull 路径 2-GPU 运行期错误**（首屏真实报错：kernel assert / nvlink barrier timeout / IMA），跑通 `test_gemm_rs.py 2`。
2. 跑通后跑 `bench_gemm_rs.py` 重测 pull 基线，与 push(v3) 对比。
3. 继续针对 `K=7168` 场景定向优化；每次改动先跑重点 5 shape 防回退。
4. 阶段性立即 `commit + push`，避免服务器回收导致进度丢失。