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
  - **push 路径已删除**：Flux 单机 RS 即 pull，故移除旧 `v3`/`gemm_rs_compute`/`sm100_bf16_gemm_rs_compute.*`
    及其 test/bench；主线唯一 = pull，无 `DG_GEMM_RS_IMPL` 开关。
- **当前状态**：正确性达标——`test_gemm_rs.py 2` **6/6 PASS, max_diff=0.0**（已修复 nvlink_barrier 死锁：
  移除与对端信号竞争的 per-call barrier memset）。**性能待优化**：2-GPU geo_mean ≈ 0.58x（fused 628T vs sep 1065T）；
  下一步把 pull reduce 改成 TMA 流水线 fetch（Flux `Sm90ReduceScatterDma` 风格）。
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

## D. 当前基线摘要（融合 GEMM-RS：epilogue push-scatter + 本地连续 reduce）

- 正确性：`test_gemm_rs.py 2` 6/6 PASS，max_diff=0.0。
- 性能：**跨 2/4/8 GPU 稳定 ~1.14x vs sep**（2 卡 1.142x / 4 卡 1.148x / 8 卡 1.140x；
  focus 中大 1.14–1.20x；avg fused 1193–1232T）。正确性 6/6 PASS @ {2,4,8} 卡。已超历史 push-v3(1.10x)。
  reduce kUnroll 随 rank 数自适应（`kNumRanks>=8?2:>=4?4:8`）防 8 卡寄存器 spilling。
- 架构（关键）：**跨卡 NVLink 传输放进 GEMM epilogue 用 TMA async store、与 MMA 重叠**（这是单机
  >1.0x 的唯一可行路径——分离 reduce kernel 因 GEMM 独占 SM 寄存器无法共驻 overlap，实测零和）；
  **reduce 改为纯 1D 连续流式本地累加**（去 flag/poll/syncthreads，可见性靠末尾 system-scope nvlink_barrier）。
- 提速路径：Iter5 高 MLP → Iter7 reduce grid 过订阅 ×2 → Iter8 epilogue push-scatter → Iter9 去冗余 CPU 同步
  → **Iter10 去 flag + 连续流式 reduce（决定性）**。
- 关键死胡同（勿重试）：SM carveout（`DG_RS_REDUCE_SMS`，零和，默认 0）；smem reserve 让 reduce 共驻（GEMM 208-reg epilogue 占满，须砍寄存器→spilling）。
- 下一步：4/8-GPU 扩展验证；超大 N·K（16384x7168x7168 1.02x）的 push 调度优化。

---

## E. 下一步最短路径

1. **性能优化（核心）**：把 pull RS reduce 从「朴素标量 P2P 读」改造为 **TMA 流水线 fetch+reduce**
   （远端→smem 的 producer/consumer，对齐 Flux `Sm90ReduceScatterDma`）；并优化 GEMM/reduce 的 SM 划分与重叠。
2. 每次改动跑 `test_gemm_rs.py 2`（正确性）+ `bench_gemm_rs.py 2 3`（性能）。
3. 阶段性立即 `commit + push`，避免服务器回收导致进度丢失。