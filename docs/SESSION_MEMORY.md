# DeepGEMM GEMM-RS 开发会话记忆

> **最后更新**: 2026-06-09
> **当前分支**: `main`
> **环境**: 开发在 1× B300 SXM6 上完成，需多卡环境测试
> **GitHub**: https://github.com/Schilings/DeepGEMM.git
> **认证**: token 已嵌入 remote URL（用户 schilings, 邮箱 1146830743@qq.com）

---

## 📌 项目概述

DeepGEMM 的 **GEMM-RS (GEMM + Reduce-Scatter)** 融合 kernel，目标是在多 GPU NVLink 互联环境下，将 GEMM 计算与 ReduceScatter 通信重叠，实现 MoE 推理中的通信掩盖。

### 当前状态

- **设计方案**: Pull-based 单 kernel，tile 级 overlap
- **代码状态**: 已写完，待多卡测试验证
- **之前的尝试（Push + PDL 两阶段）**: 已验证性能不行（8GPU 仅 0.21x NCCL），代码已删除，仅留旧文档作为参考

---

## 🚀 当前方案概要

### 设计灵感来源

1. **ByteDance Flux** — Pull 模式 + Tile 粒度 overlap + per-tile barrier
2. **DeepSeek MegaMoe** — Persistent kernel + Warp 功能分化 + SM100 TMA 模式

### 核心设计

| 维度 | 说明 |
|------|------|
| Kernel 数量 | **1** (全融合) |
| 通信方向 | **Pull (读远端)** |
| 通信模型 | **All-to-1 Pull O((N-1)/N)**, bandwidth-optimal |
| Overlap 粒度 | **Tile 级流水线** |
| 同步模型 | **Per-tile ready flag** (st_rel_sys / ld_acq_sys) |

### Warp 分工 (320 threads = 10 warps)

```
W0 (32T): TMA Load — 加载 A/B tiles 到 SMEM
W1 (32T): MMA Issue — UMMA FMA → TMEM accumulator
W2-3 (64T): Epilogue — TMEM → smem → local partial buffer + per-tile ready flag
W4-7 (128T): Comm — Pull-based Reduce-Scatter
             - Poll per-tile ready flags from ALL ranks (ld_acq_sys)
             - NVLink P2P Read (pull remote partial)
             - FP32 accumulate → write final output
```

### 关键算法

1. **M 维 Swizzle**: Rank i 优先计算 rank(i+1) 的 chunk → 使接收端能尽早开始拉取
2. **Per-tile Ready Flag**: Epilogue 完成一个 tile 后 `st_rel_sys(flag)` 通知远端
3. **Comm Warps 自旋 Pull**: `ld_acq_sys(flag)` 检测就绪 → P2P Read → FP32 reduce → 写 output
4. **单 Kernel 完整融合**: 计算、通信、reduce 全在一个 kernel 中完成

---

## 📊 之前尝试的结果（Push + PDL 方案，已废弃）

那个方案的核心问题：
- **通信模型 = Symmetric Push O(N)**: 总通信量是 NCCL ring 的 N 倍
- **无真正 overlap**: 全部 GEMM 完成后才做一次 nvlink_barrier → 再启 reduce kernel
- **8 GPU 全面落后**: Geo mean 仅为 NCCL 方案的 0.21x
- **结论：仅在 2GPU + 极小 batch 下有微弱优势，完全不实用**

详细数据见 [BENCHMARK_GEMM_RS.md](./BENCHMARK_GEMM_RS.md)。

---

## ⏭️ 下一步工作（多卡环境）

### 优先级 P0：测试正确性

```bash
# 清除 JIT 缓存
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# 2 GPU 正确性测试
python tests/test_gemm_rs.py 2

# 8 GPU 正确性测试
python tests/test_gemm_rs.py 8
```

**预期**：pull + reduce 应该得到与 `bf16_gemm_nt + nccl_reduce_scatter` 相同的结果（允许 FP32 累加误差）。

### 优先级 P1：Benchmark

```bash
# 性能对比 (Fused vs GEMM+NCCL分离)
python benchmarks/bench_gemm_rs.py 2 20
python benchmarks/bench_gemm_rs.py 4 20
python benchmarks/bench_gemm_rs.py 8 20
```

### 优先级 P2：根据测试结果调优

可能的问题和解决方向：
1. **Comm Warps 带宽不足** → 增加 Comm Warps 数量或使用 TMA Load 代替手动 P2P Read
2. **Per-tile Flag 延迟过高** → 调整 flag 粒度（多个 tile 合一个 flag）
3. **GEMM 算力下降** → Warp 分配比例调优
4. **死锁/hang** → 检查 barrier 逻辑和 M-Swizzle 调度顺序
5. **数值精度** → FP32 reduce 路径验证

### 优先级 P3：进阶优化

- **TMA Load for Pull**: 用 TMA 硬件异步拉取代替手动 global load
- **Ring 多步流水线**: 参考 NCCL ring 的多步 reduce 降低延迟
- **FP8 版本**: 扩展到 FP8 输入

---

## 📁 关键文件路径

```
=== 核心实现 ===
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh      # 核心 kernel
csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp                 # JIT runtime
csrc/jit_kernels/heuristics/gemm_rs.hpp                       # get_gemm_rs_config()
csrc/apis/gemm_rs.hpp                                         # C++ API: bf16_gemm_rs_nt()
deep_gemm/gemm_rs/__init__.py                                 # Python API: bf16_gemm_rs_nt()
tests/test_gemm_rs.py                                         # 正确性测试
benchmarks/bench_gemm_rs.py                                   # Benchmark (vs GEMM+NCCL分离)

=== 公共基础设施 ===
deep_gemm/include/deep_gemm/comm/barrier.cuh                  # nvlink_barrier / grid_sync
deep_gemm/include/deep_gemm/layout/gemm_rs.cuh               # GemmRSWorkspace 布局
deep_gemm/include/deep_gemm/layout/sym_buffer.cuh            # SymmetricBuffer (NVLink 映射)
deep_gemm/include/deep_gemm/ptx/ptx.cuh                      # PTX 内联 (st_rel_sys, ld_acq_sys 等)
```

---

## 🔄 Git 提交历史

```
035d750 docs: update all docs for V2 development continuity
b18642e feat(gemm-rs): Add V2 pull-based single-kernel GEMM+RS fusion  ← 当前方案
33520d5 feat(fp8_gemm_rs): apply Plan B to FP8 kernel + add SKIP_FP8 bench option
8054773 feat(gemm_rs): Plan B - remove per-tile fence, add kernel-end barrier + dynamic config
80bdb23 bench: add 2/4/8 GPU results to GEMM-RS benchmark report
91825a6 bench: add GEMM-RS benchmark script and performance report
```

---

## 💡 运行命令速查

```bash
# 安装（开发模式）
cd /root/.local/codebuddy/DeepGEMM
git submodule update --init --recursive
pip install -e . --no-build-isolation

# ===== 测试 =====
python tests/test_gemm_rs.py 2       # 正确性 (2 GPU)
python tests/test_gemm_rs.py 8       # 正确性 (8 GPU)
python benchmarks/bench_gemm_rs.py 2 20   # Benchmark (2 GPU)
python benchmarks/bench_gemm_rs.py 8 20   # Benchmark (8 GPU)

# ===== 清除 JIT 缓存 =====
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# ===== Git 操作 =====
cd /root/.local/codebuddy/DeepGEMM
git add -A && git commit -m "描述" && git push origin main
```

---

## ⚙️ 环境信息

- **目标 GPU**: 8× NVIDIA B300 SXM6 (NVLink 互联)
- **开发 GPU**: 1× B300 SXM6（仅编译验证，无法多卡测试）
- **架构**: SM100 (Blackwell)
- **Python 包**: `deep_gemm` (editable install from setup.py)
- **JIT 缓存**: `~/.deep_gemm/cache/`
- **第三方依赖**: CUTLASS + fmt (git submodule)

---

## 🧠 设计决策备忘

### 为什么选 Pull（而非 Push）？

1. **天然适配 Tile 级 Overlap**: 接收端看到一个 tile 就绪就拉过来 reduce，不需等全部
2. **SM100 TMA Load 更高效**: Blackwell 的 TMA Load 从远端读是硬件异步的
3. **Reduce 融入 kernel**: Pull 回来在 SMEM 中，直接寄存器 FP32 累加，无需额外 kernel
4. **Bandwidth-optimal**: 每个 rank 从 N-1 个 peer 各拉 1/N，等同 NCCL ring RS

### 参考项目

- **Flux (ByteDance)**: `/root/.local/codebuddy/flux/` — SM90 Pull-based GEMM+RS
  - 关键文件: `src/gemm_rs/sm90_reduce_scatter_utils.hpp`
  - 核心: Fetch warp TMA Load from peer → Reduce warp 本地累加
  - 差异: Flux 是 Hopper(SM90)，我们是 Blackwell(SM100)

- **MegaMoe (DeepSeek)**: `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
  - Persistent kernel + Warp 功能分化
  - SM100 TMA + UMMA 模式的最佳实践参考

---

## ⚠️ 已知风险和注意事项

1. **当前 kernel 未经多卡验证** — 需要在 2+ GPU 环境测试正确性
2. **Per-tile flag 跨 NVLink 延迟** — 如果 ld_acq_sys 自旋成本高，考虑批量 flag
3. **320 线程 = 10 warps** — 可能影响 SM 占用率，需 profiling
4. **M-Swizzle 调度** — 如果所有 CTA 同时写同一个远端 rank 的 flag 可能造成热点
5. **comm_dtype** — 目前支持 BF16/FP32 comm，默认 BF16
