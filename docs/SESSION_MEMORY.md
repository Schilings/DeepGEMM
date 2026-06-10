# DeepGEMM GEMM-RS 开发会话记忆

> **最后更新**: 2026-06-10
> **当前分支**: `main`
> **环境**: 开发在 1× B300 SXM6 上完成，需多卡环境测试
> **GitHub**: https://github.com/Schilings/DeepGEMM.git
> **认证**: token 已嵌入 remote URL（用户 schilings, 邮箱 1146830743@qq.com）

---

## 📌 项目概述

DeepGEMM 的 **GEMM-RS (GEMM + Reduce-Scatter)** 融合 kernel，目标是在多 GPU NVLink 互联环境下，将 GEMM 计算与 ReduceScatter 通信重叠，实现大模型训练中的通信掩盖。

**重点目标场景**：
- 大模型训练的长上下文场景（M_per_rank = 2048~8192）
- 大 hidden dimension（N = 7168，如 DeepSeek-V3）
- 单机多卡（2/4/8 GPU NVLink 互联）

### 当前状态（2026-06-10 更新）

- **设计方案**: Pull-based 单 kernel，tile 级 overlap
- **代码状态**: multicast=1 已验证通过（8GPU 正确性 + benchmark），multicast=2 代码修复已完成，待多卡验证
- **之前的尝试（Push + PDL 两阶段）**: 已验证性能不行（8GPU 仅 0.21x NCCL），代码已删除
- **测试和 Benchmark**: 已完善，覆盖多种 shape 和大 hidden dim 场景
- **已修复的 Bug**: reg_dealloc 死锁、partial buffer slot 寻址、本地 ready flag race、multicast=2 scheduler
- **当前阻塞**: 多 GPU 测试环境端口冲突（`torch.distributed` init 失败），multicast=2 验证待端口问题解决

---

## 🚀 当前方案概要

### 设计灵感来源

1. **ByteDance Flux** — Pull 模式 + Per-rank Pipelined Reduce + Per-tile Flag
2. **DeepSeek MegaMoe** — Persistent kernel + Warp Specialization + SM100 TMA/UMMA 模式

### 核心设计

| 维度 | 说明 |
|------|------|
| Kernel 数量 | **1** (全融合) |
| 通信方向 | **Pull (读远端)** |
| 通信模型 | **All-to-1 Pull O((N-1)/N)**, bandwidth-optimal |
| Overlap 粒度 | **Tile 级流水线** |
| 同步模型 | **Per-tile ready flag** (st_rel_sys / ld_acq_sys) |
| 线程数 | **384** (12 warps) |
| Cluster size | **2** (2-CTA UMMA + TMA multicast) |

### Warp 分工 (384 threads = 12 warps)

```
W0-W3 (128T, 48 regs): Comm — Pull-based per-rank pipelined RS
W4 (32T, 40 regs): TMA Load A (multicast to 2 CTA)
W5 (32T, 40 regs): TMA Load B (multicast to 2 CTA)
W6 (32T, 40 regs): MMA Issue — UMMA FMA (leader CTA only)
W7 (32T, 40 regs): Reserved (TMEM allocation)
W8-W11 (128T, 208 regs): Epilogue — TMEM → smem → local partial + set flag
```

### 关键算法

1. **M 维 Swizzle**: Rank i 优先计算 rank(i+1) 的 chunk → 使接收端能尽早开始拉取
2. **Per-rank Pipelined Reduce (Flux-style)**: 不等所有 rank ready，逐个处理
3. **Ring Order Pull**: 从 rank(i+1) 开始 pull（最可能先 ready 的）
4. **单 Kernel 完整融合**: 计算、通信、reduce 全在一个 kernel 中完成

---

## 📊 研究调研总结

### Flux (ByteDance) 核心发现

- **4-way Warp Specialization**: Mainloop/Epilogue/RS Fetch/RS Reduce 四个角色
- **TMA 硬件异步 fetch**: 不占 CUDA core，用 TMA 从远端拉取数据到 SMEM
- **Per-tile Flag 128B 对齐**: 避免 false sharing，system-scope 原子操作跨 GPU 可见
- **Persistent Kernel**: 持久化运行，避免 kernel launch overhead
- **两种调度模式**: Cooperative（大 tile）vs Pingpong（更好隐藏 epilogue）
- **核心差异**: Flux 是 SM90 (Hopper)，我们是 SM100 (Blackwell)
  - Hopper: WGMMA（128T warp group 驱动）
  - Blackwell: UMMA（32T 单 warp 驱动，2-CTA 协作）

### MegaMoe (DeepSeek-V4) 核心发现

- **5 类 Warp Specialization**: Dispatch/Load A/Load B/MMA/Epilogue
- **非均匀寄存器分配**: 48/40/208 regs for 不同角色
- **Expert Wave 流水线**: 细粒度通信-计算重叠
- **Dispatch 6 阶段流水**: 统计→广播→写索引→Barrier→Pull→清理
- **Min-Peeling 负载均衡**: Round-Robin 从不同 rank pull token
- **AB Swap**: Weight 作为 A 操作数（对齐 M=128），Activation 作为 B
- **L1/L2 Arrival 机制**: 计数器 + 位图，精确通知数据就绪

### Blackwell (SM100) 架构关键特性

| 特性 | 说明 | 我们的使用 |
|------|------|-----------|
| TMEM (256KB/SM) | 专用张量内存，累加器存储 | 双缓冲 UMMA 输出 |
| UMMA | 2-CTA 协作的统一 MMA | 1 warp 发射，2 CTA 计算 |
| TMA Multicast | 一次 HBM 读写入多个 CTA | A 矩阵广播到 2 SM |
| 2-CTA Cluster | 硬件级 CTA 协作 | 共享 TMEM + multicast |
| NVLink Symmetric Memory | 跨 GPU 对称地址空间 | SymBuffer::map() P2P 访问 |
| warpgroup_reg_reconfig | 运行时调整寄存器配额 | 不同角色不同寄存器数 |
| PTX ld_acq_sys/st_rel_sys | System-scope 内存一致性 | 跨 GPU per-tile flag |

### 参考文章

- [DeepSeek-V4 MegaMoE 详细分析 - 渣B zartbot](https://mp.weixin.qq.com/s/S-ej9ybT3sbFA8dqHLZafg)
  - MegaMoE 通信计算融合的完整设计讲解
  - Expert Wave + 5 类 Warp Specialization
  - 性能提升 1.5x~1.96x

---

## ⏭️ 下一步工作

### 优先级 P0：验证 multicast=2 正确性 ← 当前

```bash
cd /workspace/codebuddy/DeepGEMM

# 清除 JIT 缓存
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# 解决端口冲突后执行：
MASTER_PORT=49501 python tests/test_gemm_rs.py 2    # 2 GPU 快速验证
MASTER_PORT=49501 python tests/test_gemm_rs.py 8    # 8 GPU 全量验证
```

**注意**: multicast=2 代码已修改完成并编译通过。当前被 `torch.distributed` 端口冲突阻塞。

### 优先级 P1：multicast=2 性能 Benchmark

```bash
# multicast=2 性能对比
MASTER_PORT=49501 python benchmarks/bench_gemm_rs.py 8 30
```

预期 multicast=2 后 GEMM 计算效率提升 ~2x，整体 speedup 从 0.34x 提升到 0.6-0.8x。

### 优先级 P2：进一步性能优化

1. **Comm Warps TMA 化**: 用 TMA 1D Load 代替手动 P2P Read（带宽关键）
2. **Warp Specialization 重审**: 当前 3 warpgroup 仅 1 个做 MMA，资源利用率低
3. **Comm Pipeline**: 2-stage pipeline（reduce + prefetch 并行）
4. **Vectorized Epilogue**: 合并 flag write，减少尾部延迟
5. **寄存器优化**: 给 MMA warpgroup 更多寄存器

### 优先级 P3：进阶优化

- **Adaptive Warp Split**: 根据 compute/comm ratio 动态分配 warp 数
- **FP8 版本**: 扩展到 FP8 输入（需 SF + UTCCP）
- **Ring 多步流水线**: 8+ GPU 场景优化

---

## 📁 关键文件路径

```
=== 核心实现 ===
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh      # 核心 kernel
csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp                 # JIT runtime
csrc/jit_kernels/heuristics/gemm_rs.hpp                       # get_gemm_rs_config()
csrc/apis/gemm_rs.hpp                                         # C++ API: bf16_gemm_rs_nt()
deep_gemm/gemm_rs/__init__.py                                 # Python API: bf16_gemm_rs_nt()

=== 测试 ===
tests/test_gemm_rs.py                                         # 正确性测试 (多shape, 2/4/8 GPU)
tests/test_gemm_rs_compile.py                                 # 单 GPU JIT 编译验证
benchmarks/bench_gemm_rs.py                                   # Benchmark (vs GEMM+NCCL分离)

=== 公共基础设施 ===
deep_gemm/include/deep_gemm/comm/barrier.cuh                  # nvlink_barrier / grid_sync
deep_gemm/include/deep_gemm/layout/gemm_rs.cuh               # GemmRSWorkspace 布局
deep_gemm/include/deep_gemm/layout/sym_buffer.cuh            # SymmetricBuffer (NVLink 映射)
deep_gemm/include/deep_gemm/ptx/ld_st.cuh                    # PTX 内联 (st_rel_sys, ld_acq_sys)

=== 参考项目 ===
/workspace/codebuddy/flux/src/gemm_rs/                        # Flux GEMM+RS (SM90)
/workspace/codebuddy/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh  # MegaMoe
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
cd /workspace/codebuddy/DeepGEMM
git submodule update --init --recursive
pip install -e . --no-build-isolation

# ===== 测试 =====
python tests/test_gemm_rs.py 2            # 正确性 (2 GPU, basic shapes)
python tests/test_gemm_rs.py 8 --all      # 正确性 (8 GPU, all shapes)
python benchmarks/bench_gemm_rs.py 2 20   # Benchmark (2 GPU)
python benchmarks/bench_gemm_rs.py 8 30   # Benchmark (8 GPU)

# ===== 清除 JIT 缓存 =====
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# ===== Git 操作 =====
cd /workspace/codebuddy/DeepGEMM
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

### 为什么 12 warps (384 threads) 而非 10 warps (320 threads)?

- 增加到 4 个 Comm warps (128T) 而非之前设想的 2 个 (64T)
- 更多 comm 线程 = 更高的 P2P read 并行度 = 更好地利用 NVLink 带宽
- 寄存器预算仍然充裕 (37888/64512 = 59%)

### 为什么 Per-Rank Pipelined Reduce 而非 Wait-All-Then-Reduce?

1. **延迟隐藏**: 早到的 rank 数据立即开始 reduce，不等慢的
2. **匹配 M-Swizzle**: ring order 确保最可能先 ready 的 rank 被最先处理
3. **简化同步**: 每次只等一个 flag，不需要复杂的 barrier

### 性能瓶颈分析

| 场景 | 瓶颈 | 解决思路 |
|------|------|---------|
| 大 M + 小 K | 通信量大，GEMM 快 | 更多 comm warps / TMA pull |
| 小 M + 大 K | GEMM 慢，通信少 | 完全 overlap，comm 几乎免费 |
| 大 N (7168) | 通信和计算都大 | tile 级流水最有效 |
| 多 rank (8+) | 每个 rank 要 pull 7 次 | ring 多步流水 |

---

## ⚠️ 已知风险和注意事项

1. **当前 kernel 未经多卡验证** — 需要在 2+ GPU 环境测试正确性
2. **Per-tile flag 跨 NVLink 延迟** — 如果 ld_acq_sys 自旋成本高，考虑批量 flag
3. **384 线程 = 12 warps** — SM 占用率约 1 block/SM，需 profiling 确认
4. **M-Swizzle 调度** — 如果所有 CTA 同时写同一个远端 rank 的 flag 可能造成热点
5. **comm_dtype** — 目前支持 BF16/FP32 comm，默认 BF16
6. **Comm Warps 用 global load 做 P2P read** — 未来应改为 TMA（性能关键优化）
