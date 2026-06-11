# DeepGEMM GEMM+RS 开发规则

> **服务器 CodeBuddy 专用**：请按以下规则和环境信息执行开发任务。

---

## 1. 仓库与认证

- **主仓库**：https://github.com/Schilings/DeepGEMM.git
- **关联仓库**：https://github.com/Schilings/DeepEP.git、https://github.com/bytedance/flux.git
- **Git 用户名**：schilings
- **Git 邮箱**：1146830743@qq.com
- **GitHub Token**：见环境变量 `GITHUB_TOKEN`，勿写入文件（GitHub Push Protection 会拦截）
- **确保有 push 权限**：clone 后验证 `git push` 可正常工作

---

## 2. 项目目标

在 NVIDIA Blackwell B300 SXM6 (SM100) 8-GPU 平台上实现 **GEMM + Reduce-Scatter 融合算子**，
目标场景：大模型训练的长上下文（M_per_rank=2048~8192）、大 hidden dim（N=7168）。

---

## 3. 前置学习（按顺序阅读）

1. **Flux GEMM+RS**：`flux/src/gemm_rs/` — 字节 Hopper 上的 GEMM+RS 融合，学习调度和 overlap 设计思想
2. **Blackwell 架构特性**：`docs/SM100_2CTA_CLUSTER.md` — 2-CTA Cluster、UMMA、TMA multicast 详解
3. **标准 GEMM**：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh` — Blackwell GEMM 参考实现
4. **MegaMoE**：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` + [讲解文章](https://mp.weixin.qq.com/s/S-ej9ybT3sbFA8dqHLZafg)
5. **当前方案设计**：`docs/GEMM_RS_DESIGN.md` — Pull-based 单 kernel 融合设计
6. **当前进度**：`docs/PROGRESS.md` — bug fix 记录、benchmark 结果、当前阻塞

---

## 4. 开发规则

1. 你是优秀的 CUDA 算子开发者，熟悉 Blackwell 架构、GEMM+RS 融合、通信计算 overlap
2. **阶段性地 push 代码**到远程仓库，同时更新进度文档 `docs/PROGRESS.md`
3. 开发目标一：`tests/` 目录下实现多卡正确性测试
4. 开发目标二：`benchmarks/` 目录下实现多卡性能测试，目标是训练场景的吞吐量提升
5. 跑通所有测试和性能测试，确保多卡正确性和性能
6. 有意义的发现和思考记录到 `docs/PROGRESS.md`，有帮助的资料也记录并 push

---

## 5. AKO4ALL Skill — GPU Kernel 自动优化

### 安装

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后 skill 位于 `~/.codebuddy/skills/ako4all/SKILL.md`，CodeBuddy 自动识别。

### 使用

重启 CodeBuddy 后，在项目目录中对话触发：
```
Optimize the kernel at ./deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh for up to 30 iterations.
```

Agent 自动：创建 `opt/<kernel>` 分支 → 复制到 `solution/` → 生成 benchmark → 迭代优化。

### 环境要求

- NVIDIA GPU + CUDA PyTorch
- Python >= 3.10
- `ncu`（Nsight Compute）— 推荐，缺失时仍可运行但不做 profiling
- Git

### HINTS.md 常用配置

- 限制迭代次数：`Max iterations: 30`
- 语言偏好：`Preferred language: CUDA`
- 禁止 web search：`Web search: disabled`
- 禁止安装依赖：`No pip/apt installs`

### 支持 kernel 语言

Triton、CUDA、C++、TileLang、CuTe DSL、Python 等

---

## 6. 当前状态与最高优先级任务

### 已完成

- [x] Pull-based 单 kernel 融合方案代码 (sm100_bf16_gemm_rs.cuh)
- [x] JIT 编译 + 启发式 + Python API
- [x] multicast=1: 8 GPU 正确性 6/6 ALL PASS
- [x] multicast=1: 8 GPU benchmark (geo_mean=0.34x vs NCCL)
- [x] 修复 4 个 bug: reg_dealloc 死锁 / slot 寻址 / ready flag race / 测试阈值
- [x] 修复 multicast=2 scheduler + TMA load + epilogue

### 当前阻塞 — P0

- [ ] **multicast=2 kernel hang**：JIT 编译成功，kernel launch 成功，但 GPU 上死循环（8卡全 100%）
  - 需排查 mbarrier 同步 / Epilogue ready flag / multicast=2 专有逻辑
  - 可先强制 `num_multicast=1` 验证基线仍 PASS

### 后续优化 — P1/P2

- [ ] multicast=2 benchmark（预期 GEMM 效率 ~2x）
- [ ] Comm Warps TMA 化（TMA 1D Load 代替手动 P2P Read）
- [ ] Warp specialization 重审（3 warpgroup 仅 1 个做 MMA）
- [ ] Comm Pipeline（2-stage: reduce + prefetch 并行）

---

## 7. 核心文件路径

```
=== 核心实现 ===
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh   # 核心 kernel
csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp               # JIT runtime
csrc/jit_kernels/heuristics/gemm_rs.hpp                      # 启发式配置
csrc/apis/gemm_rs.hpp                                        # C++ API
deep_gemm/gemm_rs/__init__.py                                # Python API

=== 测试 ===
tests/test_gemm_rs.py                                        # 正确性测试 (2/4/8 GPU)
benchmarks/bench_gemm_rs.py                                  # Benchmark

=== 基础设施 ===
deep_gemm/include/deep_gemm/comm/barrier.cuh                 # nvlink_barrier
deep_gemm/include/deep_gemm/layout/gemm_rs.cuh               # GemmRSWorkspace
deep_gemm/include/deep_gemm/ptx/ld_st.cuh                    # PTX ld_acq_sys / st_rel_sys

=== 文档 ===
docs/PROGRESS.md          # 进度日志（最新）
docs/GEMM_RS_DESIGN.md    # 当前方案设计
docs/SM100_2CTA_CLUSTER.md # 2-CTA Cluster 详解
docs/RULE.md              # 本文件
```

---

## 8. 运行命令速查

```bash
# 安装
cd /workspace/codebuddy/DeepGEMM
git submodule update --init --recursive
pip install -e . --no-build-isolation

# 测试
python tests/test_gemm_rs.py 2          # 2 GPU 正确性
python tests/test_gemm_rs.py 8 --all    # 8 GPU 全量正确性

# Benchmark
python benchmarks/bench_gemm_rs.py 2 20 # 2 GPU
python benchmarks/bench_gemm_rs.py 8 30 # 8 GPU

# 清除 JIT 缓存
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# Git
git add -A && git commit -m "描述" && git push
```

> **注意**：不要用 `torchrun`，脚本内部用 `mp.spawn` 管理多进程。

---

## 9. 开发环境

- **目标 GPU**: 8× NVIDIA B300 SXM6 (SM100, 80GB HBM3e)
- **NVLink**: Gen5 (900 GB/s 双向)
- **架构**: SM100 (Blackwell)
- **Python 包**: `deep_gemm` (editable install)
- **JIT 缓存**: `~/.deep_gemm/cache/`
- **第三方**: CUTLASS + fmt (git submodule)
