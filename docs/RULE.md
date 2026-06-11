# DeepGEMM GEMM+RS 开发规则

> **服务器 CodeBuddy 专用**：请按以下规则和环境信息执行开发任务。
> **⏰ 自提醒**：上下文变长时（超过 20 轮对话或开始新任务），必须重新读取本文件刷新记忆。
> **🧠 首次使用时**：读取本文件后，立即用 update_memory 工具将以下核心规则存入 Memory（标题："DeepGEMM 开发核心规则"），确保后续对话自动记住：
> 1. 每完成阶段性改动立即 commit + push，服务器经常宕机，不push=代码可能丢失。同时更新 docs/PROGRESS.md
> 2. 每个新任务开始前或上下文变长时，重新读取 docs/RULE.md 刷新记忆
> 3. AKO4ALL 优化时：不用 solution/ 隔离，不创建 opt/ 分支，直接在 main 分支原文件迭代，用项目已有测试和 benchmark，每轮 push
> 4. 开发目标一：tests/ 多卡正确性测试；目标二：benchmarks/ 多卡性能测试
> 5. 当前状态和 P0 任务见 docs/PROGRESS.md「当前状态」章节，需要实时更新

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
目标场景：大模型训练的长上下文（如 Megatron Sequence Parallelism），M_per_rank 通常 10K~20K+ tokens，大 hidden dim（N=7168）。

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
2. **每完成一个阶段性改动就立即 commit + push**，不要攒一批再推。服务器经常莫名其妙宕机，不 push = 代码可能丢失。同时更新进度文档 `docs/PROGRESS.md`
3. 开发目标一：`tests/` 目录下实现多卡正确性测试
4. 开发目标二：`benchmarks/` 目录下实现多卡性能测试，目标是训练场景的吞吐量提升
5. 跑通所有测试和性能测试，确保多卡正确性和性能
6. 有意义的发现和思考记录到 `docs/PROGRESS.md`，有帮助的资料也记录并 push
7. **不要自我怀疑已确认的结论**：通过 debug、测试、CUTLASS 源码验证过的行为（如 2SM UMMA 的编程模型、TMA load 行为、scheduler tile 分配），不要在后续开发中反复质疑。如果不确定，先写个测试验证，而不是推翻重来。
8. **关注编程模型和使用方式，不要纠结硬件实现细节**：我们只需要正确使用 Umma/TMA/Cluster 的 API 和语义，不需要操心硬件内部怎么实现的。

---

## 5. AKO4ALL Skill — GPU Kernel 自动优化

### 安装

```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后 skill 位于 `~/.codebuddy/skills/ako4all/SKILL.md`，CodeBuddy 自动识别。

### 前置条件

**AKO4ALL 只适用于功能已完整、正确性已验证的 kernel。** 如果 kernel 还有 bug 或功能未完成，应先修复再优化。流程：
1. 完成功能实现 → 2. 正确性验证通过 → 3. 用 AKO4ALL 做性能优化

当前 P0 阻塞：`multicast=2 hang`，需先修复，不宜使用 AKO4ALL。

### 使用

重启 CodeBuddy 后，在项目目录中对话触发：
```
Optimize the kernel at ./deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh for up to 30 iterations.
```

### AKO4ALL 工作方式 vs 我们的适配

**AKO4ALL 默认流程**：
1. 在同一仓库创建 `opt/<kernel>` git 分支
2. 把 kernel 复制到 `solution/` 子目录独立迭代
3. 自动循环：benchmark → 验证正确性 → ncu profiling → 修改 `solution/` 里的代码 → 重新测速
4. 每轮自动 git commit，最终 `solution/` 里是最优版本

> 注意：我们不用默认流程，而是直接在 main 分支原文件上迭代（见下方适配方案）。

**我们的问题**：
- 我们的 kernel 跨多个文件（`.cuh` + `.hpp` + heuristics + 编译链），不适合单文件隔离到 `solution/`
- 服务器经常宕机，需要频繁 push 防丢代码
- 优化完还需要从 `solution/` 搬回原位，多文件搬运容易出错

**适配方案：直接在原文件位置迭代，不用 `solution/` 隔离**

在触发 skill 时告诉 agent：
```
Optimize the kernel at deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh
Do NOT copy to solution/. Edit the original files in place.
Related files: csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp, deep_gemm/include/deep_gemm/gemm_rs.cuh
Use project's existing test: python -m pytest tests/test_gemm_rs.py -v
Use project's existing benchmark: python benchmarks/bench_gemm_rs.py
Push after every iteration to prevent data loss from server crashes.
```

这样 agent 会：
- 直接在 main 分支上修改原文件（不创建独立分支）
- 用项目已有的测试和 benchmark 脚本
- 每轮 commit + push，代码安全
- 优化结果直接就在原文件位置，无需搬运

> **防宕机策略**：每轮迭代后必须 `git push`。在 `HINTS.md` 中加入 `Push after every iteration` 指令确保执行。

### 环境要求

- NVIDIA GPU + CUDA PyTorch
- Python >= 3.10
- **`ncu`（Nsight Compute）— 必须安装**，AKO4ALL 依赖 ncu 做 kernel profiling
  - 安装脚本：`bash dev/startup/install_ncu.sh`（自动检测 + 多回退方案）
  - 手动安装：`apt install -y nsight-compute-cli`（NVIDIA devtools repo 已配置时）
- Git

### HINTS.md 常用配置

- 限制迭代次数：`Max iterations: 30`
- 语言偏好：`Preferred language: CUDA`
- 禁止 web search：`Web search: disabled`
- 禁止安装依赖：`No pip/apt installs`
- **防宕机**：`Push after every iteration`

### 支持 kernel 语言

Triton、CUDA、C++、TileLang、CuTe DSL、Python 等

---

## 6. 核心文件路径

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

## 7. 运行命令速查

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

## 8. 开发环境

- **目标 GPU**: 8× NVIDIA B300 SXM6 (SM100, 80GB HBM3e)
- **NVLink**: Gen5 (900 GB/s 双向)
- **架构**: SM100 (Blackwell)
- **Python 包**: `deep_gemm` (editable install)
- **JIT 缓存**: `~/.deep_gemm/cache/`
- **第三方**: CUTLASS + fmt (git submodule)
