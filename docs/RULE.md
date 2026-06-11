克隆仓库：https://github.com/Schilings/DeepGEMM.git
克隆：https://github.com/Schilings/DeepEP.git
克隆：https://github.com/bytedance/flux.git
邮箱：1146830743@qq.com
用户名：schilings
配置token：（见本地环境变量 GITHUB_TOKEN，勿写入文件）
麻烦在当前目录clone该项目，并确保你有push的权限。

由于我们要在Blackwell架构上实现单机多卡的Gemm+Reduce Scatter融合性算子，因此，需要你先熟悉以下内容：
1. flux项目是字节的flux项目，有在hopper实现Gemm+Reduce Scatter融合性算子，见flux\src\gemm_rs。但是Blackwell架构与hopper存在较大差别。我们主要学习flux对于Gemm+Reduce Scatter融合性算子的调度和设计思想，其是怎么最大化计算和通信掩盖和吞吐的。
2. 你要对blackwell架构要有足够的了解，调研学习一下Blackwell架构特性。
3. 熟悉Blackwell架构的Gemm实现，见DeepGEMM\deep_gemm\include\deep_gemm\impls\sm100_bf16_gemm.cuh
4. 熟悉悉Blackwell架构下的通信和计算融合算子的实现，例子DeepGEMM\deep_gemm\include\deep_gemm\impls\sm100_fp8_fp4_mega_moe.cuh。讲解文章见：https://mp.weixin.qq.com/s/S-ej9ybT3sbFA8dqHLZafg。
5. 熟悉目前我们的开发进度，Gemm+Reduce Scatter的进度文档在DeepGEMM\docs，算子实现在DeepGEMM\deep_gemm\include\deep_gemm\impls\sm100_bf16_gemm_rs.cuh。

下面是开发规则：
1. 你是一个优秀的CUDA算子开发者，你需要在Blackwell架构上实现Gemm+Reduce Scatter融合性算子。你熟悉Blackwell架构，你熟悉Gemm+Reduce Scatter融合性算子的实现，你熟悉Blackwell架构下的通信和计算融合算子的实现。
2. 开发时，需要阶段性地push代码到远程仓库，同时更新进度文档，实时保留开发进度。
3. 开发目标一：在DeepGEMM\tests目录，实现Gemm+Reduce Scatter融合性算子的多卡正确性测试。
4. 开发目标二：在DeepGEMM\benchmarks目录，实现Gemm+Reduce Scatter融合性算子的多卡性能测试。我们地目标是，做到吞吐量和性能提升较大，特别时大模型训练地长上下文和大hidden dim=7k这种输入大地场景。
5. 我给予你足够大的权利，你需要跑通所有测试和性能测试，需要实现所有算子的多卡正确性测试和性能测试。
6. 你在调研学习或者开发中有意义的发现和思考，需要及时记录到进度文档中。有帮助的资料和文章，也需要记录到进度文档中，且实时push到远程仓库。

---

## AKO4ALL Skill 安装与使用

### 简介
AKO4ALL 是一个 GPU kernel 自动优化工具，以 CodeBuddy skill 形式交付。放到工作目录后，agent 自动迭代优化 kernel：profile → 编辑 → benchmark → 重复，直到性能不再提升。

项目地址：https://github.com/TongmingLAIC/AKO4ALL

### 安装方法（CodeBuddy）

**一键安装命令：**
```bash
git clone https://github.com/TongmingLAIC/AKO4ALL.git ~/.codebuddy/skills/ako4all
```

安装后 skill 文件位于 `~/.codebuddy/skills/ako4all/SKILL.md`，CodeBuddy 会自动识别。

> 如果已有本地 clone，也可软链接：`ln -s /path/to/AKO4ALL ~/.codebuddy/skills/ako4all`

### 使用方法

1. **重启 CodeBuddy** 使新 skill 生效
2. 在包含 kernel 的项目目录中打开 CodeBuddy
3. 直接对话触发 skill，例如：
   ```
   Optimize the kernel at ./deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh for up to 30 iterations.
   ```
4. Agent 自动完成：创建 `opt/<kernel>` 分支 → 复制 kernel 到 `solution/` → 生成 benchmark 脚本 → 验证基线 → 迭代优化

### 环境要求
- NVIDIA GPU + CUDA PyTorch
- Python >= 3.10
- `ncu`（Nsight Compute）— 推荐，缺失时仍可运行但不做 profiling
- Git

### 推荐的工作目录布局（非强制）
```
workspace/
├── source/           # Kernel + 可选的 reference 和 inputs
│   ├── kernel.py
│   ├── reference.py  # 可选：正确性参考
│   └── inputs.py     # 可选：输入数据
├── bench/            # 自定义 benchmark 脚本（可选）
├── knowledge/        # 参考资料（可选）
└── HINTS.md          # Agent 指令（可选：迭代上限、语言偏好等）
```

### HINTS.md 常用配置
- 限制迭代次数：`Max iterations: 30`
- 禁止 web search：`Web search: disabled`
- 禁止安装依赖：`No pip/apt installs`
- 语言偏好：`Preferred language: CUDA`

### 支持的 kernel 语言
Triton、CUDA、C++、TileLang、CuTe DSL、Python 等任何可 benchmark 的语言