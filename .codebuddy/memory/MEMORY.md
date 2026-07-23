# DeepGEMM 项目长期记忆

## 当前进度（2026-07-23，Ulysses Variant v2）
- **状态：v2（原生 AG + 延迟 QKV 权重梯度 overlap）已实现并完成全部实验。** 权威结果见 `examples/ulysses_variant_v2/WAN21_ULYSSES_V2_BENCH.md`，过程记忆见 `memory/2026-07-23.md`。
- v2 核心思想：POST backward 改用原生 NCCL all_gather + 原生 GEMM（不再用融合 AG+GEMM），将 QKV 权重梯度延迟到下一层 AG 通信窗口中 overlap。
- **POST backward 速度**：v2 actual autograd 0.393ms vs v1 0.765ms（8K/SP8），仅为 v1 的 0.513×，接近 baseline 0.369ms（1.065×）。原生 NCCL AG (0.213ms) 远快于 v1 融合 AG+GEMM (0.573ms)。
- **训练吞吐（manual sync）**：v2 23,142 > v1 22,772 > serial 22,289 tok/s（+3.83% vs serial, +1.62% vs v1）。
- **v2 在所有序列长度下优于 v1**（+0.9% ~ +4.7%），BWD 快 2.3% ~ 9.9%。
- **显存与 v1 一致**：8K 节省 20.5%。
- **1000 步训练收敛性一致**：前 900 步 diff < 0.02%。
- DDP 模式有局限：QKV 排除出 DDP 后手动 all-reduce 无法 overlap，推荐 manual sync。
- FA4 SM 10.3 兼容性修复：`flash_fwd_sm100.py:162` 断言 `sm_110f`（别名 `sm_101f`）改为 `sm_103f`。`third-party/cutlass/include` 从 `nvidia/mathdx` 包符号链接。

## 历史进度（2026-07-19，Wan2.1 Ulysses POST 变体 v1）
- **状态：显存与真实 14B 权重训练核心吞吐均已实测。** 权威结果表见 `examples/ulysses_variant/WAN21_ULYSSES_BENCH.md`，过程记忆见 `memory/2026-07-19.md`。
- 严格两臂：baseline=纯 torch 同步 Ulysses + FA4（无融合算子）；variant=只换 POST 为 GEMM-RS/AG-GEMM。两臂 PRE/RoPE/attention 共用同一代码且都用 FA4。
- 关键坑：本机 8 卡 = `SP=8, DP=1`，**不能把 SP group 当 FSDP group**（否则 baseline Wo 被切 1/8 抹掉收益）。
- 显存实测（attention stack + FP32 Adam）：B300×8、40 层 → 8K 省 9.9 GiB(20.6%)，32K 省 9.6 GiB(14.0%)。
- 真实权重吞吐：官方 checkpoint 严格加载 40 blocks/1080 tensors/14.056B params。最终严格 wall-clock、warmup=3/iters=10、DDP overlap、AG 发布+消费双 barrier：baseline 29,250 vs variant 28,193 tok/s，variant **-3.61%**。
- SP 梯度：所有复制参数都需跨 SP reduce；variant 只有 `Wo_r_local` 不做 SP reduce，因为其 backward 已 AG 完整 `grad_y` 且各 rank 持有不同输入列 shard。若有 DP，Wo 仍需在对应 DP group 同步。
- AG 生命周期已从 host fence 改成两个 stream-ordered `sym_buffer.handle.barrier()`：一个发布本代输入，一个确认所有 peer 已消费后才允许下一层覆盖；SP=8 实测 grad_X rel=0。最终 40 层 BWD variant 198.46ms，baseline 185.77ms。
- BWD 剩余慢点已由重复独立测试+Nsight 确认为 AG 本体：8K/SP8 远端 payload 70 vs A2A 8.75 MiB（8×）；AG kernel 约 475.7us、同形状纯 GEMM 45.1us、NCCL A2A kernel 46.9us。真实 autograd POST BWD 均值：8K variant≈1.34× baseline，32K≈1.97×。
- **通用 Profiling SOP**：新会话分析任何 GPU 算子、通信、autograd 或完整训练时先读 `docs/GPU_PROFILING_GUIDE.md` 和 `memory/2026-07-20.md`。Ulysses 仅为附录案例。核心原则：禁止 benchmark 并行争用同一批卡；先无 profiler 重复测量，再用 NVTX+Nsight 归因，最终吞吐使用同步后的 rank-max wall-clock；trace 二进制提取后删除。

## 约定（用户偏好）
- **工作记忆存放在本仓库内** `DeepGEMM/.codebuddy/memory/`，随 git 一起 commit & push（2026-06-28 用户明确要求）。
  - 与 DeepGEMM 相关的工作记录（日报 `YYYY-MM-DD.md`、本 `MEMORY.md`）都写在这里，不再放全局 `~/.local/codebuddy/.codebuddy/memory/`。
  - 全局 memory 仅保留一条指针，指向本目录。
- 用户偏好：完成实质性改动后**自动 commit + push**（无需再确认）。

## 环境 / 构建
- 本机 B300 × 8（也跑 4 卡）；CUDA 13.0 / torch 2.9.0+cu130 / Python 3.12。
- 改了 `csrc/apis` 需 `python3 setup.py build_ext --inplace --force` 重建 `_C`。
- 跑 test/bench 必须 `DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD`（否则 JIT 找不到 cutlass 头）。

## Attention：统一用 FlashAttention-4
- 安装：`pip install "flash-attn-4[cu13]==4.0.0b19"`（或 `bash scripts/install_fa4.sh`，固定版本见 `docs/INSTALL_FA4.md`）。
  CuTeDSL 纯 Python 包，运行时 JIT，无大二进制。
- import：`from flash_attn.cute import flash_attn_func, flash_attn_varlen_func`。
- 接口坑：dense 用 **[B,S,H,D]**（非 BHSD）；varlen 用 [T,H,D]+cu_seqlens（关键字传，第 4 个位置是 qv）；无 dropout_p；返回可能是 tuple。
- 测试/bench 统一经 `tests/ulysses/fa4_attn.py` 的 `fa4_attn_bhsd` / `fa4_attn_varlen_thd` 调用。

## Standard Ulysses fused benchmark（2026-07-20 清理）
- `examples/ulysses_fused/` 只保留两个文件：`bench_ulysses_full_attn_flow.py` 和 `ULYSSES_FULL_ATTN_BENCH.md`。
- 只比较两条等价标准 Ulysses forward：torch+同步 NCCL baseline vs DeepGEMM fused PRE/POST；不含其他实验策略。
- 已删除混合旧 API 且不可运行的 `bench_wan21_strategies.py` 和沿 SP group 做 FSDP2 的 `bench_wan21_fsdp2.py`。
- B300×8、10 iters、rank-max：BSHD chain geo 1.032x、PRE+POST 1.111x；THD chain 1.026x、PRE+POST 1.098x。
- 8 卡正确性 3/3 PASS，relative error 均为 1.41e-3。
- chain 是独立 PRE+FA4+POST 计时之和，不是 autograd 训练吞吐；权威结果见 `examples/ulysses_fused/ULYSSES_FULL_ATTN_BENCH.md`。

## 仓库结构 / 语义
- `tests/` 按职责分子目录：`core/`（单卡正确性/性能 + 共享 `generators.py` + sanitizer，必须同目录）、
  `comm/`（通信融合 GEMM）、`ulysses/`（Ulysses SP 端到端 pre/post/full）、`debug/`。
- shape 第一个值 = **num token per rank (M/rank)**；完整 A 行数 = `tokens_per_rank × num_ranks`。
