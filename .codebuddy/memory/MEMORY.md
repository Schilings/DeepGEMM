# DeepGEMM 项目长期记忆

## 当前进度（2026-07-19，Wan2.1 Ulysses POST 变体）
- **状态：显存与真实 14B 权重训练核心吞吐均已实测。** 详见 `memory/2026-07-19.md`。
- 严格两臂：baseline=纯 torch 同步 Ulysses + FA4（无融合算子）；variant=只换 POST 为 GEMM-RS/AG-GEMM。两臂 PRE/RoPE/attention 共用同一代码且都用 FA4。
- 关键坑：本机 8 卡 = `SP=8, DP=1`，**不能把 SP group 当 FSDP group**（否则 baseline Wo 被切 1/8 抹掉收益）。
- 显存实测（attention stack + FP32 Adam）：B300×8、40 层 → 8K 省 9.9 GiB(20.6%)，32K 省 9.6 GiB(14.0%)。
- 真实权重吞吐：官方 checkpoint 严格加载 40 blocks/1080 tensors/14.056B params。最终严格 wall-clock、warmup=3/iters=10、DDP overlap、AG 发布+消费双 barrier：baseline 29,250 vs variant 28,193 tok/s，variant **-3.61%**。
- SP 梯度：所有复制参数都需跨 SP reduce；variant 只有 `Wo_r_local` 不做 SP reduce，因为其 backward 已 AG 完整 `grad_y` 且各 rank 持有不同输入列 shard。若有 DP，Wo 仍需在对应 DP group 同步。
- AG 生命周期已从 host fence 改成两个 stream-ordered `sym_buffer.handle.barrier()`：一个发布本代输入，一个确认所有 peer 已消费后才允许下一层覆盖；SP=8 实测 grad_X rel=0。最终 40 层 BWD variant 198.46ms，baseline 185.77ms。
- BWD 剩余慢点已由重复独立测试+Nsight 确认为 AG 本体：8K/SP8 远端 payload 70 vs A2A 8.75 MiB（8×）；AG kernel 约 475.7us、同形状纯 GEMM 45.1us、NCCL A2A kernel 46.9us。真实 autograd POST BWD 均值：8K variant≈1.34× baseline，32K≈1.97×。

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

## 待办 / TODO
- （已完成 2026-06-28）~~Ulysses bench 加 async Ulysses baseline~~ → 见下「Ulysses bench async 基线」。

## Ulysses bench async-Ulysses 基线（2026-06-28 完成）
- 文件 `benchmarks/bench_ulysses_full_attn_flow.py` 现对比**三条链路**：fused(ours) / torch-native(串行) / async-Ulysses(手工多 stream 重叠)。
- async PRE：拆 Q/K/V → 用 `Wq_t/Wk_t/Wv_t` 各做一次 GEMM + 各发一次 `all_to_all_single`，
  comp 默认流算 GEMM、`comm_stream` 侧流发 A2A，event 串依赖：A2A(Q) 与 K 的 GEMM 重叠。
- async POST：token 维 `llocal_seq` 切 `nseg`(≤4，按 128 整除回退) 块，逐块在 `comm_stream_po` 上 scatter+A2A 流水线，
  comp 流上各块 GEMM 等自己那块 A2A 完成 → 与后续块 A2A 重叠（split-token，非 split-K）。
- 表格列：时间 `us=ours/torch/async`，加速比 `vs_torch/vs_async`；汇总 geo_mean 同时给两个口径。
- 文档 `docs/ULYSSES_FULL_ATTN_BENCH.md` 第 1 节已补三链路说明；第 4 节结果表标注为加 async 前的旧数据，async 列待 B300×8 重跑补全。
- 注意：脚本只 `py_compile` 过，**尚未真正多卡跑过**，下次需实跑验证（NCCL 双流重叠正确性 + 数字）。

## 仓库结构 / 语义
- `tests/` 按职责分子目录：`core/`（单卡正确性/性能 + 共享 `generators.py` + sanitizer，必须同目录）、
  `comm/`（通信融合 GEMM）、`ulysses/`（Ulysses SP 端到端 pre/post/full）、`debug/`。
- shape 第一个值 = **num token per rank (M/rank)**；完整 A 行数 = `tokens_per_rank × num_ranks`。
