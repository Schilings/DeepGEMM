# DeepGEMM 项目长期记忆

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
- **Ulysses bench 加 async Ulysses baseline**（2026-06-28 用户提出，未做）：
  当前 torch-native PRE 是串行（整块 QKV GEMM → 一次完整 A2A，不重叠）。需补一条 async Ulysses：
  把 Q/K/V 拆开分别算，用多 stream 做计算-通信流水线重叠（Q GEMM 完即发 A2A，同时算 K，依此类推），
  作为比串行 torch-native 更强的对照，以体现融合算子（单 kernel epilogue scatter 重叠）相对手工重叠的额外收益。

## 仓库结构 / 语义
- `tests/` 按职责分子目录：`core/`（单卡正确性/性能 + 共享 `generators.py` + sanitizer，必须同目录）、
  `comm/`（通信融合 GEMM）、`ulysses/`（Ulysses SP 端到端 pre/post/full）、`debug/`。
- shape 第一个值 = **num token per rank (M/rank)**；完整 A 行数 = `tokens_per_rank × num_ranks`。
