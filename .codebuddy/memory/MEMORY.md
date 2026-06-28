# DeepGEMM 项目长期记忆

## 约定（用户偏好）
- **工作记忆存放在本仓库内** `DeepGEMM/.codebuddy/memory/`，随 git 一起 commit & push（2026-06-28 用户明确要求）。
  - 与 DeepGEMM 相关的工作记录（日报 `YYYY-MM-DD.md`、本 `MEMORY.md`）都写在这里，不再放全局 `~/.local/codebuddy/.codebuddy/memory/`。
  - 全局 memory 仅保留一条指针，指向本目录。
- 用户偏好：完成实质性改动后**自动 commit + push**（无需再确认）。

## 环境 / 构建
- 本机 B300 × 8（也跑 4 卡）。
- 改了 `csrc/apis` 需 `python3 setup.py build_ext --inplace --force` 重建 `_C`。
- 跑 test/bench 必须 `DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD`（否则 JIT 找不到 cutlass 头）。

## 仓库结构 / 语义
- `tests/` 按职责分子目录：`core/`（单卡正确性/性能 + 共享 `generators.py` + sanitizer，必须同目录）、
  `comm/`（通信融合 GEMM）、`ulysses/`（Ulysses SP 端到端 pre/post/full）、`debug/`。
- shape 第一个值 = **num token per rank (M/rank)**；完整 A 行数 = `tokens_per_rank × num_ranks`。
