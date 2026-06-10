# GEMM+RS 开发进度文档

## 项目概述
在 NVIDIA Blackwell B300 SXM6 (SM100) 8-GPU 平台上实现 GEMM+Reduce-Scatter 融合算子。
目标：大模型训练的长上下文、大 hidden dim (7168) 场景下显著提升吞吐量。

---

## 进度日志

### 2026-06-10

#### Bug Fix: 移除 warpgroup_reg_dealloc（死锁根因）

**问题描述**：
kernel 在多 GPU 测试时 hang（超时）。

**根因分析**：
1. Load Warp A/B 的分支条件是 `warp_idx == X and cute::elect_one_sync()`
2. `elect_one_sync()` 导致只有 lane 0 进入分支体
3. 分支体中调用了 `cutlass::arch::warpgroup_reg_dealloc<40>()`
4. `warpgroup_reg_dealloc` 的底层 PTX 是 `setmaxnreg.dec.sync.aligned.u32`
5. `.sync.aligned` 语义要求 warp 中所有 32 个 lane 必须同时执行该指令
6. 只有 lane 0 执行 → 其他 31 lane 永远等不到 → **死锁**

**关键知识点**：
- `setmaxnreg.dec.sync.aligned` 是 warp-collective 操作，类似 `__syncwarp()` 
- 标准 GEMM（`sm100_bf16_gemm.cuh`）中 Load warp 完全没有使用 `reg_dealloc`
- MegaMoE 使用 `reg_dealloc` 时是在 `if (warp_idx == X)` 中，没有 `elect_one_sync()`

**修复方案**：
直接移除 Load Warp A/B、MMA Warp、Reserved Warp 中的 `warpgroup_reg_dealloc` 调用。
保留 Comm Warps (W0-3) 的 `reg_dealloc<48>`（无 `elect_one_sync()`，全 warp 执行，不会死锁）。
保留 Epilogue Warps (W8-11) 的 `reg_alloc<208>`（同理）。

**理由**：标准 GEMM 不用，我们也不用。先确保 kernel 能正常运行再考虑寄存器优化。

**修改文件**：
- `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
  - 移除第 450 行 `cutlass::arch::warpgroup_reg_dealloc<kNumNonEpiRegisters>()` (Load A)
  - 移除第 489 行 `cutlass::arch::warpgroup_reg_dealloc<kNumNonEpiRegisters>()` (Load B)
  - 移除第 526 行 `cutlass::arch::warpgroup_reg_dealloc<kNumNonEpiRegisters>()` (MMA)
  - 移除第 606 行 `cutlass::arch::warpgroup_reg_dealloc<kNumNonEpiRegisters>()` (Reserved)

---

#### 架构理解：标准 GEMM vs GEMM-RS 的 warp 角色对比

| 标准 GEMM (sm100_bf16_gemm.cuh) | GEMM-RS (sm100_bf16_gemm_rs.cuh) |
|------|------|
| W0: Load (elect_one_sync) | W0-3: Comm (reg_dealloc<48>) |
| W1: MMA Issue (is_leader_cta) | W4: Load A (elect_one_sync) |
| W2: TMEM Allocator | W5: Load B (elect_one_sync) |
| W3+: Epilogue | W6: MMA Issue (is_leader_cta) |
| | W7: Reserved |
| | W8-11: Epilogue (reg_alloc<208>) |

标准 GEMM 只有 128 线程非 epilogue (4 warps) + epilogue warps。
GEMM-RS 有 384 线程 = 128(comm) + 128(non-epi) + 128(epilogue)。

---

#### mbarrier 语义笔记

- `mbar.init(expected_arrivals)` 后 parity=1（wait(1) 立即通过，wait(0) 阻塞）
- 第一次 arrive 完成后 parity 翻转为 0
- 对于 full_barriers：producer arrive 后 consumer 用 `wait(phase)` 等待
- 对于 empty_barriers：consumer arrive 后 producer 用 `wait(phase^1)` 等待

---

## 当前状态

- [x] 核心 kernel 代码已完成 (sm100_bf16_gemm_rs.cuh)
- [x] JIT 编译入口完成 (sm100_bf16_gemm_rs.hpp)
- [x] 启发式配置完成 (heuristics/gemm_rs.hpp)
- [x] Python API 完成 (deep_gemm/gemm_rs/__init__.py)
- [x] 多卡正确性测试脚本完成 (tests/test_gemm_rs.py)
- [x] 多卡性能测试脚本完成 (benchmarks/bench_gemm_rs.py)
- [x] 修复 reg_dealloc 死锁 bug
- [ ] **编译通过 + 多卡正确性测试通过** ← 当前步骤
- [ ] 8 GPU 性能基准测试
- [ ] 性能分析与优化

---

## 参考资料

1. **ByteDance Flux** (flux/ 目录): Pull-based RS with per-tile flags, Comm warp specialization
2. **DeepSeek MegaMoE**: warp specialization + reg budget + NVLink symmetric memory
3. **CUTLASS SM100**: UMMA 2-CTA cooperative, TMA multicast, mbarrier pipeline
4. **NVIDIA PTX ISA**: `setmaxnreg.dec.sync.aligned` 语义，system-scope memory ordering
5. **DeepGEMM 标准 GEMM** (sm100_bf16_gemm.cuh): 参考实现（简单但正确）

---

## 开发环境

- 平台: 8× NVIDIA B300 SXM6 (SM100, 80GB HBM3e)
- NVLink: NVLink Gen5 (900 GB/s bidirectional per GPU pair)
- CUDA: 12.x + sm_100 target
- PyTorch: with CUDA support
- CUTLASS: 3.x (third-party/cutlass)
