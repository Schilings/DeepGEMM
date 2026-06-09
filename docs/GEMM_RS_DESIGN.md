# GEMM+RS: Pull-based Single-Kernel Fusion with Tile-Level Overlap

## 概述

DeepGEMM 的 GEMM-RS 融合 kernel，在 NVIDIA Blackwell (SM100) 架构上实现**计算-通信 tile 级流水线重叠**。
借鉴 ByteDance Flux 和 DeepSeek MegaMoe 的设计思想。

> **状态**: 代码已完成，待多卡环境测试验证。

## 背景与动机

之前尝试过 Push+PDL 两阶段方案，验证后发现性能不可接受：
- Symmetric Push O(N) 通信量随 rank 数线性增长
- 无真正 overlap：全部 GEMM 完成才通信
- 8GPU 下仅 NCCL 分离方案的 0.21x

当前方案的目标：
- 通信量降到 bandwidth-optimal: O((N-1)/N)
- 实现 tile 粒度的计算-通信重叠
- 单 kernel 完成所有工作，消除 inter-kernel 开销

## 核心设计

| 维度 | 说明 |
|------|------|
| Kernel 数量 | 1 (全融合) |
| 通信方向 | Pull (读远端) |
| 通信模型 | All-to-1 Pull O((N-1)/N) |
| Overlap 粒度 | Tile 级流水线 |
| 同步模型 | Per-tile ready flag |
| 通信带宽效率 | Bandwidth-optimal (= NCCL) |
| Reduce 方式 | Comm Warps 内融合 FP32 reduce |

## 架构设计

### Warp 分工 (320 threads = 10 warps)

```
┌─────────────────────────────────────────────────────────────────┐
│ Warp 0 (32T): TMA Load — 加载 A/B tiles 到共享内存              │
│ Warp 1 (32T): MMA Issue — 执行 UMMA FMA → TMEM accumulator     │
│ Warp 2-3 (64T): Epilogue — TMEM → smem → local partial buffer  │
│                            + per-tile ready flag signaling       │
├─────────────────────────────────────────────────────────────────┤
│ Warp 4-7 (128T): Comm — Pull-based Reduce-Scatter              │
│                  - Poll per-tile ready flags from ALL ranks     │
│                  - NVLink P2P Read (pull remote partial)        │
│                  - FP32 accumulate → write final output         │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流（Tile 级流水线）

```
时间 →
                  Tile 0         Tile 1         Tile 2         ...
                  ──────         ──────         ──────
GEMM Warps:      [compute]      [compute]      [compute]
                       │              │              │
Epilogue:        [TMEM→smem→    [TMEM→smem→    [TMEM→smem→
                  local buf]     local buf]     local buf]
                  set flag_0     set flag_1     set flag_2
                       │              │              │
                       ↓              ↓              ↓
Comm Warps:      (waiting)      poll flag_0 →   poll flag_1 →
                               pull + reduce   pull + reduce
                               → output[0]     → output[1]
```

### M 维 Swizzle 调度

Rank i 计算 tile 的顺序：
1. 先计算属于 rank (i+1) 的 chunk → 写本地 partial + set flag
2. 再计算属于 rank (i+2) 的 chunk
3. ...
4. 最后计算属于自己 rank i 的 chunk

**目的**：接收方的 Comm Warps 能尽早开始从 peer 拉取数据，最大化 overlap 窗口。

### 同步机制

| 同步点 | 机制 | 说明 |
|--------|------|------|
| GEMM → Epilogue | `tmem_full` / `tmem_empty` barriers | TMEM 流水线（经典模式） |
| Epilogue → Comm (跨 rank) | `__threadfence_system()` + `st_rel_sys(flag, 1)` | Per-tile ready flag |
| Comm poll (跨 rank) | `ld_acq_sys(flag)` 自旋等待 | 检测远端 tile 就绪 |
| Kernel 结束 | `nvlink_barrier` | 确保所有 rank 完成 pull 后再 reset flags |

### 通信量分析

对于 N 个 rank，每个 rank 的输出大小为 `M_per_rank × N_dim`：

- **旧方案 (Symmetric Push)**: 总通信量 = `N × (N-1) × chunk_size` (全系统)
- **当前方案 (All-to-1 Pull)**: 总通信量 = `N × (N-1) × chunk_size / N` = `(N-1) × chunk_size`
  - 等同于 NCCL ring reduce-scatter 的 bandwidth-optimal 通信量

## 文件结构

```
=== 核心实现 ===
deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh   — 核心 kernel (~450行)
csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp              — JIT runtime + launch
csrc/jit_kernels/heuristics/gemm_rs.hpp                    — get_gemm_rs_config()
csrc/apis/gemm_rs.hpp                                      — C++ API: bf16_gemm_rs_nt()
deep_gemm/gemm_rs/__init__.py                              — Python API: bf16_gemm_rs_nt()

=== 测试 ===
tests/test_gemm_rs.py                                      — 正确性测试 (支持 2/4/8 GPU)
benchmarks/bench_gemm_rs.py                                — 对比: Fused vs GEMM+NCCL分离
```

## Python API

```python
import deep_gemm

# 创建对称缓冲区
sym_buffer = deep_gemm.get_symm_buffer_for_gemm_rs(
    group,                    # ProcessGroup
    num_max_tokens_per_rank,  # 最大 token 数
    hidden,                   # N 维度
    out_dtype=torch.bfloat16,
    comm_dtype=None,          # None=BF16, torch.float32=FP32
)

# GEMM+RS
deep_gemm.bf16_gemm_rs_nt(
    y,                    # [tokens_per_rank, N], output
    a,                    # [total_tokens, K], input (BF16)
    b,                    # [N, K], weight NT layout (BF16)
    sym_buffer,           # GemmRSSymmBuffer
    num_tokens_per_rank,  # 当前实际 token 数
    compiled_dims='nk',   # JIT 编译维度
)
```

## 运行测试

```bash
cd /root/.local/codebuddy/DeepGEMM

# 清除 JIT 缓存
rm -rf ~/.deep_gemm/cache/kernel.sm100_bf16_gemm_rs*

# 正确性测试 (2 GPUs)
python tests/test_gemm_rs.py 2

# 正确性测试 (8 GPUs)
python tests/test_gemm_rs.py 8

# 性能基准测试
python benchmarks/bench_gemm_rs.py 2 20
python benchmarks/bench_gemm_rs.py 8 20
```

> **注意**：不要用 `torchrun`，脚本内部用 `mp.spawn` 管理多进程。

## 调试指南

### 如果正确性不过

1. **检查 per-tile flag**: 加 `printf` 在 comm warps 看 flag 是否被正确设置
2. **检查 M-Swizzle**: 确认每个 rank 的 tile 计算顺序和 flag 索引对应
3. **检查 FP32 reduce**: 确认累加器初始化为 0，以及 rank 自己的 partial 也被加入
4. **单步验证**: 先固定 `num_ranks=2`，关闭 M-Swizzle 测试基本逻辑

### 如果 Hang（死锁）

1. **Comm Warps 永远等不到 flag**: 可能是 Epilogue 没有正确 `st_rel_sys`
2. **nvlink_barrier 卡住**: 可能某 rank 提前结束 → 检查所有 rank 是否执行相同数量的 barrier
3. **TMA Load 超时**: 远端内存地址错误 → 检查 `sym_buffer_ptrs` 的正确性

### 如果性能不达预期

1. **nsys profile**: `nsys profile python benchmarks/bench_gemm_rs.py 2 5`
2. **检查 SM 占用率**: 320 线程可能限制 occupancy → 考虑减少 Comm Warps
3. **检查 NVLink 利用率**: `nvidia-smi nvlink -i 0 -s` 查看 NVLink 吞吐
4. **Comm Warps 瓶颈**: 如果 pull 延迟高，考虑用 TMA Load 代替手动 global load

## 后续扩展方向

1. **TMA Load for Pull**: 用 TMA 的硬件异步 bulk load 代替 Comm Warps 的手动 load
2. **FP8 版本**: 扩展到 FP8 输入（需要 SF 处理逻辑）
3. **Multi-step Ring**: 参考 NCCL ring 的流水线，进一步减少延迟
4. **Adaptive Warp Split**: 根据 compute/comm ratio 动态分配 Warp 数量
