# GEMM-RS 设计说明（唯一主线）

> 本文档只描述当前有效实现：`bf16_gemm_rs_nt`。
> 历史探索内容已归档，不作为当前决策依据。

---

## 1. 目标

在 SM100 路径上提供稳定可复现的 **GEMM + Reduce-Scatter 融合**，优先保证：

1. 多卡正确性稳定
2. benchmark 可复现
3. 在可复现基线之上持续迭代性能

---

## 2. 主线实现（融合 GEMM-RS：epilogue push-scatter + flagless 本地 reduce，dual-kernel）

> 本节为当前 `main` 的有效实现。早期曾尝试「pull + per-tile flag + 本地 scatter」结构，已被下述
> push-scatter + flagless 流式 reduce 取代（演进过程见 `GEMM_RS_ITERATION.md` Iter 8/10/14）。

- **算子入口**：`deep_gemm.bf16_gemm_rs_nt`（唯一实现，无 impl 开关）。
- **Kernel 1（GEMM + push-scatter）**：`deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm_rs.cuh`
  - 256T(8 warps)，无 comm warps（消除旧 384T 的寄存器 spilling）；
  - epilogue **push-scatter（核心收益）**：每个 tile 按 `dst_rank`，用**整块 2D TMA store**
    （`SM90_TMA_STORE_2D`，复用标准 `epilogue::sm100_store_cd`）**跨 NVLink 直推**到 `dst_rank` 的
    symmetric scatter slot[my_rank]；self（dst==my_rank）即本地 store。跨卡 store 由 TMA 引擎异步发起，
    **与后续 tile 的 MMA 重叠**——这是单机 NVLink 上让 RS 藏进 compute、从而 >1.0x 的关键。
  - **flagless**：不再用 per-tile ready flag；所有 push 的全局可见性由 kernel 末尾的
    **system-scope `nvlink_barrier`**（release/acquire）保证。
- **Kernel 2（RS reduce, flagless 本地累加）**：`deep_gemm/include/deep_gemm/impls/sm100_rs_reduce.cuh`
  - push 完成后，rank R 的本地 buffer 已聚齐 R 份各 src 推来的 partial（slot[0..R-1] 各 `[m_per_rank x N]`）；
  - reduce = **纯 1D 连续流式**遍历 `m_per_rank*N`，对各 src slot 做 FP32 累加 → output；
    无 flag/poll/`__syncthreads`；`kUnroll` 随 rank 数自适应（`kNumRanks>=8?2:>=4?4:8`）防 8 卡 spilling。
- **JIT 运行时**：`csrc/jit_kernels/impls/sm100_bf16_gemm_rs.hpp`
  - dual-kernel 编排：`compute_stream` 跑 GEMM、`comm_stream` 跑 reduce、CUDA event 同步实现流级排序；
  - host 为每个 dst_rank 建一个 scatter slot 的 `CUtensorMap`（`make_tma_2d_desc_raw`，self=本地 base、
    peer=P2P-mapped base），打包成 `GemmRSScatterMaps` 传入 GEMM kernel 供 2D TMA push。
- **RS reduce 运行时**：`csrc/jit_kernels/impls/sm100_rs_reduce.hpp`。
- **Python 接口**：`deep_gemm/gemm_rs/__init__.py`。
- **测试**：`tests/test_gemm_rs.py`、`tests/test_gemm_rs_quick.py`；**性能脚本**：`benchmarks/bench_gemm_rs.py`。

### 数据契约

- 每 rank 计算完整 partial `[total_m=num_ranks*m_per_rank, N]`，按 M 切成 num_ranks 个 chunk（chunk-for-d 属于 dst_rank d）。
- GEMM(r) epilogue 把 chunk-for-d **TMA-store 推到** dst_rank d 的 buffer `slot[r]`（跨卡 P2P，self 为本地）。
- Reduce(R) 读本地已聚齐的 `slot[0..R-1]`（各 src 推来的、目标为 R 的 partial）求和 → output。
- 跨迭代正确性：GEMM 起始 `nvlink_barrier` + host event 门控，保证「上一轮所有 reduce 完成 → 本轮 push 开始」，
  GEMM 末尾 system-scope `nvlink_barrier` 保证「本轮所有 push 全局可见 → reduce 读取」。

> 通信通道选型：单机 NVLink 下把跨卡传输**融进 GEMM epilogue 用 TMA async store**（push），
> 与 MMA 重叠；reduce 退化为纯本地累加。这与 Flux「让 comm overlap compute」同源，但因 SM100 上
> GEMM 独占 SM 寄存器、分离 reduce kernel 无法共驻 overlap，故选择 push 进 epilogue 而非 pull。

---

## 3. 评测口径（统一）

只保留以下对比：

- `Separate`：`bf16_gemm_nt + torch.distributed.reduce_scatter_tensor`
- `Main Fused`：`bf16_gemm_rs_nt`

不再在主文档中维护多版本横向对比术语，避免误导。

---

## 4. 运行与验证

```bash
cd /root/.local/codebuddy/DeepGEMM
python3 setup.py build_ext --inplace --force

# 正确性
DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python tests/test_gemm_rs.py 2

# 性能
DG_BENCH_MAX_SHAPES=3 DG_JIT_USE_NVRTC=1 PYTHONPATH=/root/.local/codebuddy/DeepGEMM \
python benchmarks/bench_gemm_rs.py 2 3
```

可选：
- `DG_BENCH_SINGLE_SHAPE=M,N,K`
- `DG_BENCH_SYNC_EACH_ITER=1`

---

## 5. 当前设计原则

1. **先稳后快**：先保证正确性和可复现，再做激进优化。
2. **最小化分叉**：只维护唯一主线路径，减少并行分支维护成本。
3. **结果驱动**：所有结论以本机可复现测试和 benchmark 输出为准。
