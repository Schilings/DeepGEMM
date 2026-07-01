# Fused QKV GEMM + RMSNorm + A2A-transpose 迭代记录

> 入口符号：`bf16_fused_qkv_norm_a2a_transpose_nt`
> 设计文档：`docs/FUSED_QKV_NORM_A2A_DESIGN.md`
> test / bench：`tests/comm/test_fused_qkv_norm_a2a.py` / `benchmarks/bench_fused_qkv_norm_a2a.py`

---

## 当前状态（接班看这里）

- **正确性**：8 卡 20/20 全 PASS（MHA+norm / MHA / GQA+norm / GQA / EXT 全组合）
- **架构改动**：扩展 `epilogue::transform` 接口，加 `pre_cast` hook（向后兼容，现有算子零影响）
  - `EpilogueIdentity::pre_cast` = no-op
  - `EpilogueX2Sum::pre_cast` = 在 STSM cast 前算 x² partial sum + atomic add
- **v2b norm 推迟方案**（Python 编排）：正确性 PASS，但性能 0.751x（两次 NCCL A2A 太贵）
- **关键结论**：Python 编排的 norm 推迟没有优势，必须 CUDA 融合才能加速
- **分支**：`feat/fused-qkv-norm-a2a`

### 8 卡 benchmark（us，iters=20）：serial vs v2b(norm 推迟)

| Shape | Serial | v2b | v2b/serial |
|---|---:|---:|---:|
| 1,1024,40,8,128,5120 | 1136.2 | 1479.0 | 0.77x |
| 1,2048,40,8,128,5120 | 1068.1 | 1249.0 | 0.86x |
| 1,4096,40,8,128,5120 | 2114.2 | 3171.4 | 0.67x |
| 1,8192,40,8,128,5120 | 2954.1 | 3749.0 | 0.79x |
| 1,1024,32,32,128,4096 | 680.4 | 951.3 | 0.72x |
| 1,2048,64,64,128,8192 | 3530.1 | 3194.1 | 1.11x |
| 1,4096,32,32,128,5120 | 1897.2 | 2930.1 | 0.65x |
| 2,1024,40,8,128,5120 | 1475.5 | 2577.8 | 0.57x |

**Geo-mean: 0.751x**（v2b 慢于 serial，因两次 NCCL A2A）

> 结论：norm 推迟方案正确，但 Python 编排下额外 A2A 开销 > 节省的 norm 开销。
> 下一步：CUDA 融合——在 GEMM epilogue 里用 `pre_cast` 算 x²sum + P2P TMA scatter，
> rms 用单独轻量 kernel 或融进 epilogue，避免第二次 NCCL A2A。

---

## 迭代历史

### 2026-07-01 v1（Phase 2 v1：Python 编排）

- **设计**：支持 norm 可选 + GQA，API 为 `bf16_fused_qkv_norm_a2a_transpose_nt`
- **实现**：Python 编排（torch.matmul + RMSNorm + NCCL all_to_all_single）
  - Q/K/V 分段各自按 head 组 A2A scatter
  - norm_q_weight/norm_k_weight=None 时跳过 RMSNorm
  - bias 可选
- **正确性**：8 卡 20/20 PASS
- **性能**：~parity（预期，底层同走 NCCL）
- **下一步**：Phase 2 v2 — 实现 CUDA kernel（sm100_bf16_rmsnorm_a2a_scatter），
  用 P2P TMA scatter 代替 NCCL A2A

### 参考算子

- `sm100_tf32_hc_prenorm_gemm`：GEMM + sqr_sum reduce 模式（cast-and-reduce warps）
- `bf16_gemm_a2a_transpose_nt`：GEMM + A2A scatter epilogue（P2P TMA push）
- `flux/src/gemm_a2a_transpose`：Hopper pull-based 参考（per-tile flag）
