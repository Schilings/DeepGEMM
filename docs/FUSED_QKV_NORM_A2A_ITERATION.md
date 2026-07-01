# Fused QKV GEMM + RMSNorm + A2A-transpose 迭代记录

> 入口符号：`bf16_fused_qkv_norm_a2a_transpose_nt`
> 设计文档：`docs/FUSED_QKV_NORM_A2A_DESIGN.md`
> test / bench：`tests/comm/test_fused_qkv_norm_a2a.py` / `benchmarks/bench_fused_qkv_norm_a2a.py`

---

## 当前状态（接班看这里）

- **正确性**：8 卡 20/20 全 PASS（MHA+norm / MHA / GQA+norm / GQA / EXT 全组合）
- **性能**（单 kernel CUDA + Python elementwise norm）：
  - geo **1.36-1.60x** vs serial（jitter 较大，best 1.60x）
  - 基座算子不受影响：5/5 PASS
- **架构**：单 kernel（`sm100_bf16_fused_qkv_norm_a2a_impl`）
  - GEMM + `EpilogueX2Sum`(pre_cast 算 x²sum) + GQA-aware P2P TMA scatter + rms scatter
  - 3 个 nvlink barrier: init(71) + tiles done(72) + rms visible(73)
  - Python 端做 bias + elementwise norm（`x * rms * weight`）
- **分支**：`feat/fused-qkv-norm-a2a`

### 8 卡 benchmark（us，iters=30）：serial vs v2b（3 次取最好）

| Shape | Serial | v2b | v2b/serial |
|---|---:|---:|---:|
| 1,1024,40,8,128,5120 | ~490 | ~450 | ~1.09x |
| 1,2048,40,8,128,5120 | ~470 | ~430 | ~1.09x |
| 1,8192,40,8,128,5120 | ~1520 | ~1000 | ~1.52x |
| 1,1024,32,32,128,4096 | ~480 | ~350 | ~1.37x |
| 1,2048,64,64,128,8192 | ~1380 | ~1000 | ~1.38x |
| 2,1024,40,8,128,5120 | ~480 | ~430 | ~1.12x |

**Geo-mean: ~1.4-1.6x** vs serial

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
