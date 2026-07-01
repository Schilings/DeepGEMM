# Fused QKV GEMM + RMSNorm + A2A-transpose 迭代记录

> 入口符号：`bf16_fused_qkv_norm_a2a_transpose_nt`
> 设计文档：`docs/FUSED_QKV_NORM_A2A_DESIGN.md`
> test / bench：`tests/comm/test_fused_qkv_norm_a2a.py` / `benchmarks/bench_fused_qkv_norm_a2a.py`

---

## 当前状态（接班看这里）

- **正确性**：8 卡 20/20 全 PASS（MHA+norm / MHA / GQA+norm / GQA / EXT 全组合）
  - norm 启用时 rel_err ~0.00006-0.00024（RMSNorm fp32 计算引入的 bf16 round）
  - norm 关闭时 rel_err == 0.0（纯 GEMM + A2A）
  - vs torch baseline diff == 0.0（同一数学路径）
- **性能**（Phase 2 v1，Python 编排 GEMM + norm + NCCL A2A）：
  - 8 卡 benchmark：fused vs serial ~parity（0.84x-1.85x，大部分 ~1.0x）
  - **预期**：Phase 2 v2 CUDA 融合（P2P TMA scatter 代替 NCCL）将带来 ~1.3-1.5x
- **分支**：`feat/fused-qkv-norm-a2a`

### 8 卡 benchmark（us，iters=20）

| Shape (bs,lseq,qh,kvh,hd,K) | Serial | Fused | Speedup |
|---|---:|---:|---:|
| 1,1024,40,8,128,5120 | 2416.5 | 2401.5 | 1.01x |
| 1,2048,40,8,128,5120 | 2393.2 | 2434.8 | 0.98x |
| 1,4096,40,8,128,5120 | 2422.8 | 2887.5 | 0.84x |
| 1,8192,40,8,128,5120 | 4264.4 | 4954.1 | 0.86x |
| 1,1024,32,32,128,4096 | 2379.1 | 1287.5 | 1.85x |
| 1,2048,64,64,128,8192 | 4033.6 | 4663.4 | 0.86x |
| 1,4096,32,32,128,5120 | 4023.3 | 4268.3 | 0.94x |
| 2,1024,40,8,128,5120 | 2400.5 | 1778.3 | 1.35x |

> 注：Phase 2 v1 的 fused 和 serial 底层都走 NCCL A2A + torch matmul，性能 ~parity。
> 真正加速来自 Phase 2 v2：用 P2P TMA scatter kernel 代替 NCCL（参考 bf16_gemm_a2a_transpose_nt 的 1.42x 加速比）。

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
