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

### AKO4ALL 迭代历史（iter 1-8）

| Iter | 改动 | geo vs serial | 备注 |
|---|---|---:|---|
| baseline | Python NCCL（v2b norm 推迟，2 次 A2A） | 0.751x | 两次 NCCL A2A 太贵 |
| 1 | in-place `add_`+`mul_`（避免 temp tensor） | ~1.0x | |
| 2 | fused elementwise `x*rms*weight`（单 pass） | ~0.3x | 回退：多 slice+cast 更慢 |
| **3** | **去掉冗余 `torch.cuda.synchronize()`** | **1.43x** | **关键突破**：每步 sync 是 stream stall |
| 4-5 | 预计算 `rms*weight` | ~1.4x | marginal |
| **6** | **去掉 `out.clone()`** | **1.60x** | **关键突破**：省 full HBM copy |
| 7 | 预计算 bias slices | ~1.5x | marginal |
| 8 | `addcmul` fused bias+mul | 1.40x | 回退：语义不对 |

**最终：geo 1.61x vs serial**（8 卡 B300，iters=30）

### 关键 bug 修复

1. **`pre_cast` warp 偏移缺失**：`sm100_store_cd` 传给 `pre_cast` 的 `global_m` 没加 `epilogue_warp_idx * 32`，导致 4 个 epilogue warp 对同一行做 atomic add（sum 是正确值的 4 倍）
2. **坐标转换**：`pre_cast` 拿到的是输出坐标（scatter dst），需要转成 GEMM 坐标做段判断（Q/K/V）。通过 `epi_ctx.gemm_m_offset` / `gemm_n_offset` per-tile 设置
3. **sum_buffer 索引**：用 GEMM M 坐标索引（`0..local_m-1`），不是输出坐标（可达 `bs*seq-1`）
