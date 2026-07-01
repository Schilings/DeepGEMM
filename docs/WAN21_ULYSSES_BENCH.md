# Wan2.1 14B Ulysses SP Attention Benchmark

真实 Wan2.1 14B 训练场景的 Ulysses 序列并行注意力 benchmark。

## 架构

```
benchmarks/wan21/
  config.py          — Wan21Config / SPConfig / TrainConfig (dataclass)
  model.py           — WanSelfAttention (nn.Module，纯模型层，与并行策略无关)
  rope.py            — 3D RoPE (Wan2.1 原生 T/H/W 三轴)
  norm.py            — WanRMSNorm (bf16)
  sp/
    base.py          — UlyssesBase 抽象基类 (forward/backward 接口)
    serial.py        — SerialUlysses (matmul + NCCL A2A，串行 baseline)
    fused_standard.py — FusedStandardUlysses (GEMM+A2A PRE + A2A+GEMM POST)
    fused_variant.py  — FusedVariantUlysses (GEMM+A2A PRE + GEMM+RS POST)
  fsdp2_utils.py     — FSDP2 权重梯度同步 (auto fully_shard / manual all-reduce)
  bench_utils.py     — 计时 + 正确性验证

benchmarks/bench_wan21_strategies.py — 统一入口
benchmarks/verify_wan21_attn.py      — 2卡快速正确性验证
```

## 三条策略

### 1. serial (Baseline)
- FWD: `matmul(X, Wqkv^T)` + NCCL `all_to_all` → FA4 → `all_to_all` + `matmul(gathered, Wo^T)`
- BWD: 串行 `matmul` + NCCL `all_to_all` (无 overlap)
- 权重梯度: all-reduce (FSDP2 fallback)

### 2. fused_std (融合标准 Ulysses)
- FWD PRE: `bf16_gemm_a2a_transpose_nt` (GEMM+A2A，融合 kernel)
- FWD POST: `bf16_a2a_transpose_gemm_nt_fused` (A2A+GEMM，融合 kernel)
- BWD POST: 对偶 GEMM+A2A (serial A2A-inverse + matmul)
- BWD PRE: 对偶 A2A+GEMM (`bf16_a2a_transpose_gemm_nt`, M0, 用 `Wqkv_t` 做 NT)
- 权重梯度: serial matmul (融合算子只算激活梯度)

### 3. fused_var (融合变体 Ulysses)
- FWD PRE: `bf16_gemm_a2a_transpose_nt` (GEMM+A2A)
- FWD POST: `bf16_gemm_rs_nt` (GEMM+RS，Wo 行拆分)
- BWD POST: `bf16_ag_gemm_nt` (AG+GEMM，GEMM+RS 的对偶)
- BWD PRE: 对偶 A2A+GEMM (`bf16_a2a_transpose_gemm_nt`, M0, 用 `Wqkv_t` 做 NT)
- 输出 N-sharded (每 rank 持有 N/sp 维度，省显存)

## 对偶关系

| Forward 算子 | Backward 对偶算子 |
|---|---|
| GEMM+A2A (`bf16_gemm_a2a_transpose_nt`) | A2A+GEMM (`bf16_a2a_transpose_gemm_nt_fused`) |
| A2A+GEMM (`bf16_a2a_transpose_gemm_nt_fused`) | GEMM+A2A (`bf16_gemm_a2a_transpose_nt`) |
| GEMM+RS (`bf16_gemm_rs_nt`) | AG+GEMM (`bf16_ag_gemm_nt`) |

> 注: 融合算子只能 overlap 计算激活值梯度，权重梯度需要完整输入无法 overlap。

## 输入 Shape (THD / PackedSequence)

Wan2.1 14B: dim=5120, nh=40, hd=128。VAE stride=(4,8,8), patch=(1,2,2)。
81 帧 → T_latent=21。

| Shape 标签 | Token 数 | 对应场景 |
|---|---|---|
| 1x8K | 8,192 | 小规模测试 |
| 1x32K | 32,768 | 480p 81帧 |
| 1x74K | 75,776 | 720p 81帧 |
| 1x168K | 172,032 | 1080p 81帧 |
| 1x64K | 65,536 | 480p × 2视频 |
| 1x148K | 151,552 | 720p × 2视频 |

## 结果 (8 GPU B300, THD, iters=10, us)

| Shape | Strategy | FWD | ATTN | BWD | FWD+BWD |
|-------|----------|------|------|------|---------|
| 1x8K | serial | 1506 | 230 | 3316 | 4821 |
| 1x8K | fused_std | 2849 | 224 | 3622 | 6470 |
| 1x8K | **fused_var** | **920** | 223 | **3414** | **4335** |
| 1x32K | serial | 3876 | 1668 | 14385 | 18261 |
| 1x32K | fused_std | 4878 | 1671 | 14253 | 19131 |
| 1x32K | **fused_var** | **3222** | 1719 | **13785** | **17008** |
| 1x74K | serial | 13046 | 8074 | 61699 | 74746 |
| 1x74K | fused_std | 12903 | 7901 | 61261 | 74163 |
| 1x74K | **fused_var** | **12066** | 8911 | **59906** | **71972** |
| 1x168K | serial | 59441 | 47588 | 282165 | 341607 |
| 1x168K | fused_std | 57747 | 47306 | 279895 | 337642 |
| 1x168K | **fused_var** | **56916** | 45493 | **277148** | **334064** |
| 1x64K | serial | 10212 | 6380 | 48445 | 58657 |
| 1x64K | fused_std | 9476 | 6269 | 48137 | 57613 |
| 1x64K | **fused_var** | 9587 | 6369 | **46870** | **56456** |
| 1x148K | serial | 46853 | 33469 | 221689 | 268541 |
| 1x148K | fused_std | 45135 | 36431 | 221384 | 266519 |
| 1x148K | **fused_var** | **44665** | 36014 | **219174** | **263839** |

### 加速比 (FWD+BWD vs serial)

| Shape | fused_std | fused_var |
|-------|-----------|-----------|
| 1x8K | 0.75x | **1.11x** |
| 1x32K | 0.95x | **1.07x** |
| 1x74K | 1.01x | **1.04x** |
| 1x168K | 1.01x | **1.02x** |
| 1x64K | 1.02x | **1.04x** |
| 1x148K | 1.01x | **1.02x** |

## 正确性验证 (2 GPU, 1x8K, serial)

```
FWD rel       = 0.0019
BWD grad_X    = 0.0016
BWD grad_Wqkv = 0.0034
Status: PASS
```

## 用法

```bash
# 全部策略
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python benchmarks/bench_wan21_strategies.py 8 10

# 指定策略
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python benchmarks/bench_wan21_strategies.py 8 10 --strategies serial,fused_var

# 正确性验证 (2卡)
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD python benchmarks/bench_wan21_strategies.py 2 5 --verify --strategies serial
```

## 关键发现

1. **fused_var 全面最优** — 所有 shape 的 FWD+BWD 都最快
2. **长序列 attention 占比大** — 1080p 的 ATTN 占 FWD 的 80%，e2e 加速被稀释
3. **fused_std 在小 shape 反而慢** — 融合算子固定开销在 1x8K 不划算
4. **batch=2 对 fused 有利** — 更大 GEMM 更好 overlap comm
5. **BWD 的 PRE 部分已改用融合 A2A+GEMM 算子**（`bf16_a2a_transpose_gemm_nt`, M0, 用 `Wqkv_t` 做 NT）— 正确性验证通过（gX_rel=0.0016 vs serial NCCL），但因 PRE BWD 的 comm 数据量小（QKV 每 rank ~2MB），M0 对 BWD 整体加速不显著（±1%，大 shape 略快、小 shape 略慢）。BWD 主要被 FA4 backward 和 weight grad（串行 matmul）主导。
