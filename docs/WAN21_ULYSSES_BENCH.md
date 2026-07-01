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

## 结果 (8 GPU B300, THD, iters=10, us, BWD 含梯度同步)

> BWD 现在包含完整的梯度同步：Wqkv 总是 all-reduce；Wo 对 serial/fused_std all-reduce，对 **fused_var 不同步**（Wo 行切分，梯度天然本地）。

| Shape | Strategy | FWD | ATTN | BWD | FWD+BWD |
|-------|----------|------|------|------|---------|
| 1x8K | serial | 1514 | 230 | 4022 | 5536 |
| 1x8K | fused_std | 2813 | 221 | 4621 | 7434 |
| 1x8K | **fused_var** | **919** | 222 | **4000** | **4919** |
| 1x32K | serial | 3890 | 1664 | 14966 | 18856 |
| 1x32K | fused_std | 4598 | 1673 | 14984 | 19583 |
| 1x32K | **fused_var** | **3238** | 1688 | **14345** | **17584** |
| 1x74K | serial | 13012 | 8723 | 62078 | 75089 |
| 1x74K | fused_std | 12642 | 8582 | 61869 | 74511 |
| 1x74K | **fused_var** | **11905** | 9003 | **60500** | **72405** |
| 1x168K | serial | 58929 | 46981 | 283291 | 342220 |
| 1x168K | fused_std | 57520 | 47488 | 283287 | 340807 |
| 1x168K | **fused_var** | **57256** | 46863 | **279157** | **336413** |
| 1x64K | serial | 10129 | 6351 | 49055 | 59184 |
| 1x64K | fused_std | 9673 | 6354 | 48920 | 58593 |
| 1x64K | **fused_var** | **9408** | 6724 | **47206** | **56614** |
| 1x148K | serial | 46719 | 36699 | 222565 | 269284 |
| 1x148K | fused_std | 45324 | 35359 | 221561 | 266886 |
| 1x148K | **fused_var** | **43930** | 36979 | **219173** | **263104** |

### 加速比 (FWD+BWD 含梯度同步, vs serial)

| Shape | fused_std | fused_var |
|-------|-----------|-----------|
| 1x8K | 0.74x | **1.13x** |
| 1x32K | 0.96x | **1.07x** |
| 1x74K | 1.01x | **1.04x** |
| 1x168K | 1.00x | **1.02x** |
| 1x64K | 1.01x | **1.05x** |
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
6. **fused_var 的 Wo 行切分省掉 Wo 梯度 all-reduce** — serial/fused_std 每步 BWD 需 all-reduce Wqkv+Wo 两个权重梯度；fused_var 只 all-reduce Wqkv，Wo 梯度天然本地（行切分，每 rank 只更新自己的块）。这是 variant 在梯度同步阶段的额外优势。
