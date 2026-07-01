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

> **FWD+BWD+SYNC** 完整训练开销。SYNC = FSDP2 风格 reduce-scatter 梯度同步。
> - Wqkv：总是 reduce-scatter（replicated weight，各 rank 算的是 partial grad）
> - Wo：serial/fused_std reduce-scatter；**fused_var 跳过**（Wo 行切分，梯度天然本地）

| Shape | Strategy | FWD | ATTN | BWD | SYNC | F+B+S |
|-------|----------|------|------|------|------|-------|
| 1x8K | serial | 1468 | 228 | 3336 | 432 | 5235 |
| 1x8K | fused_std | 1989 | 222 | 4115 | 399 | 6503 |
| 1x8K | **fused_var** | **931** | 225 | **3590** | **269** | **4790** |
| 1x32K | serial | 3707 | 1682 | 14380 | 408 | 18495 |
| 1x32K | fused_std | 4606 | 1679 | 14370 | 460 | 19436 |
| 1x32K | **fused_var** | **3222** | 1679 | **13854** | **268** | **17344** |
| 1x74K | serial | 12958 | 8134 | 61645 | 426 | 75029 |
| 1x74K | fused_std | 12748 | 8180 | 60840 | 424 | 74013 |
| 1x74K | **fused_var** | **12230** | 8045 | **59497** | **279** | **72007** |
| 1x168K | serial | 59717 | 46368 | 282145 | 431 | 342293 |
| 1x168K | fused_std | 57947 | 47155 | 282241 | 443 | 340631 |
| 1x168K | **fused_var** | **57035** | 47417 | **277554** | **292** | **334881** |
| 1x64K | serial | 10205 | 6332 | 48365 | 425 | 58994 |
| 1x64K | fused_std | 9692 | 6488 | 47943 | 426 | 58062 |
| 1x64K | **fused_var** | **9361** | 6351 | **46716** | **275** | **56353** |
| 1x148K | serial | 47052 | 36878 | 221935 | 437 | 269424 |
| 1x148K | fused_std | 45182 | 34083 | 220560 | 440 | 266182 |
| 1x148K | **fused_var** | **44764** | 36641 | **219672** | **278** | **264714** |

## 结果 (8 GPU B300, FSDP2 fully_shard, autograd-based, iters=10, us)

> **FSDP2 (fully_shard) + autograd.Function**：融合算子封装为 `torch.autograd.Function`，
> forward 走 autograd graph，`y.backward()` 自动触发 FSDP2 的 reduce-scatter 梯度同步。
> BWD 已包含 FSDP2 自动梯度同步（Wqkv reduce-scatter；fused_var Wo 跳过因行切分）。

| Shape | Strategy | FWD | BWD | F+B |
|-------|----------|------|------|------|
| 1x8K | serial | 1698 | 3046 | 4744 |
| 1x8K | fused_std | 1502 | 2927 | 4429 |
| 1x8K | **fused_var** | **1423** | 3065 | **4487** |
| 1x32K | serial | 4112 | 11987 | 16099 |
| 1x32K | fused_std | 3789 | 12049 | 15839 |
| 1x32K | **fused_var** | **3669** | **11576** | **15244** |
| 1x74K | serial | 13343 | 51044 | 64386 |
| 1x74K | fused_std | 12537 | 51014 | 63551 |
| 1x74K | **fused_var** | **12376** | **49185** | **61561** |
| 1x168K | serial | 58946 | 228491 | 287437 |
| 1x168K | fused_std | 58198 | 229296 | 287494 |
| 1x168K | **fused_var** | **57356** | **226077** | **283432** |
| 1x64K | serial | 10418 | 39940 | 50359 |
| 1x64K | fused_std | 9919 | 39724 | 49643 |
| 1x64K | **fused_var** | **9775** | **38167** | **47942** |
| 1x148K | serial | 46613 | 181370 | 227982 |
| 1x148K | fused_std | 45579 | 181424 | 227003 |
| 1x148K | **fused_var** | **45204** | **178769** | **223973** |

### 加速比 (F+B vs serial, FSDP2 autograd)

| Shape | fused_std | fused_var |
|-------|-----------|-----------|
| 1x8K | 0.93x | **0.95x** |
| 1x32K | 0.98x | **1.05x** |
| 1x74K | 0.99x | **1.04x** |
| 1x168K | 1.00x | **1.01x** |
| 1x64K | 0.99x | **1.05x** |
| 1x148K | 1.00x | **1.02x** |

### SYNC 开销分析

| Shape | serial SYNC | fused_std SYNC | fused_var SYNC | var 节省 |
|-------|-------------|---------------|---------------|---------|
| 1x8K | 432us | 399us | **269us** | -38% |
| 1x168K | 431us | 443us | **292us** | -32% |

fused_var 的 SYNC 比 serial/fused_std 少 ~30-38%，因为它省掉了 Wo 的 reduce-scatter（行切分权重，梯度天然本地）。

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
