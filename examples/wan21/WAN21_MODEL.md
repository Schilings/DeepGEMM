# Wan2.1 Model Implementation

Complete, faithful implementation of the official [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) model.

## 架构

```
examples/wan21/
  model.py          — Complete WanModel (patch/text/time embedding + blocks + head)
  norm.py           — WanRMSNorm (QK norm), WanLayerNorm (pre-norm)
  rope.py           — 3D RoPE (T/H/W axes, complex, float64)
  config.py         — Wan21Config dataclass
  autograd_ops.py   — torch.autograd.Function wrappers for fused ops
  fsdp2_utils.py    — FSDP2 (fully_shard) integration
  sp/               — SP strategies (serial, fused_standard, fused_variant)
```

## 官方权重验证

### 14B 完整 forward ✅

```bash
python examples/verify_wan21_14b_full.py
```

- 下载全部 6 shards（1095 keys，14.29B params）
- 权重 key 100% 对齐官方 checkpoint
- 完整 forward: `[C=16, F=4, H=64, W=64]` → `[C=16, F=4, H=64, W=64]` PASS
- 2026-07-19 BF16 实测输出: mean=0.072576, std=0.133118, norm=77.6277

### 对齐官方的关键点

| 项目 | 实现 |
|------|------|
| Q/K/V | 分开 `nn.Linear(dim, dim)` with bias=True |
| FFN | `Sequential(Linear, GELU(tanh), Linear)` with bias |
| RMSNorm | 参数名 `weight`（对齐 checkpoint key） |
| norm3 | LayerNorm with `elementwise_affine=True` |
| Cross-attn | T2VCrossAttention（text context → video） |
| Modulation | `[1, 6, dim]` float32（time embedding e） |
| RoPE | 3D (T/H/W), float64 complex, split `[d-4*(d//6), 2*(d//6), 2*(d//6)]` |
| Patch embedding | `nn.Conv3d(in_dim, dim, patch_size, stride=patch_size)` |
| Head | LayerNorm + modulation + Linear → unpatchify |
| 权重 dtype | float32（官方 checkpoint） |
| FA4 输入 | attention 前 cast bf16 |

## Wan2.1 14B 配置

```
dim=5120, num_heads=40, head_dim=128, ffn_dim=13824, num_layers=40
patch_size=(1,2,2), text_len=512, in_dim=16, freq_dim=256, text_dim=4096, out_dim=16
```

## 官方 14B 权重训练核心吞吐

POST backward 的独立测试、NVTX 和 Nsight Systems 操作见 `docs/ULYSSES_POST_BWD_PROFILING.md`。

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD/examples:$PWD \
python examples/ulysses_variant/bench_wan21_14b_train.py 8 \
  --layers 40 --seq 8192 --warmup 2 --iters 5 \
  --strategies serial,fused_var --sync-mode manual
```

该入口默认下载/加载官方权重；只有显式传 `--synthetic` 才使用随机权重。它运行 40 个完整 Transformer block（14.056B 参数），只排除 patch/text/time embedding、输出 head 和 optimizer step。

B300×8、SP=8、8K 实测：

| Sync | Baseline | Variant | Variant/Baseline |
|---|---:|---:|---:|
| manual bucketed | 24,486.2 tok/s | 23,988.2 tok/s | 0.9797x (-2.03%) |
| DDP overlapped（最终 wall-clock） | 29,249.9 tok/s | 28,193.0 tok/s | 0.9639x (-3.61%) |

最终口径使用 warmup=3、iters=10、同步后的 rank-max wall-clock，并用两个 symmetric-memory 设备端 barrier 分别保证 AG 输入发布和 peer 消费完成；SP=8 数值验证 `grad_X rel=0`。剩余差距主要来自 AG 通信本体：每 rank 单向远端 payload 为 70 MiB，而 baseline A2A 为 8.75 MiB，正好 8×。

### SP 梯度同步语义

- baseline 所有复制参数（包括完整 Wo）都需要跨 SP reduce。
- variant 除 `Wo_r_local` 外的参数仍需跨 SP reduce；Wo 每个 rank 持有不同输入列 shard，backward 已 AG 完整 `grad_y`，因此本地 dW 已完整且不能在 SP group 互相归约。
- 若存在 DP，Wo shard 仍需在相同 SP 坐标的 DP group 同步。普通 DDP 应用于 DP group；实验的 SP-DDP 模式会显式排除 `_sp_sharded` Wo。
