# Standard Ulysses Baseline vs DeepGEMM Fused Benchmark

本目录只比较**标准 Ulysses**的两条等价 forward 路径：

| 路径 | PRE | Attention | POST |
|---|---|---|---|
| `baseline` | BF16 `torch.matmul` + 同步 NCCL `all_to_all_single` | FlashAttention-4 | 同步 NCCL `all_to_all_single` + BF16 `torch.matmul` |
| `fused` | `bf16_gemm_a2a_transpose_nt` | FlashAttention-4 | `bf16_a2a_transpose_gemm_nt_fused` |

两条路径都持有完整、复制的 Q/K/V/Wo 方阵权重，并实现相同的数据布局变换。目录中不包含其他实验策略。

## 1. 数据流

每个 sequence-parallel rank：

```text
X_local[bs, local_seq, hidden]
  -- PRE: QKV projection + heads/sequence A2A transpose
  --> q,k,v[bs, seq, local_nheads, head_dim]
  -- FlashAttention-4
  --> attention output
  -- POST: heads/sequence A2A transpose + full Wo projection
  --> y[bs * local_seq, hidden]
```

权重：

```text
Wq/Wk/Wv: [hidden, hidden]
Wqkv:     [3 * hidden, hidden]
Wo:       [hidden, hidden]
hidden = nheads * head_dim
```

## 2. Benchmark 口径

脚本：

```text
examples/ulysses_fused/bench_ulysses_full_attn_flow.py
```

运行：

```bash
DG_JIT_USE_NVRTC=1 \
PYTHONPATH=$PWD/examples:$PWD \
PYTHONWARNINGS=ignore \
python3 examples/ulysses_fused/bench_ulysses_full_attn_flow.py 8 10
```

计时方法：

- 每个组件 warmup 3 次；
- 每次 measured iteration 前做跨 rank barrier；
- CUDA Event 记录本 rank GPU 时间；
- 每次对 elapsed time 做跨 rank MAX；
- 表中单位为 microseconds；
- Attention 两臂完全相同，每个 shape 只计时一次。

`chain` 是独立计时的 `PRE + ATTN + POST` 之和，用于估算标准 attention forward 链路；当前脚本不是张量真实串联的 autograd 训练 benchmark。

加速比：

```text
e2e = chain_baseline / chain_fused
c+g = (PRE_baseline + POST_baseline) / (PRE_fused + POST_fused)
```

其中 `c+g` 更直接反映两个通信融合算子的收益；长序列下 FA4 attention 占比很高，因此 e2e 收益会被稀释。

## 3. BSHD 与 THD

每个原始 shape `(bs, nheads, seq, head_dim)` 以两种布局运行：

- BSHD：`bs × seq`；
- THD：把同样 token 打包为 `1 × (bs*seq)`。

对于 `bs=1`，二者实际 shape 相同；对于 `bs>1`，THD 行用于验证相同 token 总数下的 packed layout。本文不测试变长序列免 padding 收益。

## 4. B300×8 结果

环境：

```text
GPU: NVIDIA B300 SXM6 AC ×8
Dtype: BF16
Attention: FlashAttention-4
Iterations: 10 per component
Timing: rank-max CUDA Event
```

列格式：`fused/baseline`，时间单位 us。

| Shape | Layout | PRE f/base | ATTN | POST f/base | Chain f/base | e2e | c+g |
|---|---|---:|---:|---:|---:|---:|---:|
| h4096 nh32 1×32K L4K | BSHD | 334/551 | 1390 | 348/258 | 2072/2199 | 1.06× | 1.19× |
| h4096 nh32 1×32K L4K | THD | 335/561 | 1390 | 343/259 | 2068/2209 | 1.07× | 1.21× |
| h8192 nh64 1×32K L4K | BSHD | 1325/1601 | 2483 | 895/742 | 4703/4826 | 1.03× | 1.06× |
| h8192 nh64 1×32K L4K | THD | 1327/1613 | 2483 | 902/657 | 4712/4753 | 1.01× | 1.02× |
| h8192 nh64 1×64K L8K | BSHD | 2632/3771 | 11381 | 1806/1231 | 15819/16383 | 1.04× | 1.13× |
| h8192 nh64 1×64K L8K | THD | 2632/3885 | 11381 | 1821/1230 | 15834/16495 | 1.04× | 1.15× |
| h4096 nh32 1×128K L16K | BSHD | 1219/2178 | 23487 | 1327/885 | 26033/26549 | 1.02× | 1.20× |
| h4096 nh32 1×128K L16K | THD | 1209/2174 | 23487 | 1311/885 | 26007/26546 | 1.02× | 1.21× |
| h4096 nh32 2×32K L4K | BSHD | 628/1031 | 2541 | 677/467 | 3846/4039 | 1.05× | 1.15× |
| h4096 nh32 1×64K L8K | THD | 632/1032 | 2541 | 675/455 | 3849/4028 | 1.05× | 1.14× |
| h5120 nh40 1×32K L4K | BSHD | 530/733 | 1734 | 439/388 | 2702/2855 | 1.06× | 1.16× |
| h5120 nh40 1×32K L4K | THD | 517/736 | 1734 | 494/330 | 2745/2799 | 1.02× | 1.05× |
| h5120 nh40 1×74K L9472 | BSHD | 1147/1725 | 9428 | 1045/715 | 11620/11868 | 1.02× | 1.11× |
| h5120 nh40 1×74K L9472 | THD | 1147/1716 | 9428 | 1038/749 | 11613/11893 | 1.02× | 1.13× |
| h2048 nh16 1×32K L4K | BSHD | 167/262 | 705 | 194/141 | 1067/1108 | 1.04× | 1.11× |
| h2048 nh16 1×32K L4K | THD | 166/259 | 705 | 231/142 | 1102/1107 | 1.00× | 1.01× |
| h2048 nh16 1×74K L9472 | BSHD | 299/501 | 3397 | 513/245 | 4209/4143 | 0.98× | 0.92× |
| h2048 nh16 1×74K L9472 | THD | 306/501 | 3397 | 446/243 | 4149/4142 | 1.00× | 0.99× |

几何均值：

| Layout | Chain speedup | PRE+POST speedup |
|---|---:|---:|
| BSHD | **1.032×** | **1.111×** |
| THD | **1.026×** | **1.098×** |

## 5. 结果解读

1. 融合 PRE 稳定快于 baseline；它是当前主要收益来源。
2. 当前融合 POST 在部分大 hidden/长序列 shape 上慢于 NCCL A2A + cuBLAS GEMM，抵消了一部分 PRE 收益。
3. 标准 Ulysses forward 链路的几何平均收益约为 2.6%～3.2%。
4. 只看 PRE+POST，几何平均收益约为 9.8%～11.1%。
5. `h2048 nh16 74K` 是当前短板，融合路径略慢，需要单独 profile POST kernel 与 baseline NCCL/cuBLAS。
6. 由于 attention 在长序列下占主要时间，即使通信融合部分提升约 10%，chain e2e 也只提升约 3%。

## 6. 正确性

标准 Ulysses 测试入口：

```bash
DG_JIT_USE_NVRTC=1 PYTHONPATH=$PWD \
python3 tests/ulysses/test_ulysses_full_attn_flow.py 8
```

2026-07-20 在 B300×8 实跑 3 个 shape，结果 3/3 PASS：

| Shape | Relative error | PRE | POST | PRE+POST |
|---|---:|---:|---:|---:|
| `(1,32,2048,128,4096)` | 1.41e-3 | 199.1 us | 101.9 us | 301.0 us |
| `(1,56,2048,128,7168)` | 1.41e-3 | 193.5 us | 130.6 us | 324.2 us |
| `(8,56,4096,128,7168)` | 1.41e-3 | 1040.8 us | 407.6 us | 1448.4 us |

融合 PRE 的 q/k/v 在对齐布局和 dtype 后可逐元素一致；全 BF16 链路相对 FP32 参考的约 `1e-3` 误差主要来自 BF16 attention/output GEMM 的归约和量化顺序。

性能优化后必须先复跑正确性，再运行本 benchmark。

## 7. Profiling

通用操作见：

```text
docs/GPU_PROFILING_GUIDE.md
```

对本脚本继续分析时，应分别给 `pre_fused`、`pre_baseline`、`post_fused`、`post_baseline` 添加 NVTX，并使用少量 iteration 的 Nsight Systems trace。正式吞吐仍以无 profiler、重复多轮的 rank-max 时间为准。

## 8. 范围边界

本 benchmark 只测标准 Ulysses forward 的 baseline 与 fused 两臂：

- 不包含 backward；
- 不包含 DDP/FSDP/optimizer；
- 不加载模型 checkpoint；
- 不报告训练显存；
- chain 时间为组件之和，不是真实 tensor-connected end-to-end。

需要模型级训练 benchmark 时，应另建标准 fused autograd 路径，不能把本脚本结果解释为训练吞吐。
