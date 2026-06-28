# 快速安装 FlashAttention-4（FA4）

Ulysses 链路的 attention 统一使用 **FlashAttention-4**。FA4 是 **CuTeDSL 纯 Python 包**
（运行时 JIT 编译内核，无大二进制），所以安装很轻量，只需固定几个 pip 版本即可在相同环境下复现。

## 已验证环境（本仓库基线）

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA B300（Blackwell, `sm_100`） |
| CUDA | 13.0 |
| PyTorch | 2.9.0+cu130 |
| Python | 3.12 |

> FA4 的 sm100 内核针对 Blackwell；Hopper 请用对应的 sm90 内核（同一包内自动选择）。

## 一行安装（推荐，固定版本）

```bash
pip install "flash-attn-4[cu13]==4.0.0b19"
```

`[cu13]` extra 会自动拉取匹配 CUDA 13 的 `nvidia-cutlass-dsl[cu13]`。
如果是 CUDA 12 环境，把 `[cu13]` 换成 `[cu12]`。

## 完全锁定版本（精确复现）

为避免传递依赖漂移，固定如下版本（本基线实测组合）：

```bash
pip install \
  "flash-attn-4==4.0.0b19" \
  "nvidia-cutlass-dsl[cu13]==4.5.2" \
  "quack-kernels==0.5.0" \
  "apache-tvm-ffi==0.1.12" \
  "torch-c-dlpack-ext==0.1.5" \
  "einops==0.8.2"
```

或直接运行脚本：

```bash
bash scripts/install_fa4.sh
```

## 验证安装

```bash
python -c "
import torch, math
from flash_attn.cute import flash_attn_func
q=torch.randn(2,2048,8,128,dtype=torch.bfloat16,device='cuda')  # [B,S,H,D]
k=torch.randn_like(q); v=torch.randn_like(q)
o=flash_attn_func(q,k,v,softmax_scale=1.0/math.sqrt(128),causal=False)
o=o[0] if isinstance(o,tuple) else o
print('FA4 OK, out', tuple(o.shape))
"
```

首次调用会触发 CuTeDSL JIT 编译（数秒），属正常现象，之后按 shape/config 缓存复用。

## FA4 调用约定（本仓库用法）

- **dense（BSHD）**：`flash_attn_func(q, k, v, softmax_scale=, causal=)`，`q/k/v = [B, S, H, D]`，返回 `[B, S, H, D]`。
- **varlen（THD）**：`flash_attn_varlen_func(q, k, v, cu_seqlens_q=, cu_seqlens_k=, max_seqlen_q=, max_seqlen_k=, softmax_scale=, causal=)`，`q/k/v = [total_tokens, H, D]`，返回 `[T, H, D]`。

注意：
- FA4 **没有** `dropout_p` 参数；varlen 的第 4 个位置参数是 `qv`，**必须用关键字**传 `cu_seqlens_*`。
- 若返回 `return_lse=True` 则结果是 tuple，取 `[0]` 为输出张量。

测试/benchmark 中已封装为 `tests/ulysses/fa4_attn.py` 的 `fa4_attn_bhsd` / `fa4_attn_varlen_thd`。
