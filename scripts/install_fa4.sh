#!/usr/bin/env bash
# Quick, version-pinned FlashAttention-4 install for the Ulysses attention chain.
# Verified baseline: NVIDIA B300 (sm_100), CUDA 13.0, PyTorch 2.9.0+cu130, Python 3.12.
# For CUDA 12 environments, replace [cu13] with [cu12].
# See docs/INSTALL_FA4.md for details.
set -euo pipefail

echo "[install_fa4] installing version-pinned FlashAttention-4 ..."
pip install \
  "flash-attn-4==4.0.0b19" \
  "nvidia-cutlass-dsl[cu13]==4.5.2" \
  "quack-kernels==0.5.0" \
  "apache-tvm-ffi==0.1.12" \
  "torch-c-dlpack-ext==0.1.5" \
  "einops==0.8.2"

echo "[install_fa4] verifying ..."
python - <<'PY'
import torch, math
from flash_attn.cute import flash_attn_func
q = torch.randn(2, 2048, 8, 128, dtype=torch.bfloat16, device='cuda')  # [B,S,H,D]
k = torch.randn_like(q); v = torch.randn_like(q)
o = flash_attn_func(q, k, v, softmax_scale=1.0 / math.sqrt(128), causal=False)
o = o[0] if isinstance(o, tuple) else o
assert tuple(o.shape) == (2, 2048, 8, 128), o.shape
print("[install_fa4] FA4 OK, out", tuple(o.shape))
PY
echo "[install_fa4] done."
