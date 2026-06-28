"""FlashAttention-4 (flash-attn-4) helpers for the Ulysses tests/benchmarks.

The whole Ulysses attention chain uses FlashAttention-4. Install it with the pinned versions in
docs/INSTALL_FA4.md (e.g. `pip install "flash-attn-4[cu13]==4.0.0b19"`).

FA4 native layouts:
  * dense  : flash_attn_func(q, k, v)              q/k/v = [B, S, H, D]    -> [B, S, H, D]
  * varlen : flash_attn_varlen_func(q, k, v,       q/k/v = [total_T, H, D] -> [T, H, D]
             cu_seqlens_q=, cu_seqlens_k=, max_seqlen_q=, max_seqlen_k=)

We expose two drop-in helpers that keep the existing call-site layouts:
  * fa4_attn_bhsd(q, k, v, hd):              q/k/v = [B, H, S, D] -> [B, H, S, D]  (drop-in for SDPA)
  * fa4_attn_varlen_thd(q, k, v, cu, max_seq, hd): q/k/v = [T, H, D] -> [T, H, D]
"""

import math
import torch

try:
    from flash_attn.cute import flash_attn_func as _fa4_func, flash_attn_varlen_func as _fa4_varlen
    HAVE_FA4 = True
    _IMPORT_ERR = None
except Exception as _e:                                   # pragma: no cover
    HAVE_FA4 = False
    _IMPORT_ERR = _e


def _require():
    if not HAVE_FA4:
        raise RuntimeError(
            f"FlashAttention-4 (flash-attn-4) is required but not importable: {_IMPORT_ERR}. "
            "Install it via docs/INSTALL_FA4.md  (pip install 'flash-attn-4[cu13]==4.0.0b19').")


def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def fa4_attn_bhsd(q, k, v, hd):
    """Drop-in for SDPA. q/k/v: [B, H, S, D] -> [B, H, S, D].

    FA4 wants BSHD ([B, S, H, D]), so we transpose H<->S around the call.
    """
    _require()
    scale = 1.0 / math.sqrt(hd)
    qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))   # [B, S, H, D]
    o = _unwrap(_fa4_func(qb, kb, vb, softmax_scale=scale, causal=False))
    return o.transpose(1, 2).contiguous()                              # [B, H, S, D]


def fa4_attn_varlen_thd(q, k, v, cu, max_seq, hd):
    """Packed varlen THD attention. q/k/v: [T, H, D] -> [T, H, D]."""
    _require()
    scale = 1.0 / math.sqrt(hd)
    return _unwrap(_fa4_varlen(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                               max_seqlen_q=max_seq, max_seqlen_k=max_seq,
                               softmax_scale=scale, causal=False))
