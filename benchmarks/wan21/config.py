"""Wan2.1 model & SP configuration dataclasses.

These configs decouple model architecture from parallelism strategy and training setup,
so the same model can run under different SP strategies / FSDP modes without code changes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Wan21Config:
    """Wan2.1 model architecture config (framework-agnostic, no SP/FSDP logic)."""
    dim: int = 5120              # hidden dim (14B=5120, 1.3B=2048)
    num_heads: int = 40          # attention heads (14B=40, 1.3B=16)
    head_dim: int = 128          # per-head dim
    ffn_dim: int = 13824         # FFN intermediate (14B=13824, 1.3B=8192)
    num_layers: int = 40         # transformer layers
    qk_norm: bool = True         # QK RMSNorm
    cross_attn_norm: bool = True
    eps: float = 1e-6
    causal: bool = False          # Wan2.1 uses non-causal self-attn

    @property
    def n_qkv(self) -> int:
        return 3 * self.dim

    @property
    def scale(self) -> float:
        import math
        return 1.0 / math.sqrt(self.head_dim)


@dataclass
class SPConfig:
    """Sequence-parallelism config — describes HOW the model is sharded."""
    sp_size: int = 8             # world size / SP degree
    group: object = None         # dist.ProcessGroup (set at runtime)
    layout: str = 'BSHD'         # 'BSHD' or 'THD'
    use_fused_ops: bool = True   # use DeepGEMM fused kernels vs torch matmul+NCCL

    # Wo strategy (only relevant for POST):
    #   'a2a_gemm'    = standard Ulysses: A2A-transpose + Wo GEMM
    #   'gemm_rs'     = variant: Wo row-split + GEMM+RS
    post_strategy: str = 'a2a_gemm'

    @property
    def local_nh(self) -> int:
        return self.num_heads // self.sp_size if hasattr(self, 'num_heads') else 0


@dataclass
class TrainConfig:
    """Training-related config: FSDP2, gradient sync, etc."""
    use_fsdp2: bool = False       # wrap with torch FSDP2 fully_shard
    fsdp_reshard_after_forward: bool = True
    manual_grad_sync: bool = True # manual all-reduce weight grads (fallback if no FSDP2)
    dtype: str = 'bfloat16'
    init_seed: int = 42
    grad_seed: int = 123          # deterministic grad_y for correctness comparison
