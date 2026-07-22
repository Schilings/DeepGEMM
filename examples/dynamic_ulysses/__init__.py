"""Dynamic Ulysses SP — runtime SP-size-aware training framework.

Components:
  - DynamicSPGroupManager: pre-creates power-of-2 SP/DP NCCL groups
  - BalancedDataLoader: FLOPs-aware sequence packing
  - DynamicUlyssesLayer: runtime SP-size-aware attention
  - DynamicGradientSync: bucketed gradient sync across all ranks
  - DynamicTrainer: complete training loop with microbatch scheduling
  - SymBufferPool: pre-allocated UnifiedSymmBuffer per SP size
  - OverlapGradientSync: overlap gradient AllReduce with backward

See DESIGN.md for the full design document.
"""

from .sp_group_manager import DynamicSPGroupManager, SPGroupInfo
from .balanced_loader import BalancedDataLoader, Microbatch, PackedMicrobatch
from .dynamic_ulysses import DynamicUlyssesLayer, SPConfig
from .grad_sync import DynamicGradientSync
from .dynamic_trainer import DynamicTrainer
from .buffer_pool import SymBufferPool
from .overlap_grad_sync import OverlapGradientSync

__all__ = [
    'DynamicSPGroupManager', 'SPGroupInfo',
    'BalancedDataLoader', 'Microbatch', 'PackedMicrobatch',
    'DynamicUlyssesLayer', 'SPConfig',
    'DynamicGradientSync',
    'DynamicTrainer',
    'SymBufferPool',
    'OverlapGradientSync',
]
