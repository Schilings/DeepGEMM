"""Dynamic Ulysses SP — runtime SP-size-aware training framework.

Components:
  - DynamicSPGroupManager: pre-creates power-of-2 SP/DP NCCL groups
  - BalancedDataLoader: FLOPs-aware sequence packing
  - DynamicUlyssesLayer: runtime SP-size-aware attention
  - DynamicGradientSync: bucketed gradient sync across all ranks

See DESIGN.md for the full design document.
"""

from .sp_group_manager import DynamicSPGroupManager, SPGroupInfo
from .balanced_loader import BalancedDataLoader, Microbatch
from .dynamic_ulysses import DynamicUlyssesLayer, SPConfig
from .grad_sync import DynamicGradientSync

__all__ = [
    'DynamicSPGroupManager', 'SPGroupInfo',
    'BalancedDataLoader', 'Microbatch',
    'DynamicUlyssesLayer', 'SPConfig',
    'DynamicGradientSync',
]
