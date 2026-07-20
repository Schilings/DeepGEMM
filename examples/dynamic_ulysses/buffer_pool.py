"""Buffer pool for dynamic SP — pre-allocates UnifiedSymmBuffer per SP size.

Each SP size needs a different buffer layout (different local_nheads, local_seq).
This pool pre-allocates buffers for all configured SP sizes and returns the
correct one at runtime, avoiding allocation during forward.

Memory: only the largest SP size's buffer is active at a time (others are
allocated but zeroed). Total memory = max(buffer_size_per_sp).
"""

import torch
import torch.distributed as dist
from typing import Dict, Optional, List
from dataclasses import dataclass


class SymBufferPool:
    """Pre-allocated UnifiedSymmBuffer pool for dynamic SP.

    Creates one buffer per SP size. Only the buffer matching the current
    SP size is "active"; others are idle (memory still allocated but not used).

    Args:
        group_manager: DynamicSPGroupManager with pre-created groups.
        bs:            Batch size.
        hidden:        Hidden dimension.
        num_heads:     Total attention heads.
        head_dim:      Head dimension.
        sp_sizes:      SP sizes to pre-allocate for (default: all valid).
        max_seq:       Maximum sequence length (for buffer sizing).
    """

    def __init__(self,
                 group_manager,
                 bs: int,
                 hidden: int,
                 num_heads: int,
                 head_dim: int,
                 sp_sizes: Optional[List[int]] = None,
                 max_seq: int = 32768):
        self.gm = group_manager
        self.bs = bs
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq = max_seq

        if sp_sizes is None:
            sp_sizes = group_manager.get_valid_sp_sizes()
        self.sp_sizes = [s for s in sp_sizes if s > 1]  # SP=1 needs no buffer

        # Lazy allocation: only allocate when first used
        self._buffers: Dict[int, object] = {}
        self._current_sp: int = 0

    def get_buffer(self, sp_size: int) -> Optional[object]:
        """Get the UnifiedSymmBuffer for the given SP size.

        Returns None for SP=1 (no buffer needed).
        Lazily allocates on first access.
        """
        if sp_size == 1:
            return None

        if sp_size not in self._buffers:
            self._allocate(sp_size)

        self._current_sp = sp_size
        return self._buffers[sp_size]

    def _allocate(self, sp_size: int):
        """Allocate a UnifiedSymmBuffer for the given SP size."""
        from deep_gemm import get_unified_symm_buffer

        info = self.gm.get_groups(sp_size)
        # Use max_seq for buffer sizing (actual seq may be shorter)
        seq = self.max_seq

        buf = get_unified_symm_buffer(
            info.sp_group, self.bs, seq, self.hidden,
            q_nheads=self.num_heads, kv_nheads=self.num_heads,
            head_dim=self.head_dim, out_dtype=torch.bfloat16,
        )
        self._buffers[sp_size] = buf

    def destroy(self):
        """Destroy all allocated buffers."""
        for buf in self._buffers.values():
            buf.destroy()
        self._buffers.clear()

    def total_memory_mb(self) -> float:
        """Total allocated memory in MB."""
        total = 0
        for sp, buf in self._buffers.items():
            if hasattr(buf, 'num_bytes'):
                total += buf.num_bytes
        return total / 1024 / 1024
