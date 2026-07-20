"""Symmetric buffer pool — pre-allocated UnifiedSymmBuffer per SP size."""
import torch
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BufferSpec:
    sp_size: int
    seq: int
    hidden: int
    q_nheads: int
    kv_nheads: int
    head_dim: int

class SymBufferPool:
    """Pool of UnifiedSymmBuffer instances, one per SP size."""
    def __init__(self, group, world_size, hidden, num_heads, head_dim, out_dtype=torch.bfloat16):
        self.group = group
        self.world_size = world_size
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.out_dtype = out_dtype
        self.sp_sizes = [1 << i for i in range(world_size.bit_length()) if num_heads // (1 << i) >= 1]
        self._buffers: Dict[int, object] = {}

    def get_buffer(self, sp_size, seq):
        if sp_size == 1:
            return None
        assert sp_size in self.sp_sizes, f"Invalid sp_size: {sp_size}"
        assert seq % sp_size == 0, f"seq={seq} not divisible by sp_size={sp_size}"
        existing = self._buffers.get(sp_size)
        if existing is not None and getattr(existing, 'seq', 0) == seq:
            return existing
        if existing is not None:
            existing.destroy()
        from deep_gemm import get_unified_symm_buffer
        buf = get_unified_symm_buffer(
            self.group, bs=1, seq=seq, hidden=self.hidden,
            q_nheads=self.num_heads, kv_nheads=self.num_heads,
            head_dim=self.head_dim, out_dtype=self.out_dtype)
        self._buffers[sp_size] = buf
        return buf

    def destroy_all(self):
        for buf in self._buffers.values():
            if buf is not None:
                buf.destroy()
        self._buffers.clear()

    def total_memory_mb(self):
        total = sum(getattr(b, 'num_bytes', 0) for b in self._buffers.values() if b)
        return total / 1024 / 1024
