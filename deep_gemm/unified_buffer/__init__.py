"""
Unified Symmetric Buffer — one sym buffer for all communication-fused operators.

All operators share the same symmetric memory allocation:
  [0 .. 32)        barrier/signal region (shared across all operators)
  [32 .. 32+MAX)   data region (reinterpreted by each operator's workspace struct)

Operators are executed serially (not concurrently), so the data region can be
overwritten between calls. The barrier region uses a self-resetting +1/-1 protocol,
so NO per-call memset of the sym buffer is needed.

Usage:
  # Create once at model init
  sym = get_unified_symm_buffer(group, bs, seq, q_nheads, kv_nheads, head_dim, hidden)

  # Reuse across all operators
  deep_gemm.bf16_gemm_a2a_transpose_nt(x, Wqkv, sym, local_seq, ...)
  deep_gemm.bf16_a2a_transpose_gemm_nt(d, Wo, sym, ...)
  deep_gemm.bf16_gemm_rs_nt(d, Wo, sym, ...)
  deep_gemm.bf16_ag_gemm_nt(d, Wo, sym, ...)
  out, rms = deep_gemm.bf16_fused_qkv_norm_a2a_nt(x, Wqkv, sym, local_seq, ...)
"""

import torch
from typing import Optional

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load symm_mem: {exception}')


class UnifiedSymmBuffer:
    """One symmetric buffer for all communication-fused operators.

    Layout:
      [0 .. 32)            barrier/signal region (kNumBarrierSignalBytes=32)
      [32 .. 32+RMS)       rms region: [bs*seq, 2] float32 (for Fused QKV+Norm+A2A)
      [32+RMS .. end)      data region (reinterpreted per operator)

    The data region is sized to accommodate the LARGEST operator's needs.
    Each operator's workspace struct points to the same base pointer but
    interprets the data region differently.

    Operators MUST execute serially (not concurrently) since they share the data region.
    """

    def __init__(self, group: 'dist.ProcessGroup',
                 bs: int,
                 seq: int,
                 q_nheads: int,
                 kv_nheads: int,
                 head_dim: int,
                 hidden: int,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.group = group
        self.world_size = group.size()
        self.bs = bs
        self.seq = seq
        self.local_seq = seq // self.world_size
        self.q_nheads = q_nheads
        self.kv_nheads = kv_nheads
        self.head_dim = head_dim
        self.hidden = hidden  # nheads * head_dim (for GEMM-RS/AG-GEMM)
        self.out_dtype = out_dtype

        # Derived dimensions
        self.local_q_nheads = q_nheads // self.world_size
        self.local_kv_nheads = kv_nheads // self.world_size
        self.local_q_n = self.local_q_nheads * head_dim
        self.local_kv_n = self.local_kv_nheads * head_dim
        self.local_n = self.local_q_n + 2 * self.local_kv_n  # for Fused QKV
        self.nheads = q_nheads  # for A2A-transpose-GEMM (assumes MHA for post-attn)
        self.local_nheads = self.nheads // self.world_size
        self.local_hidden = self.local_nheads * head_dim  # for A2A-transpose-GEMM

        elem_size = 4 if out_dtype == torch.float32 else 2

        # ── Compute max data region size across all operators ──

        # 1. GEMM-A2A-transpose (pre-attn): [bs*seq, local_n] bf16
        gemm_a2a_bytes = bs * seq * self.local_n * elem_size

        # 2. A2A-transpose-GEMM (post-attn): input + gathered
        #    input = [bs, local_nheads, seq, head_dim], gathered = [bs*local_seq, hidden]
        #    Both = bs * local_nheads * seq * head_dim * 2 (bf16)
        a2a_gemm_data_bytes = bs * self.local_nheads * seq * head_dim * elem_size
        a2a_gemm_bytes = 2 * a2a_gemm_data_bytes  # input + gathered

        # 3. GEMM-RS (post-attn variant): partial[num_ranks][m_per_rank][hidden] + ready_flags
        m_per_rank = bs * self.local_seq
        gemm_rs_partial = self.world_size * m_per_rank * hidden * elem_size
        gemm_rs_flags = self.world_size * ((m_per_rank + 127) // 128) * ((hidden + 127) // 128) * 4
        gemm_rs_bytes = gemm_rs_partial + gemm_rs_flags

        # 4. AG-GEMM (bwd): local_x + slots[num_ranks] + slot_state
        ag_gemm_local_x = m_per_rank * hidden * elem_size
        ag_gemm_slots = self.world_size * m_per_rank * hidden * elem_size
        ag_gemm_state = self.world_size * 4 * 4  # kNumReadyChunksPerSlot=4
        ag_gemm_bytes = ag_gemm_local_x + ag_gemm_slots + ag_gemm_state

        # 5. Fused QKV+Norm+A2A: rms + output
        rms_bytes = bs * seq * 2 * 4  # [bs*seq, 2] float32
        fused_qkv_out = bs * seq * self.local_n * elem_size
        fused_qkv_bytes = rms_bytes + fused_qkv_out

        # Max data region (after 32B barrier)
        max_data_bytes = max(gemm_a2a_bytes, a2a_gemm_bytes, gemm_rs_bytes,
                             ag_gemm_bytes, fused_qkv_bytes)

        # Total: barrier(32) + max_data
        num_bytes = 32 + max_data_bytes
        num_bytes = (num_bytes + 127) // 128 * 128  # align to 128

        self.num_bytes = num_bytes
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # Local (non-symmetric) sum_buffer for Fused QKV (x² partial sums)
        self.sum_buffer = torch.zeros(m_per_rank, 2, dtype=torch.float32, device='cuda')

    def reset_sum_buffer(self):
        """Zero sum_buffer (for Fused QKV+Norm+A2A). Call before each fused QKV kernel."""
        self.sum_buffer.zero_()

    # ── Views for GEMM-A2A-transpose (pre-attn) ──
    def get_gemm_a2a_out_view(self):
        """[bs, seq, local_n] bf16 — output of GEMM+A2A-transpose."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, 32 // elem_size, (self.bs, self.seq, self.local_n))

    # ── Views for A2A-transpose-GEMM (post-attn) ──
    def get_a2a_transpose_gemm_views(self):
        """Returns (x, gathered) views for A2A-transpose+GEMM.
        x: [bs, local_nheads, seq, head_dim] — write attention output here
        gathered: [bs*local_seq, hidden] — A matrix for Wo GEMM"""
        # barrier region for this op is variable; use 128B aligned start
        barrier_bytes = 128  # fixed: enough for per-tile barriers + signal
        data_bytes = self.bs * self.local_nheads * self.seq * self.head_dim * 2  # bf16
        x = torch.empty(
            (self.bs, self.local_nheads, self.seq, self.head_dim),
            dtype=torch.bfloat16, device=self.buffer.device
        ).set_(self.buffer, barrier_bytes // 2,
               (self.bs, self.local_nheads, self.seq, self.head_dim))
        gathered = torch.empty(
            (self.bs * self.local_seq, self.hidden),
            dtype=torch.bfloat16, device=self.buffer.device
        ).set_(self.buffer, (barrier_bytes + data_bytes) // 2,
               (self.bs * self.local_seq, self.hidden))
        return x, gathered

    def reset_a2a_barriers(self):
        """Zero per-tile barrier region for A2A-transpose-GEMM fused path."""
        barrier_bytes = 128
        self.buffer[:barrier_bytes].zero_()

    # ── Views for GEMM-RS (post-attn variant) ──
    # GEMM-RS uses its own C++ workspace struct with base=buffer.data_ptr()
    # The struct reads from offset 32, so it works as long as buffer is big enough.

    # ── Views for AG-GEMM (bwd) ──
    # AG-GEMM also uses its own C++ workspace struct with base=buffer.data_ptr()

    # ── Views for Fused QKV+Norm+A2A ──
    def get_fused_qkv_out_view(self):
        """[bs, seq, local_n] bf16 — un-normed QKV scattered by head."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        rms_bytes = self.bs * self.seq * 2 * 4
        out_offset = 32 + rms_bytes
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, out_offset // elem_size, (self.bs, self.seq, self.local_n))

    def get_fused_qkv_rms_view(self):
        """[bs, seq, 2] float32 — Q rms at [:,:,0], K rms at [:,:,1]."""
        return torch.empty(
            (self.bs, self.seq, 2),
            dtype=torch.float32, device=self.buffer.device
        ).set_(self.buffer, 32 // 4, (self.bs, self.seq, 2))

    @property
    def buffer_ptrs(self):
        """Symmetric buffer pointers for C++ kernel calls."""
        return self.handle.buffer_ptrs

    @property
    def rank(self):
        return self.group.rank()

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.sum_buffer = None
        self.group = None


def get_unified_symm_buffer(group: 'dist.ProcessGroup',
                             bs: int,
                             seq: int,
                             q_nheads: int,
                             kv_nheads: int,
                             head_dim: int,
                             hidden: int,
                             out_dtype: torch.dtype = torch.bfloat16
                             ) -> UnifiedSymmBuffer:
    """Create a unified symmetric buffer shared by all communication-fused operators.

    Args:
        group: Process group for SP
        bs: Batch size
        seq: Full sequence length (must be divisible by group.size())
        q_nheads: Total Q heads
        kv_nheads: Total K/V heads
        head_dim: Head dimension
        hidden: Model hidden dim (nheads * head_dim, for GEMM-RS/AG-GEMM)
        out_dtype: Output dtype (bf16 or fp32)
    """
    assert q_nheads % group.size() == 0, 'q_nheads must be divisible by sp_size'
    assert kv_nheads % group.size() == 0, 'kv_nheads must be divisible by sp_size'
    assert seq % group.size() == 0, 'seq must be divisible by sp_size'
    return UnifiedSymmBuffer(group, bs, seq, q_nheads, kv_nheads, head_dim, hidden, out_dtype)
