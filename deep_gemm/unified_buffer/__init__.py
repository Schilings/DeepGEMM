"""
Unified Symmetric Buffer — one sym buffer for all communication-fused operators.

All operators share the same symmetric memory allocation:
  [0 .. 32)        barrier/signal region (shared across all operators)
  [32 .. 32+MAX)   data region (reinterpreted by each operator's workspace struct)

Operators are executed serially (not concurrently), so the data region can be
overwritten between calls. The barrier region uses a self-resetting +1/-1 protocol,
so NO per-call memset of the sym buffer is needed.

Attention-specific parameters (q_nheads, kv_nheads, head_dim) are OPTIONAL.
They are only needed if you use attention-fused operators:
  - GEMM-A2A-transpose (pre-attn QKV scatter)
  - A2A-transpose-GEMM (post-attn Wo gather)
  - Fused-QKV-Norm-A2A (pre-attn with RMSNorm)

GEMM-RS and AG-GEMM are general-purpose linear+comm operators — they only need
``bs``, ``seq``, and ``hidden`` (the N dimension of the GEMM).  Use them for any
linear layer (MLP, FFN, classifier, etc.), not just attention.

Usage (attention):
  sym = get_unified_symm_buffer(group, bs, seq, hidden,
                                 q_nheads=32, kv_nheads=32, head_dim=128)

Usage (non-attention, e.g. MLP+RS):
  sym = get_unified_symm_buffer(group, bs, seq, hidden)
  # sym.get_gemm_a2a_out_view()  →  raises (no head info)
  # sym.get_fused_qkv_out_view() →  raises (no head info)
  # GEMM-RS / AG-GEMM work via C++ workspace struct as usual
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
      [32 .. 32+RMS)      rms region: [bs*seq, 2] float32 (Fused QKV only)
      [32+RMS .. end)     data region (reinterpreted per operator)

    The data region is sized to accommodate the LARGEST operator's needs.
    Each operator's workspace struct points to the same base pointer but
    interprets the data region differently.

    Operators MUST execute serially (not concurrently) since they share the data region.

    Args:
        group:      Process group for sequence parallelism.
        bs:         Batch size.
        seq:        Full sequence length (must be divisible by group.size()).
        hidden:     Output dimension N of the GEMM (model hidden dim).
        q_nheads:   Total Q heads.  **Optional** — only for attention-fused ops.
        kv_nheads:  Total K/V heads. **Optional** — only for attention-fused ops.
        head_dim:   Head dimension.  **Optional** — only for attention-fused ops.
        out_dtype:  Output dtype (bf16 or fp32).

    When ``q_nheads``/``kv_nheads``/``head_dim`` are ``None``, the buffer still
    supports GEMM-RS and AG-GEMM.  Attention-specific view methods will raise
    ``RuntimeError`` if called without head info.
    """

    def __init__(self,
                 group: 'dist.ProcessGroup',
                 bs: int,
                 seq: int,
                 hidden: int,
                 *,
                 q_nheads: Optional[int] = None,
                 kv_nheads: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 out_dtype: torch.dtype = torch.bfloat16):
        self.group = group
        self.world_size = group.size()
        self.bs = bs
        self.seq = seq
        self.local_seq = seq // self.world_size
        self.hidden = hidden
        self.out_dtype = out_dtype

        # ── Attention-specific dims (optional) ──
        self._has_attn = q_nheads is not None and kv_nheads is not None and head_dim is not None
        if self._has_attn:
            assert q_nheads % self.world_size == 0, 'q_nheads must be divisible by sp_size'
            assert kv_nheads % self.world_size == 0, 'kv_nheads must be divisible by sp_size'
            self.q_nheads = q_nheads
            self.kv_nheads = kv_nheads
            self.head_dim = head_dim
            self.local_q_nheads = q_nheads // self.world_size
            self.local_kv_nheads = kv_nheads // self.world_size
            self.local_q_n = self.local_q_nheads * head_dim
            self.local_kv_n = self.local_kv_nheads * head_dim
            self.local_n = self.local_q_n + 2 * self.local_kv_n  # for Fused QKV
            self.nheads = q_nheads  # for A2A-transpose-GEMM (assumes MHA for post-attn)
            self.local_nheads = self.nheads // self.world_size
            self.local_hidden = self.local_nheads * head_dim  # for A2A-transpose-GEMM
        else:
            self.q_nheads = self.kv_nheads = self.head_dim = None
            self.local_q_nheads = self.local_kv_nheads = None
            self.local_q_n = self.local_kv_n = self.local_n = None
            self.nheads = self.local_nheads = self.local_hidden = None

        assert seq % self.world_size == 0, 'seq must be divisible by sp_size'

        # Compatibility attrs for direct use with bf16_gemm_rs_nt / bf16_ag_gemm_nt
        self.num_max_tokens_per_rank = bs * self.local_seq
        self.use_fp32_comm = False  # bf16 comm (matches GemmRSSymmBuffer default)
        self.num_slots = self.world_size  # matches BF16AGGemmSymmBuffer default

        elem_size = 4 if out_dtype == torch.float32 else 2
        m_per_rank = bs * self.local_seq

        # ── Compute max data region size across applicable operators ──
        candidate_bytes = []

        # GEMM-RS: partial[num_ranks][m_per_rank][hidden] + ready_flags
        gemm_rs_partial = self.world_size * m_per_rank * hidden * elem_size
        gemm_rs_flags = self.world_size * ((m_per_rank + 127) // 128) * ((hidden + 127) // 128) * 4
        candidate_bytes.append(gemm_rs_partial + gemm_rs_flags)

        # AG-GEMM: local_x[num_ranks] + slots[num_slots] + slot_state
        # C++ layout: barrier(32) + num_ranks*M*H + num_slots*M*H + num_slots*4*4
        ag_gemm_local_x = self.world_size * m_per_rank * hidden * elem_size
        ag_gemm_slots = self.num_slots * m_per_rank * hidden * elem_size
        ag_gemm_state = self.num_slots * 4 * 4  # kNumReadyChunksPerSlot=4
        candidate_bytes.append(ag_gemm_local_x + ag_gemm_slots + ag_gemm_state)

        if self._has_attn:
            # GEMM-A2A-transpose (pre-attn): [bs*seq, local_n]
            candidate_bytes.append(bs * seq * self.local_n * elem_size)

            # A2A-transpose-GEMM (post-attn): input + gathered
            a2a_gemm_data = bs * self.local_nheads * seq * head_dim * elem_size
            candidate_bytes.append(2 * a2a_gemm_data)

            # Fused QKV+Norm+A2A: rms + output
            rms_bytes = bs * seq * 2 * 4  # [bs*seq, 2] float32
            fused_qkv_out = bs * seq * self.local_n * elem_size
            candidate_bytes.append(rms_bytes + fused_qkv_out)

        max_data_bytes = max(candidate_bytes)

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
        if self._has_attn:
            self.sum_buffer = torch.zeros(m_per_rank, 2, dtype=torch.float32, device='cuda')
        else:
            self.sum_buffer = None

    @property
    def has_attention(self) -> bool:
        """Whether attention-fused operators are available on this buffer."""
        return self._has_attn

    def _require_attn(self, op_name: str):
        if not self._has_attn:
            raise RuntimeError(
                f'{op_name} requires attention parameters (q_nheads, kv_nheads, head_dim). '
                f'Create the buffer with get_unified_symm_buffer(..., q_nheads=, kv_nheads=, head_dim=) '
                f'to use attention-fused operators. '
                f'GEMM-RS and AG-GEMM do not need these and work without them.')

    def reset_sum_buffer(self):
        """Zero sum_buffer (for Fused QKV+Norm+A2A). Call before each fused QKV kernel."""
        self._require_attn('reset_sum_buffer')
        self.sum_buffer.zero_()

    # ── Views for GEMM-A2A-transpose (pre-attn) ──
    def get_gemm_a2a_out_view(self):
        """[bs, seq, local_n] bf16 — output of GEMM+A2A-transpose."""
        self._require_attn('get_gemm_a2a_out_view')
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, 32 // elem_size, (self.bs, self.seq, self.local_n))

    # ── Views for A2A-transpose-GEMM (post-attn) ──
    @property
    def x(self):
        """[bs, local_nheads, seq, head_dim] bf16 — write attention output here.

        Compatible with ``BF16A2ATransposeGemmSymmBuffer.x`` so that
        ``bf16_a2a_transpose`` / ``bf16_a2a_transpose_gemm_nt`` can accept a
        ``UnifiedSymmBuffer`` directly.

        Raises ``AttributeError`` when attention params are not set, so that
        ``hasattr(sym_buffer, 'x')`` returns ``False`` for non-attention buffers
        (e.g. variant's GEMM-RS/AG-GEMM buffer).
        """
        if not self._has_attn:
            raise AttributeError('x')
        return self.get_a2a_transpose_gemm_views()[0]

    @property
    def gathered(self):
        """[bs*local_seq, hidden] bf16 — A matrix for Wo GEMM (filled by comm).

        Compatible with ``BF16A2ATransposeGemmSymmBuffer.gathered``.

        Raises ``AttributeError`` when attention params are not set.
        """
        if not self._has_attn:
            raise AttributeError('gathered')
        return self.get_a2a_transpose_gemm_views()[1]

    def get_a2a_transpose_gemm_views(self):
        """Returns (x, gathered) views for A2A-transpose+GEMM.
        x: [bs, local_nheads, seq, head_dim] — write attention output here
        gathered: [bs*local_seq, hidden] — A matrix for Wo GEMM"""
        self._require_attn('get_a2a_transpose_gemm_views')
        # barrier region: align((num_m_tiles+1)*4, 128) — matches C++ layout
        barrier_bytes = self._a2a_barrier_bytes()
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

    def _a2a_barrier_bytes(self):
        """Match C++ BF16A2ATransposeGemmWorkspace::get_barrier_bytes()."""
        k_tile_m = 128
        local_seq = self.local_seq
        num_m_tiles = self.bs * ((local_seq + k_tile_m - 1) // k_tile_m)
        n = (num_m_tiles + 1) * 4
        return (n + 127) // 128 * 128

    def reset_a2a_barriers(self):
        """Zero per-tile barrier region for A2A-transpose-GEMM fused path."""
        self._require_attn('reset_a2a_barriers')
        barrier_bytes = self._a2a_barrier_bytes()
        self.buffer[:barrier_bytes].zero_()

    # Alias for compatibility with BF16A2ATransposeGemmSymmBuffer.reset_barriers
    def reset_barriers(self):
        """Alias for reset_a2a_barriers (BF16A2ATransposeGemmSymmBuffer compat)."""
        self.reset_a2a_barriers()

    # ── Views for GEMM-RS (general linear, no attention needed) ──
    # GEMM-RS uses its own C++ workspace struct with base=buffer.data_ptr()
    # The struct reads from offset 32, so it works as long as buffer is big enough.
    # bf16_gemm_rs_nt() directly accepts this object as sym_buffer.

    # ── Views for AG-GEMM (general linear, no attention needed) ──
    # AG-GEMM also uses its own C++ workspace struct. The C++ kernel re-creates
    # the layout from the raw buffer, so bf16_ag_gemm_nt() works directly.
    # However, the caller needs to copy input data into the buffer first.
    @property
    def ag_x(self):
        """View into the AG-GEMM local_x region for copying input data.
        [m_per_rank, hidden] bf16 — write grad_y here before calling bf16_ag_gemm_nt."""
        return torch.empty(
            (self.num_max_tokens_per_rank, self.hidden),
            dtype=torch.bfloat16, device=self.buffer.device
        ).set_(self.buffer, 32 // 2,
               (self.num_max_tokens_per_rank, self.hidden))

    @property
    def ag_slots_x(self):
        """View into the AG-GEMM slots_x region (populated by the kernel during gather).
        [num_slots, m_per_rank, hidden] bf16 — read gathered data from here."""
        slots_offset_bytes = 32 + self.world_size * self.num_max_tokens_per_rank * self.hidden * 2
        return torch.empty(
            (self.num_slots, self.num_max_tokens_per_rank, self.hidden),
            dtype=torch.bfloat16, device=self.buffer.device
        ).set_(self.buffer, slots_offset_bytes // 2,
               (self.num_slots, self.num_max_tokens_per_rank, self.hidden))

    # ── Views for Fused QKV+Norm+A2A ──
    def get_fused_qkv_out_view(self):
        """[bs, seq, local_n] bf16 — un-normed QKV scattered by head."""
        self._require_attn('get_fused_qkv_out_view')
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        rms_bytes = self.bs * self.seq * 2 * 4
        out_offset = 32 + rms_bytes
        return torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, out_offset // elem_size, (self.bs, self.seq, self.local_n))

    def get_fused_qkv_rms_view(self):
        """[bs, seq, 2] float32 — Q rms at [:,:,0], K rms at [:,:,1]."""
        self._require_attn('get_fused_qkv_rms_view')
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
                             hidden: int,
                             *,
                             q_nheads: Optional[int] = None,
                             kv_nheads: Optional[int] = None,
                             head_dim: Optional[int] = None,
                             out_dtype: torch.dtype = torch.bfloat16
                             ) -> UnifiedSymmBuffer:
    """Create a unified symmetric buffer shared by all communication-fused operators.

    Args:
        group:      Process group for sequence parallelism.
        bs:         Batch size.
        seq:        Full sequence length (must be divisible by group.size()).
        hidden:     Output dimension N of the GEMM (model hidden dim).
        q_nheads:   Total Q heads.  Optional — only for attention-fused ops
                    (GEMM-A2A, A2A-GEMM, Fused-QKV-Norm-A2A).
        kv_nheads:  Total K/V heads. Optional — same as above.
        head_dim:   Head dimension.  Optional — same as above.
        out_dtype:  Output dtype (bf16 or fp32).

    Without attention parameters, the buffer supports GEMM-RS and AG-GEMM
    (general-purpose linear + reduce-scatter / all-gather fused operators).

    With attention parameters, it additionally supports GEMM-A2A-transpose,
    A2A-transpose-GEMM, and Fused-QKV-Norm-A2A.
    """
    return UnifiedSymmBuffer(group, bs, seq, hidden,
                             q_nheads=q_nheads, kv_nheads=kv_nheads,
                             head_dim=head_dim, out_dtype=out_dtype)
