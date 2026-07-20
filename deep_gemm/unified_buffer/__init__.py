"""
Unified Symmetric Buffer — one sym buffer for all communication-fused operators.

Design follows the MegaMoE pattern: the constructor calls a C++ ``slice_buffers``
to compute the total size and create all tensor views in one shot.  No lazy
properties, no aliases, no ``hasattr`` checks — every view is a plain attribute
(set to ``None`` when the corresponding operator family is not configured).

Operators share the same physical memory and must execute serially.
"""

import torch
from typing import Optional, NamedTuple

try:
    import torch.distributed._symmetric_memory as symm_mem
    import torch.distributed as dist
except Exception as exception:
    print(f'Failed to load symm_mem: {exception}')


# ---------------------------------------------------------------------------
# View containers (like MegaMoE's slice_input_buffers return tuple)
# ---------------------------------------------------------------------------

class GemmRSViews(NamedTuple):
    """Views for ``bf16_gemm_rs_nt`` (GEMM + ReduceScatter)."""
    pass  # GEMM-RS uses C++ workspace struct directly, no Python views needed


class AGGemmViews(NamedTuple):
    """Views for ``bf16_ag_gemm_nt`` / ``bf16_ag_gemm_nt_with_input``."""
    local_x: torch.Tensor    # [m_per_rank, hidden] — write grad_y here
    slots_x: torch.Tensor    # [num_slots, m_per_rank, hidden] — gathered data


class A2ATransposeGemmViews(NamedTuple):
    """Views for ``bf16_a2a_transpose_gemm_nt`` (A2A + Wo GEMM)."""
    x: torch.Tensor          # [bs, local_nheads, seq, head_dim] — write attn output
    gathered: torch.Tensor   # [bs*local_seq, hidden] — A matrix for Wo GEMM


class FusedQKVNormA2AViews(NamedTuple):
    """Views for ``bf16_fused_qkv_norm_a2a_nt`` (GEMM + Norm + A2A)."""
    out: torch.Tensor        # [bs, seq, local_n] — scattered QKV
    rms: torch.Tensor        # [bs, seq, 2] float32 — Q/K rms values
    sum_buffer: torch.Tensor # [m_per_rank, 2] float32 — x² partial sums (local)


class UnifiedSymmBuffer:
    """One symmetric buffer for all communication-fused operators.

    All tensor views are created in the constructor (MegaMoE-style).
    Unused view groups are ``None``.

    Args:
        group:      Process group.
        bs:         Batch size.
        seq:        Full sequence length.
        hidden:     Hidden dimension (N of GEMM).
        q_nheads:   Total Q heads (optional, attention ops only).
        kv_nheads:  Total K/V heads (optional, attention ops only).
        head_dim:   Head dimension (optional, attention ops only).
        out_dtype:  Output dtype.
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
        self.rank = group.rank()
        self.bs = bs
        self.seq = seq
        self.local_seq = seq // self.world_size
        self.hidden = hidden
        self.out_dtype = out_dtype
        self.num_max_tokens_per_rank = bs * self.local_seq

        assert seq % self.world_size == 0

        # Attention dims
        self._has_attn = q_nheads is not None and kv_nheads is not None and head_dim is not None
        if self._has_attn:
            assert q_nheads % self.world_size == 0
            assert kv_nheads % self.world_size == 0
            self.q_nheads = q_nheads
            self.kv_nheads = kv_nheads
            self.head_dim = head_dim
            self.local_q_nheads = q_nheads // self.world_size
            self.local_kv_nheads = kv_nheads // self.world_size
            self.local_q_n = self.local_q_nheads * head_dim
            self.local_kv_n = self.local_kv_nheads * head_dim
            self.local_n = self.local_q_n + 2 * self.local_kv_n
            self.nheads = q_nheads
            self.local_nheads = q_nheads // self.world_size
            self.local_hidden = self.local_nheads * head_dim
        else:
            self.q_nheads = self.kv_nheads = self.head_dim = None
            self.local_q_nheads = self.local_kv_nheads = None
            self.local_q_n = self.local_kv_n = self.local_n = None
            self.nheads = self.local_nheads = self.local_hidden = None

        # Compatibility attrs for C++ kernel calls
        self.num_slots = self.world_size
        self.use_fp32_comm = False

        # ── Compute buffer size (max across all configured operators) ──
        elem_size = 4 if out_dtype == torch.float32 else 2
        m_per_rank = self.num_max_tokens_per_rank
        sp = self.world_size

        candidates = []

        # GEMM-RS: partial[sp][m][hidden] + flags
        candidates.append(
            sp * m_per_rank * hidden * elem_size
            + sp * ((m_per_rank + 127) // 128) * ((hidden + 127) // 128) * 4
        )

        # AG-GEMM: local_x[sp] + slots[num_slots] + state
        candidates.append(
            sp * m_per_rank * hidden * elem_size
            + self.num_slots * m_per_rank * hidden * elem_size
            + self.num_slots * 4 * 4
        )

        if self._has_attn:
            # A2A-transpose-GEMM: x + gathered
            a2a_data = bs * self.local_nheads * seq * head_dim * elem_size
            candidates.append(2 * a2a_data)

            # Fused-QKV-Norm-A2A: rms + out
            rms_bytes = bs * seq * 2 * 4
            fused_out = bs * seq * self.local_n * elem_size
            candidates.append(rms_bytes + fused_out)

        max_data = max(candidates)
        num_bytes = (32 + max_data + 127) // 128 * 128

        # ── Allocate symmetric memory ──
        self.num_bytes = num_bytes
        self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device='cuda')
        self.handle = symm_mem.rendezvous(self.buffer, group=group)
        self.buffer.zero_()
        self.group.barrier()
        torch.cuda.synchronize()

        # ── Create all views upfront (MegaMoE-style) ──
        self.gemm_rs = GemmRSViews() if not self._has_attn else GemmRSViews()
        self.ag_gemm = self._make_ag_gemm_views()
        self.a2a_gemm = self._make_a2a_gemm_views() if self._has_attn else None
        self.fused_qkv = self._make_fused_qkv_views() if self._has_attn else None

    # ── View creation (called once in __init__) ──

    def _make_ag_gemm_views(self) -> AGGemmViews:
        """AG-GEMM views: local_x at offset 32, slots_x after local_x."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        sp = self.world_size
        m = self.num_max_tokens_per_rank
        h = self.hidden

        local_x = torch.empty(
            (m, h), dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, 32 // elem_size, (m, h))

        slots_offset = 32 + sp * m * h * elem_size
        slots_x = torch.empty(
            (self.num_slots, m, h), dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, slots_offset // elem_size, (self.num_slots, m, h))

        return AGGemmViews(local_x=local_x, slots_x=slots_x)

    def _make_a2a_gemm_views(self) -> A2ATransposeGemmViews:
        """A2A-transpose-GEMM views: x at barrier offset, gathered after x."""
        barrier_bytes = self._a2a_barrier_bytes()
        data_bytes = self.bs * self.local_nheads * self.seq * self.head_dim * 2

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

        return A2ATransposeGemmViews(x=x, gathered=gathered)

    def _make_fused_qkv_views(self) -> FusedQKVNormA2AViews:
        """Fused-QKV-Norm-A2A views: rms at offset 32, out after rms."""
        elem_size = 4 if self.out_dtype == torch.float32 else 2
        rms_bytes = self.bs * self.seq * 2 * 4

        rms = torch.empty(
            (self.bs, self.seq, 2),
            dtype=torch.float32, device=self.buffer.device
        ).set_(self.buffer, 32 // 4, (self.bs, self.seq, 2))

        out = torch.empty(
            (self.bs, self.seq, self.local_n),
            dtype=self.out_dtype, device=self.buffer.device
        ).set_(self.buffer, (32 + rms_bytes) // elem_size,
               (self.bs, self.seq, self.local_n))

        sum_buffer = torch.zeros(
            self.num_max_tokens_per_rank, 2, dtype=torch.float32, device='cuda')

        return FusedQKVNormA2AViews(out=out, rms=rms, sum_buffer=sum_buffer)

    def _a2a_barrier_bytes(self) -> int:
        """Match C++ BF16A2ATransposeGemmWorkspace::get_barrier_bytes()."""
        k_tile_m = 128
        num_m_tiles = self.bs * ((self.local_seq + k_tile_m - 1) // k_tile_m)
        return ((num_m_tiles + 1) * 4 + 127) // 128 * 128

    # ── Convenience properties (thin wrappers, no logic) ──

    @property
    def buffer_ptrs(self):
        return self.handle.buffer_ptrs

    @property
    def has_attention(self) -> bool:
        return self._has_attn

    # ── Lifecycle ──

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.gemm_rs = None
        self.ag_gemm = None
        self.a2a_gemm = None
        self.fused_qkv = None


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
    return UnifiedSymmBuffer(group, bs, seq, hidden,
                             q_nheads=q_nheads, kv_nheads=kv_nheads,
                             head_dim=head_dim, out_dtype=out_dtype)
