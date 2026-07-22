"""Balanced data loader — FLOPs-aware sequence packing for dynamic SP.

Inspired by ByteScale's Balance Scheduler and Megatron's HybridCPDataLoaderWrapper.

Core idea: sort sequences by length, group them into buckets with similar
attention FLOPs (O(S²)), and assign each bucket to an SP group of appropriate
size so that all SP groups finish at approximately the same wall-clock time.

For 8 GPUs with SP sizes {1, 2, 4, 8}:

  Long sequences  (S > 16K):  SP=8 (1 copy, all 8 GPUs collaborate)
  Medium sequences (4K-16K):  SP=4 (2 copies, 4 GPUs each)
  Short sequences  (1K-4K):   SP=2 (4 copies, 2 GPUs each)
  Tiny sequences   (S < 1K):  SP=1 (8 copies, no SP comm)

The scheduler outputs a list of microbatches, each tagged with its SP size.
Microbatches are ordered so that larger SP groups run first (they take longer).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class Microbatch:
    """A single microbatch with its assigned SP size.

    Attributes:
        sp_size:     Number of GPUs in the SP group.
        seq_len:     Total sequence length (before SP sharding).
        local_seq:   Per-rank sequence length = seq_len / sp_size.
        dp_copy:     Which DP copy this microbatch belongs to (0-indexed).
        tokens:      Token count = seq_len (for gradient scaling).
    """
    sp_size: int
    seq_len: int
    local_seq: int
    dp_copy: int
    tokens: int

    def __repr__(self):
        return (f"Microbatch(sp={self.sp_size}, seq={self.seq_len}, "
                f"local_seq={self.local_seq}, dp_copy={self.dp_copy})")


@dataclass
class PackedMicrobatch:
    """A packed microbatch containing multiple variable-length sequences.

    All sequences share the same SP size and are packed into a single
    contiguous tensor for one forward pass (THD layout, no padding).

    Attributes:
        sp_size:      SP group size.
        seq_lens:     List of sequence lengths packed together.
        total_tokens: sum(seq_lens).
        local_tokens: total_tokens // sp_size (per-rank token count).
        cu_seqlens:   Cumulative sequence lengths [0, s1, s1+s2, ...].
        dp_copy:      Which DP copy this packed batch belongs to.
    """
    sp_size: int
    seq_lens: List[int]
    total_tokens: int
    local_tokens: int
    cu_seqlens: List[int]
    dp_copy: int

    def __repr__(self):
        return (f"PackedMicrobatch(sp={self.sp_size}, seqs={self.seq_lens}, "
                f"total={self.total_tokens}, dp={self.dp_copy})")


class BalancedDataLoader:
    """FLOPs-aware data loader for dynamic SP.

    Args:
        world_size:    Total GPU count.
        sp_sizes:      Valid SP sizes (must be powers of 2 dividing world_size).
        seq_align:     Sequence length alignment (for FA4 tile, typically 128).
        max_seq_per_sp: Max sequences per SP group (to control memory).
    """

    def __init__(self,
                 world_size: int,
                 sp_sizes: Optional[List[int]] = None,
                 seq_align: int = 128,
                 max_seq_per_sp: int = 1):
        self.world_size = world_size
        self.seq_align = seq_align
        self.max_seq_per_sp = max_seq_per_sp

        if sp_sizes is None:
            sp_sizes = [1 << i for i in range(world_size.bit_length())]
        self.sp_sizes = sorted(sp_sizes)

        # FLOPs threshold for each SP size (attention is O(S²))
        # Larger SP → handles longer sequences
        self._compute_thresholds()

    def _compute_thresholds(self):
        """Compute sequence length thresholds for each SP size.

        Threshold logic: a sequence of length S needs SP=n if
        S² / n > local_compute_budget (per-GPU FLOPs).

        Simplified for 8 GPUs:
          SP=1: S < 2K    (tiny, no SP comm needed)
          SP=2: 2K-8K     (short, 2-GPU Ulysses)
          SP=4: 8K-32K    (medium, 4-GPU Ulysses)
          SP=8: S > 32K   (long, 8-GPU Ulysses)
        """
        # Base: 2K per GPU is the "comfortable" local seq for B300
        base_local = 2048
        self.thresholds = {}
        for sp in self.sp_sizes:
            self.thresholds[sp] = base_local * sp

    def assign_sp_size(self, seq_len: int) -> int:
        """Determine the minimum SP size that can handle this sequence length.

        Picks the smallest SP where local_seq = seq_len / sp <= threshold.
        This minimizes communication while ensuring memory safety.
        """
        for sp in self.sp_sizes:
            local_seq = seq_len // sp
            if local_seq <= self.thresholds.get(sp, float('inf')):
                return sp
        return self.sp_sizes[-1]  # fallback: largest SP

    def schedule(self, sequence_lengths: List[int]) -> List[Microbatch]:
        """Create a microbatch schedule from a list of sequence lengths.

        Steps:
        1. Sort sequences by length (descending).
        2. Assign SP size to each sequence.
        3. Group sequences with the same SP size into DP copies.
        4. Ensure all SP groups have similar total FLOPs.

        Returns:
            List of Microbatch, ordered by SP size (largest first).
        """
        # Sort by length descending
        sorted_seqs = sorted(enumerate(sequence_lengths),
                             key=lambda x: x[1], reverse=True)

        # Assign SP size and group by SP
        sp_buckets: Dict[int, List[Tuple[int, int]]] = {sp: [] for sp in self.sp_sizes}
        for idx, seq_len in sorted_seqs:
            # Align sequence length
            aligned_len = ((seq_len + self.seq_align - 1) // self.seq_align) * self.seq_align
            sp = self.assign_sp_size(aligned_len)
            sp_buckets[sp].append((idx, aligned_len))

        # Create microbatches: each SP group gets one microbatch per DP copy
        microbatches: List[Microbatch] = []
        for sp in reversed(self.sp_sizes):  # largest SP first
            dp_size = self.world_size // sp
            bucket = sp_buckets[sp]
            if not bucket:
                continue

            # Distribute sequences across DP copies (round-robin by FLOPs)
            # For simplicity: each DP copy gets one sequence (max_seq_per_sp=1)
            for dp_copy in range(dp_size):
                if dp_copy < len(bucket):
                    idx, seq_len = bucket[dp_copy]
                else:
                    # Not enough sequences for this DP copy — skip
                    continue

                local_seq = seq_len // sp
                microbatches.append(Microbatch(
                    sp_size=sp,
                    seq_len=seq_len,
                    local_seq=local_seq,
                    dp_copy=dp_copy,
                    tokens=seq_len,
                ))

        return microbatches

    def total_tokens(self, microbatches: List[Microbatch]) -> int:
        """Sum of all tokens in the schedule (for gradient scaling)."""
        return sum(mb.tokens for mb in microbatches)

    def max_wall_clock_flops(self, microbatches: List[Microbatch],
                             hidden: int, num_layers: int) -> float:
        """Estimate max wall-clock FLOPs across SP groups.

        Attention FLOPs per layer ≈ 4 * S² * hidden (QK + AV).
        GEMM FLOPs per layer ≈ 12 * S * hidden² (QKV + O + FFN).
        """
        per_sp_flops = {}
        for mb in microbatches:
            s = mb.seq_len
            attn_flops = 4 * s * s * hidden * num_layers
            gemm_flops = 12 * s * hidden * hidden * num_layers
            total = attn_flops + gemm_flops
            # Each GPU in SP group does 1/sp of the work
            per_gpu = total / mb.sp_size
            # Accumulate per DP copy
            key = (mb.sp_size, mb.dp_copy)
            per_sp_flops[key] = per_sp_flops.get(key, 0) + per_gpu

        return max(per_sp_flops.values()) if per_sp_flops else 0.0

    # ------------------------------------------------------------------
    # THD packed scheduling
    # ------------------------------------------------------------------
    def schedule_packed(self, sequence_lengths: List[int]) -> List[PackedMicrobatch]:
        """Create a packed microbatch schedule using THD layout.

        Same SP size sequences are packed into contiguous tensors (THD layout,
        no padding). Each PackedMicrobatch is one forward pass using
        flash_attn_varlen_func.

        Packing strategy: minimize the number of DP copies used (to reduce
        dummy fill), while keeping per-DP-copy FLOPs balanced. If we have
        N sequences and dp_size slots, we use min(N, dp_size) slots — packing
        multiple sequences into the same slot when beneficial.

        Steps:
        1. Assign SP size to each sequence.
        2. Group by SP size.
        3. Within each SP group, pack into min(count, dp_size) balanced bins.
        4. Return PackedMicrobatch list, ordered by SP size (largest first).
        """
        # Step 1: assign SP and group
        sp_buckets: Dict[int, List[int]] = {sp: [] for sp in self.sp_sizes}
        for seq_len in sequence_lengths:
            aligned = ((seq_len + self.seq_align - 1) // self.seq_align) * self.seq_align
            sp = self.assign_sp_size(aligned)
            sp_buckets[sp].append(aligned)

        # Step 2: pack each SP group into balanced bins
        packed: List[PackedMicrobatch] = []
        for sp in reversed(self.sp_sizes):  # largest SP first
            bucket = sp_buckets[sp]
            if not bucket:
                continue
            dp_size = self.world_size // sp
            # Use min(count, dp_size) bins — pack multiple sequences per bin
            # when we have fewer sequences than DP slots.
            num_bins = min(len(bucket), dp_size)
            bins = self._greedy_pack(bucket, num_bins)

            for dp_copy, bin_seqs in enumerate(bins):
                if not bin_seqs:
                    continue
                total = sum(bin_seqs)
                # Ensure total is divisible by sp for even A2A split
                if total % sp != 0:
                    pad = sp - (total % sp)
                    bin_seqs = list(bin_seqs) + [pad]
                    total += pad
                cu_seqlens = [0]
                for s in bin_seqs:
                    cu_seqlens.append(cu_seqlens[-1] + s)

                packed.append(PackedMicrobatch(
                    sp_size=sp,
                    seq_lens=bin_seqs,
                    total_tokens=total,
                    local_tokens=total // sp,
                    cu_seqlens=cu_seqlens,
                    dp_copy=dp_copy,
                ))

        return packed

    @staticmethod
    def _greedy_pack(seqs: List[int], num_bins: int) -> List[List[int]]:
        """Greedily pack sequences into num_bins bins, balanced by FLOPs (S²).

        Uses LPT (Longest Processing Time first) rule: sort by FLOPs descending,
        assign each to the bin with smallest current load. This gives a
        4/3-approximation of optimal makespan.
        """
        sorted_seqs = sorted(seqs, key=lambda s: s * s, reverse=True)

        bins: List[List[int]] = [[] for _ in range(num_bins)]
        bin_flops = [0.0] * num_bins

        for s in sorted_seqs:
            min_bin = bin_flops.index(min(bin_flops))
            bins[min_bin].append(s)
            bin_flops[min_bin] += s * s

        return bins
