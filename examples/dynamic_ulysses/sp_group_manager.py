"""Dynamic SP group manager — pre-creates power-of-2 SP groups.

Inspired by Megatron's create_hybrid_dp_cp_groups and ByteScale's global
communication group reuse.  For 8 GPUs, pre-creates SP groups of size
1, 2, 4, 8 so that any microbatch can use the appropriate SP degree
without runtime NCCL group creation overhead.

Layout for 8 ranks (world_size=8):

  size=1: [0], [1], [2], [3], [4], [5], [6], [7]          (8 DP copies)
  size=2: [0,1], [2,3], [4,5], [6,7]                      (4 DP copies)
  size=4: [0,1,2,3], [4,5,6,7]                            (2 DP copies)
  size=8: [0,1,2,3,4,5,6,7]                               (1 SP group)

Each rank stores a dict: {sp_size: (sp_group, dp_group, sp_rank, dp_rank)}.
Groups are created once at init and reused across all steps.
"""

import torch
import torch.distributed as dist
from typing import Dict, Tuple, List, Optional


class DynamicSPGroupManager:
    """Pre-created power-of-2 SP/DP groups for dynamic sequence parallelism.

    Args:
        world_size: Total number of GPUs.
        group:      The world process group (or a subgroup).
    """

    def __init__(self, world_size: int, group: 'dist.ProcessGroup' = None):
        assert world_size > 0 and (world_size & (world_size - 1)) == 0, \
            f"world_size must be power of 2, got {world_size}"
        self.world_size = world_size
        self.group = group if group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.group)

        # Generate all valid SP sizes: 1, 2, 4, ..., world_size
        self.sp_sizes: List[int] = [1 << i for i in range(world_size.bit_length())]

        # Per-rank storage: {sp_size: SPGroupInfo}
        # SPGroupInfo contains sp_group, dp_group, sp_rank, dp_size, dp_rank
        self.groups: Dict[int, SPGroupInfo] = {}

        self._create_all_groups()

    def _create_all_groups(self):
        """Pre-create NCCL groups for all power-of-2 SP sizes."""
        for sp_size in self.sp_sizes:
            dp_size = self.world_size // sp_size
            assert dp_size >= 1

            sp_group = None
            dp_group = None
            sp_rank = -1
            dp_rank = -1

            # Create SP groups: contiguous blocks of sp_size ranks
            for i in range(0, self.world_size, sp_size):
                ranks = list(range(i, i + sp_size))
                new_group = dist.new_group(ranks)
                if self.rank in ranks:
                    sp_group = new_group
                    sp_rank = ranks.index(self.rank)

            # Create DP groups: strided ranks with step=sp_size
            # DP group j contains ranks [j, j+sp_size, j+2*sp_size, ...]
            for j in range(sp_size):
                ranks = list(range(j, self.world_size, sp_size))
                if len(ranks) < 2:
                    # DP size 1: no group needed
                    if self.rank in ranks:
                        dp_group = None  # single rank, no DP comm
                        dp_rank = 0
                    continue
                new_group = dist.new_group(ranks)
                if self.rank in ranks:
                    dp_group = new_group
                    dp_rank = ranks.index(self.rank)

            self.groups[sp_size] = SPGroupInfo(
                sp_size=sp_size,
                dp_size=dp_size,
                sp_group=sp_group,
                dp_group=dp_group,
                sp_rank=sp_rank,
                dp_rank=dp_rank,
            )

    def get_groups(self, sp_size: int) -> 'SPGroupInfo':
        """Get pre-created SP/DP groups for the given SP size."""
        assert sp_size in self.groups, \
            f"sp_size {sp_size} not available (valid: {list(self.groups.keys())})"
        return self.groups[sp_size]

    def get_valid_sp_sizes(self) -> List[int]:
        """Return all valid SP sizes."""
        return self.sp_sizes

    def destroy(self):
        """Destroy all created groups (optional, NCCL cleans up on exit)."""
        self.groups.clear()


class SPGroupInfo:
    """Container for SP/DP group metadata (MegaMoE-style plain attributes)."""
    __slots__ = ('sp_size', 'dp_size', 'sp_group', 'dp_group', 'sp_rank', 'dp_rank')

    def __init__(self, sp_size: int, dp_size: int,
                 sp_group: Optional['dist.ProcessGroup'],
                 dp_group: Optional['dist.ProcessGroup'],
                 sp_rank: int, dp_rank: int):
        self.sp_size = sp_size
        self.dp_size = dp_size
        self.sp_group = sp_group
        self.dp_group = dp_group
        self.sp_rank = sp_rank
        self.dp_rank = dp_rank

    def __repr__(self):
        return (f"SPGroupInfo(sp={self.sp_size}, dp={self.dp_size}, "
                f"sp_rank={self.sp_rank}, dp_rank={self.dp_rank})")
