"""Pure PyTorch synchronous Ulysses baseline.

PRE and attention are implemented once in :mod:`wan21.sp.base`.  POST performs a
synchronous NCCL all-to-all followed by the replicated output projection.  No
DeepGEMM communication-fused operator is used anywhere in this strategy.
"""

import torch

from .base import NCCLAllToAll, UlyssesBase


class SerialUlysses(UlyssesBase):
    """Ablation baseline: torch linear, FA4 and synchronous NCCL collectives."""

    def __init__(self, config, sp_config):
        sp_config.use_fused_ops = False
        sp_config.post_strategy = 'a2a_gemm'
        super().__init__(config, sp_config)

    def _post_forward(self, o, lbs, lseq, llseq, grid, **kw):
        """POST: synchronous A2A transpose, then ordinary torch output linear."""
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        local_m = lbs * llseq
        send = (
            o.transpose(1, 2)
            .reshape(lbs, self.local_nh, sp, llseq, hd)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(local_m, hidden)
        return self.model.o(gathered)
