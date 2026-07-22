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
        """POST: synchronous A2A transpose, then ordinary torch output linear.

        Works for both fixed-shape and THD packed modes:
        - Fixed: o [lbs, lseq, local_nh, hd]
        - Packed: o [total_tokens, local_nh, hd] (lbs=1, lseq=total_tokens)
        """
        sp = self.sp_size
        hd = self.head_dim
        hidden = self.cfg.dim
        local_m = lbs * llseq
        # Reshape o to [lbs, lseq, local_nh, hd] (packed: already [1, total, nh, hd])
        if o.dim() == 3:
            o = o.unsqueeze(0)  # [total_tokens, nh, hd] → [1, total_tokens, nh, hd]
        send = (
            o.transpose(1, 2)
            .reshape(lbs, self.local_nh, sp, llseq, hd)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        recv = NCCLAllToAll.apply(send, self.group)
        gathered = recv.permute(1, 2, 0, 3, 4).reshape(local_m, hidden)
        return self.model.o(gathered)
