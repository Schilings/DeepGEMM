"""Compatibility alias for the standard synchronous Ulysses path.

There is intentionally no "fused standard" arm in the POST-only ablation.  The
class remains for older command lines, but it executes exactly the pure PyTorch
baseline and allocates no DeepGEMM workspace.
"""

from .serial import SerialUlysses


class FusedStandardUlysses(SerialUlysses):
    pass
