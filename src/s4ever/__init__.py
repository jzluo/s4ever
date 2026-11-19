"""S4 and S4D state space sequence layers for PyTorch.

The GPU kernels (TileLang on CUDA and ROCm, Metal on MPS) are compiled at first use, so the package
has no extension to build.
"""

from ._block import S4Block
from ._layers import S4, S4D, param_groups

__all__ = ["S4", "S4D", "S4Block", "param_groups"]
