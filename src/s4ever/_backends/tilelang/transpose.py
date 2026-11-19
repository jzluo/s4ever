"""Transposes between the ops' (B, L, H) layout and the kernels' (B, H, L) layout.

The FFT, chunk and final-state kernels read and write sequences in the (B, H, L) layout.
"""

import torch
from torch import Tensor

from s4ever._backends.shared import register_op

from .runtime import launcher


def transposed(x: Tensor, dtype: torch.dtype) -> Tensor:
    return x.new_empty(x.shape[0], x.shape[2], x.shape[1], dtype=dtype)


@register_op("transpose", transposed)
def transpose_op(x: Tensor, dtype: torch.dtype) -> Tensor:
    y = transposed(x, dtype)
    transpose_kernel(x.device, x.dtype, dtype)(x, y)
    return y


@launcher
def transpose_kernel(device: torch.device, dtype: torch.dtype, out: torch.dtype):
    from s4ever._backends.tilelang.kernels import transpose as t
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    return t.kernel(DTYPES[dtype], DTYPES[out])


def transpose(x: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Return x (B, M, N) as a contiguous (B, N, M), in dtype if it is given."""
    dtype = dtype or x.dtype
    # PyTorch makes x.mT contiguous without a transpose when x has stride 1 along M, as a
    # transposed view of a contiguous tensor has. Below 32 rows or columns, PyTorch transposes x
    # too, because the kernel's 32 x 64 tiles would be partly empty.
    if min(x.shape[1:]) < 32 or x.stride(1) == 1:
        return x.mT.to(dtype).contiguous()
    return transpose_op(x.contiguous(), dtype)


def features_last(y: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Return the kernels' y (B, C, H, L) as a contiguous (B, L, C, H), in dtype if it is given."""
    return transpose(y.flatten(1, 2), dtype).unflatten(2, y.shape[1:3])


def sequences_last(gy: Tensor) -> Tensor:
    """Return gy (B, L, C, H) as a contiguous (B, C, H, L) for the kernels."""
    return transpose(gy.flatten(2)).unflatten(1, gy.shape[2:])
