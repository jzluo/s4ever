"""Transposes between the ops' (B, L, H) layout and the kernels' (B, H, L) layout.

The FFT and chunk kernels read and write sequences in the (B, H, L) layout.
"""

import functools

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div

from . import msl

TILE = 32, 64  # the rows and columns of x that a threadgroup transposes
TILE_THREADS = 256


@functools.cache
def transposes(dtype: torch.dtype):
    BM, BN = TILE
    prelude = f"constant constexpr uint BM = {BM}, BN = {BN}, THREADS = {TILE_THREADS};\n"
    src = msl.source(prelude + f"typedef {msl.TYPES[dtype]} IO;\n", "transpose.metal")
    return msl.library(src, {"transpose": TILE_THREADS})


def transposed(x: Tensor) -> Tensor:
    return x.new_empty(x.shape[0], x.shape[2], x.shape[1])


@shared.register_op("metal_transpose", transposed, "MPS")
def transpose_op(x: Tensor) -> Tensor:
    B, M, N = x.shape
    BM, BN = TILE
    y = transposed(x)
    threads = [B * ceil_div(M, BM) * ceil_div(N, BN) * TILE_THREADS, 1, 1]
    transposes(x.dtype).transpose(y, x, M, N, threads=threads, group_size=[TILE_THREADS, 1, 1])
    return y


def transpose(x: Tensor) -> Tensor:
    t = x.mT
    return t if t.is_contiguous() else transpose_op(x.contiguous())
