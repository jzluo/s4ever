"""Recurrent scans on MPS, computed with the method of `_backends.tilelang.kernels.recurrent`.

The ops run these scans for inference, in mode='recurrent' and in stepping.
"""

import functools

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div, recurrent_outputs

from . import msl

ROWS = 4  # batch rows per threadgroup, one per SIMD group
# With 4 channels, the `s4` kernel keeps 72 bytes of tables per mode in threadgroup memory. Apple
# GPUs allow 32 KB of threadgroup memory, which 512 modes exceed.
MAX_MODES = 256


@functools.cache
def library(lane_modes: int, C: int, disc: str, dtype: torch.dtype):
    constants = f"""
constant constexpr uint ROWS = {ROWS}, LANE_MODES = {lane_modes}, C = {C};
typedef {msl.TYPES[dtype]} IO;
"""
    src = msl.source(msl.discretize(disc) + constants, "discretize.metal", "recurrent.metal")
    return msl.library(src, dict.fromkeys(("s4d", "s4"), 32 * ROWS))


def fits(N: int, dtype: torch.dtype) -> bool:
    return N <= MAX_MODES and builds(dtype)


@msl.constant
def builds(dtype: torch.dtype) -> bool:
    """Return whether the kernels build for the most modes and channels.

    These sizes need the most registers.
    """
    args = (MAX_MODES // 32, shared.MAX_CHANNELS, "zoh", dtype)
    return msl.builds(f"the recurrent kernels for {dtype}", library, *args)


def launch(name, disc, u, params, C, D, x0, reverse):
    (Bsz, L, H), N = u.shape, C.shape[2]
    y, xN = recurrent_outputs(u, params[1], C)
    lib = library(ceil_div(N, 32), C.shape[0], disc, u.dtype)
    flags = (int(D is not None), int(x0 is not None), int(reverse))
    # A stands in for D and x0 when the flags tell the kernels not to read them.
    D, x0 = (params[1] if t is None else t for t in (D, x0))
    threads = [H * ceil_div(Bsz, ROWS) * 32 * ROWS, 1, 1]
    ins = (u, *params, C, D, x0, Bsz, H, N, L, *flags)
    getattr(lib, name)(y, xN, *ins, threads=threads, group_size=[32 * ROWS, 1, 1])
    return y, xN


@shared.register_op(
    "metal_s4d_recurrent", lambda u, dt, A, B, C, *_: recurrent_outputs(u, A, C), "MPS"
)
def s4d_recurrent_op(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    x0: Tensor | None,
    disc: str,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    return launch("s4d", disc, u, (dt, A, B), C, D, x0, reverse)


@shared.register_op(
    "metal_s4_recurrent", lambda u, dt, A, P, B, c, *_: recurrent_outputs(u, A, c), "MPS"
)
def s4_recurrent_op(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    P: Tensor,
    B: Tensor,
    c: Tensor,
    D: Tensor | None,
    x0: Tensor | None,
    reverse: bool,
) -> tuple[Tensor, Tensor]:
    # The s4 kernel does not discretize, so it can use any library, and "zoh" selects the one that
    # S4D's default discretization uses.
    return launch("s4", "zoh", u, (dt, A, P, B), c, D, x0, reverse)


def s4d_recurrent(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    initial_state: Tensor | None,
    output_final_state: bool,
    reverse: bool,
    disc: str,
) -> tuple[Tensor, Tensor | None]:
    """Run `s4ever.ops.s4d` in recurrent mode on MPS without gradients.

    Callers check `fits` first.
    """
    params = shared.contiguous_f32(dt, A, B, C, D, initial_state)
    y, xN = s4d_recurrent_op(u.contiguous(), *params, disc, reverse)
    return y, xN if output_final_state else None


def s4_recurrent(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    P: Tensor,
    B: Tensor,
    cvec: Tensor,
    D: Tensor | None,
    initial_state: Tensor | None,
    output_final_state: bool,
    reverse: bool,
) -> tuple[Tensor, Tensor | None]:
    """Run `s4ever.ops.s4` in recurrent mode on MPS without gradients.

    Callers check `fits` first. `cvec` (C, H, 2N) is the first output of
    `s4ever.ops._s4.s4_recurrence`, the vector 2 conj(c) for the readout c.
    """
    cvec = cvec.unflatten(-1, (-1, 2))
    params = shared.contiguous_f32(dt, A, P, B, cvec, D, initial_state)
    y, xN = s4_recurrent_op(u.contiguous(), *params, reverse)
    return y, xN if output_final_state else None
