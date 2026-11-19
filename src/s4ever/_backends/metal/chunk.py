"""S4D's chunk mode on MPS, computed with the method of `_backends.tilelang.kernels.chunk`.

The kernels scan 8 batch rows at a time in chunks of T steps. U (8, T) holds the inputs of a chunk,
and X holds the states at its start, with the re and im of each mode in adjacent columns. The
outputs Y of the chunk and the states X' after it are

    Y = U KtT + X OutT
    X' = z^T X + U InT
"""

import functools
import math

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div

from . import msl
from .transpose import transpose, transposes

T = 16  # steps of a chunk
SEG = 4  # chunks between the states that the forward saves
FS = 2  # row blocks of 8 batch rows in a forward threadgroup
GROUP = 32  # most state tiles of a mode group
THREADS = 128  # threads of a `matrices` or `grads` threadgroup
# `matrices` keeps 16 bytes of tables per mode in threadgroup memory. Apple GPUs allow 32 KB of
# threadgroup memory, which 2048 modes exceed.
MAX_MODES = 1024


def tiles(N: int, C: int) -> tuple[int, int, int, int]:
    """Return the counts (state_tiles, group_tiles, bwd_simds, fwd_simds) of the chunk kernels.

    A state tile holds 4 modes. The kernels split the state tiles into mode groups of group_tiles
    tiles, and they split the tiles of a mode group over the bwd_simds SIMD groups of `backward`
    and over the fwd_simds SIMD groups of a row block in `forward`. The tile counts are padded so
    that both SIMD-group counts divide group_tiles.
    """
    state_tiles = ceil_div(N, 4)
    groups = ceil_div(state_tiles, GROUP)
    group_tiles = ceil_div(state_tiles, groups)
    # A backward SIMD group handles at most 4 tiles with one channel and at most 2 tiles with more
    # channels, because the accumulators of 4 tiles and 2 channels do not fit in registers. The
    # backward has at least 2 SIMD groups, so its last SIMD group, which computes G KtT^T, is never
    # SIMD group 0, which computes U^T G. A forward SIMD group holds at most 16 state tiles.
    bwd_simds = max(2, ceil_div(group_tiles, 4 if C == 1 else 2))
    fwd_simds = ceil_div(group_tiles, 16)
    step = math.lcm(bwd_simds, fwd_simds)
    group_tiles = ceil_div(group_tiles, step) * step
    return groups * group_tiles, group_tiles, bwd_simds, fwd_simds


def layout(state_tiles: int, C: int) -> dict[str, int]:
    """Return the offsets and sizes in floats of `mats` and of the partials.

    chunk.metal takes them as constants and describes what `mats` holds for a feature and what the
    partials hold for a row block.
    """
    M = 8 * state_tiles
    mats_z = C * T * T + T * M + C * M * T
    part_in = C * M * T
    part_kt = part_in + T * M
    part_diag = part_kt + C * T * T
    return {
        "MATS_Z": mats_z,
        "MATS_SIZE": mats_z + 2 * M,
        "PART_IN": part_in,
        "PART_KT": part_kt,
        "PART_DIAG": part_diag,
        "PART_SIZE": part_diag + 16 * M,
    }


@functools.cache
def library(
    state_tiles: int,
    group_tiles: int,
    bwd_simds: int,
    fwd_simds: int,
    C: int,
    disc: str,
    dtype: torch.dtype,
):
    offsets = ", ".join(f"{name} = {value}" for name, value in layout(state_tiles, C).items())
    constants = f"""
constant constexpr uint T = {T}, SEG = {SEG}, FS = {FS}, FWD_SIMDS = {fwd_simds};
constant constexpr uint STATE_TILES = {state_tiles}, GROUP_TILES = {group_tiles};
constant constexpr uint BWD_SIMDS = {bwd_simds}, C = {C}, THREADS = {THREADS};
constant constexpr uint {offsets};
typedef {msl.TYPES[dtype]} IO;
typedef {msl.TYPES[out_dtype(state_tiles, group_tiles, dtype)]} YO;  // y and du
"""
    src = msl.source(msl.discretize(disc) + constants, "discretize.metal", "chunk.metal")
    threads = {
        "matrices": THREADS,
        "forward": 32 * FS * fwd_simds,
        "backward": 32 * bwd_simds,
        "grads": THREADS,
    }
    return msl.library(src, threads)


def out_dtype(state_tiles: int, group_tiles: int, dtype: torch.dtype) -> torch.dtype:
    """Return the dtype in which the kernels write y and du.

    With several mode groups, each group adds its terms to y and du, so the kernels write them in
    float32, and the result is rounded to `dtype` once.
    """
    return dtype if state_tiles == group_tiles else torch.float32


def fits(N: int, C: int, disc: str, dtype: torch.dtype) -> bool:
    return N <= MAX_MODES and builds(*tiles(N, C), C, disc, dtype)


@msl.constant
def builds(
    state_tiles: int,
    group_tiles: int,
    bwd_simds: int,
    fwd_simds: int,
    C: int,
    disc: str,
    dtype: torch.dtype,
) -> bool:
    def build():
        library(state_tiles, group_tiles, bwd_simds, fwd_simds, C, disc, dtype)
        transposes(dtype)

    return msl.builds(f"S4D's chunk kernels for {dtype}", build)


def forward_outputs(u, dt, A, B, C, D, x0, disc, reverse, final, save):
    (Bsz, H, L), N = u.shape, A.shape[1]
    state_tiles = tiles(N, len(C))[0]
    segments = ceil_div(L, T * SEG)
    xN = A.new_empty((Bsz, H, N, 2) if final else 0)
    Xs = A.new_empty((H, ceil_div(Bsz, 8), segments, 8, 8 * state_tiles) if save else 0)
    mats = A.new_empty(H, layout(state_tiles, len(C))["MATS_SIZE"])
    return u.new_empty(Bsz, len(C), H, L), xN, Xs, mats


@shared.register_op("metal_s4d_chunk", forward_outputs, "MPS")
def s4d_chunk_op(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    x0: Tensor | None,
    disc: str,
    reverse: bool,
    final: bool,
    save: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    y, xN, Xs, mats = forward_outputs(u, dt, A, B, C, D, x0, disc, reverse, final, save)
    (Bsz, H, L), N = u.shape, A.shape[1]
    state_tiles, group_tiles, bwd_simds, fwd_simds = tiles(N, len(C))
    lib = library(state_tiles, group_tiles, bwd_simds, fwd_simds, len(C), disc, u.dtype)
    dtype = out_dtype(state_tiles, group_tiles, u.dtype)
    out = y if dtype == y.dtype else torch.empty_like(y, dtype=dtype)
    # A stands in for the buffers that the flags tell the kernels not to access.
    ins = (dt, A, B, C, A if D is None else D, H, N, L, int(D is not None))
    lib.matrices(mats, *ins, threads=[H * THREADS, 1, 1], group_size=[THREADS, 1, 1])
    groups, width = H * ceil_div(Bsz, 8 * FS), 32 * FS * fwd_simds
    xN_, Xs_, x0_ = (A if t.numel() == 0 else t for t in (xN, Xs, A if x0 is None else x0))
    ins = (u, x0_, mats, H, N, L, Bsz, int(reverse), int(x0 is not None), int(final), int(save))
    lib.forward(out, xN_, Xs_, *ins, threads=[groups * width, 1, 1], group_size=[width, 1, 1])
    if out is not y:
        y.copy_(out)
    return y, xN, Xs, mats


def backward_outputs(u, gy, gf, dt, A, B, C, D, Xs, mats, disc, reverse, want_dx0):
    dD = A.new_empty(0) if D is None else torch.empty_like(D)
    dx0 = A.new_empty((len(u), *A.shape[:2], 2) if want_dx0 else 0)
    return [torch.empty_like(u), *(torch.empty_like(t) for t in (dt, A, B, C)), dD, dx0]


@shared.register_op("metal_s4d_chunk_backward", backward_outputs, "MPS")
def s4d_chunk_backward_op(
    u: Tensor,
    gy: Tensor,
    gf: Tensor | None,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    Xs: Tensor,
    mats: Tensor,
    disc: str,
    reverse: bool,
    want_dx0: bool,
) -> list[Tensor]:
    grads = backward_outputs(u, gy, gf, dt, A, B, C, D, Xs, mats, disc, reverse, want_dx0)
    du, ddt, dA, dB, dC, dD, dx0 = grads
    (Bsz, H, L), N = u.shape, A.shape[1]
    state_tiles, group_tiles, bwd_simds, fwd_simds = tiles(N, len(C))
    lib = library(state_tiles, group_tiles, bwd_simds, fwd_simds, len(C), disc, u.dtype)
    dtype = out_dtype(state_tiles, group_tiles, u.dtype)
    out = du if dtype == du.dtype else torch.empty_like(du, dtype=dtype)
    blocks = H * ceil_div(Bsz, 8)
    part = A.new_empty(blocks, layout(state_tiles, len(C))["PART_SIZE"])
    scratch = A.new_empty(blocks, SEG * 64 * state_tiles)
    dx0_, gf_, dD_ = (A if t is None or t.numel() == 0 else t for t in (dx0, gf, dD))
    ins = (u, gy, gf_, Xs, mats, H, N, L, Bsz, int(reverse), int(gf is not None), int(want_dx0))
    threads = [blocks * 32 * bwd_simds, 1, 1]
    lib.backward(out, dx0_, part, scratch, *ins, threads=threads, group_size=[32 * bwd_simds, 1, 1])
    ins = (dt, A, B, C, H, N, L, blocks // H, int(D is not None))
    lib.grads(
        ddt, dA, dB, dC, dD_, part, *ins, threads=[H * THREADS, 1, 1], group_size=[THREADS, 1, 1]
    )
    if out is not du:
        du.copy_(out)
    return grads


class S4DChunk(torch.autograd.Function):
    """Autograd Function for S4D's chunk mode on u (B, L, H)."""

    @staticmethod
    def forward(ctx, u, dt, A, B, C, D, x0, config, grad):
        ctx.set_materialize_grads(False)
        u = transpose(u)
        y, xN, Xs, mats = s4d_chunk_op(u, dt, A, B, C, D, x0, *config, grad)
        ctx.save_for_backward(u, dt, A, B, C, D, Xs, mats)
        ctx.config = config
        y = transpose(y.flatten(1, 2)).unflatten(2, y.shape[1:3])
        # Without `final`, the second output is an empty tensor and not None. When a backward
        # queries the layout of a gradient, as `transpose` does, Dynamo (PyTorch 2.14 and 2.15)
        # retraces it and calls contiguous() on every gradient, and the gradient of a None output
        # is None. To reproduce, compile with fullgraph=True an autograd.Function whose forward
        # returns (x * 2, None) and whose backward calls gy.mT.is_contiguous(). The call raises
        # AttributeError('contiguous').
        return y, xN if config[-1] else y.new_empty(0)

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        (u, dt, A, B, C, D, Xs, mats), (gy, gf) = ctx.saved_tensors, grads
        disc, reverse, final = ctx.config
        if gy is None:
            gy = u.new_zeros(len(u), len(C), *u.shape[1:])
        else:
            gy = transpose(gy.flatten(2)).unflatten(1, gy.shape[2:])
        gf = gf.contiguous() if final and gf is not None else None
        want_dx0 = ctx.needs_input_grad[6]
        grads = s4d_chunk_backward_op(u, gy, gf, dt, A, B, C, D, Xs, mats, disc, reverse, want_dx0)
        dD, dx0 = grads[5] if D is not None else None, grads[6] if want_dx0 else None
        du = transpose(grads[0]) if ctx.needs_input_grad[0] else None
        return du, *grads[1:5], dD, dx0, None, None


def s4d_chunk(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    x0: Tensor | None,
    final: bool,
    reverse: bool,
    disc: str,
) -> tuple[Tensor, Tensor | None]:
    """Run `s4ever.ops.s4d` in chunk mode on MPS, for inputs that `fits` accepts.

    The output y (B, L, C, H) has the dtype of u, and the final states are returned only with
    `final`.
    """
    grad = shared.tracks_grad(u, dt, A, B, C, D, x0)
    params = shared.contiguous_f32(dt, A, B, C, D, x0)
    y, xN = S4DChunk.apply(u, *params, (disc, reverse, final), grad)
    return y, xN if final else None
