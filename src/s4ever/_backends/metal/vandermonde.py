"""S4D's convolution kernel and its gradients on Metal.

The kernels use the method of `_backends.tilelang.kernels.vandermonde`. The backward reduces the
gradient g of k to the sums S0 and S1 and computes the gradients of the parameters from them. The
kernels compute k, S0 and S1 as sums of simdgroup matrix products over blocks of T positions.

    z_n = dA_n
    k[c, l] = 2 Re sum_n C[c, n] dB_n z_n^l
    S0[n] = sum_l g_l conj(z_n)^l
    S1[n] = sum_l l g_l conj(z_n)^l
"""

import functools

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div

from . import msl

THREADS = 128  # threads per threadgroup
T = 32  # positions of a block
KM = 32  # modes per step of `forward`, one per lane
TILES = 16  # matrix tiles per threadgroup of `forward`
MT = 16  # modes per threadgroup of `sums`
ROWS = 16  # rows of the matrix G of `sums`, one per segment of a channel
MAX_LENGTH = 1 << 24  # the kernels hold positions in float32, which is exact up to 2^24


@functools.cache
def library(C: int, disc: str):
    constants = f"""
constant constexpr uint THREADS = {THREADS}, T = {T}, STEP_MODES = {KM}, TILES = {TILES};
constant constexpr uint GROUP_MODES = {MT}, ROWS = {ROWS}, C = {C}, SIMDS = THREADS / 32;
"""
    src = msl.source(msl.discretize(disc) + constants, "discretize.metal", "vandermonde.metal")
    return msl.library(src, dict.fromkeys(("forward", "sums", "grads"), THREADS))


def fits(N: int, L: int, disc: str) -> bool:
    fit = N <= shared.MAX_CONV_KERNEL_MODES and L <= MAX_LENGTH
    return fit and builds(disc)


@msl.constant
def builds(disc: str) -> bool:
    """Return whether the kernels build for the most channels, and warn if they do not.

    The kernels use the most registers at this channel count.
    """
    return msl.builds("S4D's convolution kernels", library, shared.MAX_CHANNELS, disc)


def kernel_outputs(dt, A, B, C, L, disc):
    return C.new_empty(*C.shape[:2], L)


@shared.register_op("metal_s4d_conv_kernel", kernel_outputs, "MPS")
def kernel_op(dt: Tensor, A: Tensor, B: Tensor, C: Tensor, L: int, disc: str) -> Tensor:
    k = kernel_outputs(dt, A, B, C, L, disc)
    Cc, H, N, _ = C.shape
    tiles = ceil_div(L, 8 * T)  # tiles of 8 T positions per channel
    threads = [H * ceil_div(Cc * tiles, TILES) * THREADS, 1, 1]
    library(Cc, disc).forward(k, dt, A, B, C, H, N, L, threads=threads, group_size=[THREADS, 1, 1])
    return k


def grads_outputs(dt, A, B, C, g, disc):
    return [torch.empty_like(t) for t in (dt, A, B, C)]


@shared.register_op("metal_s4d_conv_kernel_grads", grads_outputs, "MPS")
def grads_op(dt: Tensor, A: Tensor, B: Tensor, C: Tensor, g: Tensor, disc: str) -> list[Tensor]:
    (Cc, H, N, _), L = C.shape, g.shape[-1]
    lib = library(Cc, disc)
    blocks = ceil_div(L, T)
    # R is the number of channels per threadgroup that gives the fewest blocks in total, where the
    # setup of each threadgroup counts as one block. R is at most 4, because the ops support at
    # most MAX_CHANNELS = 4 channels.
    R = min((1, 2, 4), key=lambda R: ceil_div(Cc, R) * (ceil_div(blocks * R, ROWS) + 1))
    segments = ROWS // R  # segments per channel in a threadgroup
    groups = ceil_div(Cc, R) * H * ceil_div(N, MT)  # threadgroups per chunk
    # The chunks give the dispatch about msl.GROUPS threadgroups, unless each segment of a chunk
    # would then be shorter than one block. Rounding the segments up to whole blocks can leave
    # fewer chunks than `target`.
    target = min(ceil_div(msl.GROUPS, groups), ceil_div(blocks, segments))
    span = ceil_div(blocks, segments * target) * T  # positions per segment
    chunks = ceil_div(L, segments * span)
    S = A.new_empty(Cc, H, chunks, N, 4)
    ins = (g, dt, A, B, H, N, L, span, segments.bit_length() - 1, chunks)
    lib.sums(S, *ins, threads=[groups * chunks * THREADS, 1, 1], group_size=[THREADS, 1, 1])
    out = grads_outputs(dt, A, B, C, g, disc)
    threads = [H * THREADS, 1, 1]
    lib.grads(*out, S, dt, A, B, C, H, N, chunks, threads=threads, group_size=[THREADS, 1, 1])
    return out


class S4DConvKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, B, C, L, disc):
        ctx.save_for_backward(dt, A, B, C)
        ctx.disc = disc
        return kernel_op(dt, A, B, C, L, disc)

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        return *grads_op(*ctx.saved_tensors, grads[0].contiguous(), ctx.disc), None, None


def s4d_conv_kernel(dt: Tensor, A: Tensor, B: Tensor, C: Tensor, L: int, disc: str) -> Tensor:
    """Return `s4ever.ops.s4d_conv_kernel` computed by the Metal kernels.

    Callers check `fits` first.
    """
    return S4DConvKernel.apply(*shared.contiguous_f32(dt, A, B, C), L, disc)
