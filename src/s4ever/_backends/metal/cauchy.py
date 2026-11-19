"""S4's convolution kernel from its parameters (`s4_conv_kernel`) on Metal.

The kernels use the method that `_backends.tilelang.kernels.cauchy` describes.
"""

import functools

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div

from . import fft, msl

BK = 128  # frequencies per threadgroup of `spectrum`, one per thread


@functools.cache
def spectrum_library(N: int, C: int):
    prelude = f"constant constexpr uint THREADS = {BK}, N = {N}, C = {C};\n"
    names = ("spectrum", "spectrum_grads", "spectrum_finish")
    return msl.library(msl.source(prelude, "cauchy_spectrum.metal"), dict.fromkeys(names, BK))


def fits(N: int) -> bool:
    return N <= shared.MAX_CONV_KERNEL_MODES and builds()


@msl.constant
def builds() -> bool:
    """Return whether the kernels build for the most modes and channels, and warn if they do not.

    The kernels use the most registers and threadgroup memory at these sizes.
    """
    args = (shared.MAX_CONV_KERNEL_MODES, shared.MAX_CHANNELS)
    return msl.builds("S4's convolution kernels", spectrum_library, *args)


@shared.register_op(
    "metal_s4_conv_kernel", lambda dt, A, P, B, C, L: C.new_empty(*C.shape[:2], L), "MPS"
)
def s4_conv_kernel_op(dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, L: int) -> Tensor:
    Cc, H, N, _ = C.shape
    F = L // 2 + 1
    k_f = C.new_empty(Cc, H, F, 2)
    threads, group = [H * ceil_div(F, BK) * BK, 1, 1], [BK, 1, 1]
    # The kernels run for at most `shared.MAX_CHANNELS` channels at a time, but
    # `ops._s4.state_response` passes each batch row of x0 as a channel, and the batch can have
    # more rows.
    for c in range(0, Cc, shared.MAX_CHANNELS):
        Cs = C[c : c + shared.MAX_CHANNELS]
        lib = spectrum_library(N, len(Cs))
        lib.spectrum(k_f[c : c + len(Cs)], dt, A, P, B, Cs, H, L, threads=threads, group_size=group)
    if fft.real_fits(L):
        return fft.real_inverse(k_f, L)
    return torch.fft.irfft(torch.view_as_complex(k_f), n=L, norm="forward")


def grads_outputs(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, g: Tensor
) -> list[Tensor]:
    return [torch.empty_like(x) for x in (dt, A, P, B, C)]


@shared.register_op("metal_s4_conv_kernel_grads", grads_outputs, "MPS")
def s4_conv_kernel_grads_op(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, g: Tensor
) -> list[Tensor]:
    (Cc, H, N, _), L = C.shape, g.shape[-1]
    gf = torch.view_as_real(torch.fft.rfft(g, n=L))
    n_freq = (L + 1) // 2 - 1  # the frequencies 0 < k < L / 2
    # Each threadgroup sums `split_tiles` tiles of BK frequencies, so that the dispatch has about
    # msl.GROUPS threadgroups.
    tiles = ceil_div(n_freq, BK)
    group = [BK, 1, 1]
    split_tiles = ceil_div(tiles, min(tiles, ceil_div(msl.GROUPS, H))) if tiles else 0
    splits = ceil_div(tiles, split_tiles) if tiles else 1
    # Each kernel runs once for all channels. The backward has at most `shared.MAX_CHANNELS`
    # channels, because S4 runs `ops._s4.state_response` only without gradients.
    grads = grads_outputs(dt, A, P, B, C, g)
    lib = spectrum_library(N, Cc)
    rows = 4 * (Cc + 1) + 2  # COEFS + 2 in cauchy_spectrum.metal
    part = C.new_empty(H, splits, rows, N) if tiles else C.new_zeros(H, 1, rows, N)
    args = (dt, A, P, B, C, H, L, splits)
    if tiles:
        lib.spectrum_grads(
            part, gf, *args, split_tiles, threads=[H * splits * BK, 1, 1], group_size=group
        )
    lib.spectrum_finish(*grads, part, gf, *args, threads=[H * BK, 1, 1], group_size=group)
    return grads


class S4ConvKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, P, B, C, L):
        ctx.save_for_backward(dt, A, P, B, C)
        return s4_conv_kernel_op(dt, A, P, B, C, L)

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        return *s4_conv_kernel_grads_op(*ctx.saved_tensors, grads[0].contiguous()), None


def s4_conv_kernel(dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, L: int) -> Tensor:
    """Return `s4ever.ops.s4_conv_kernel` computed by the Metal kernels.

    Callers check `fits` first.
    """
    return S4ConvKernel.apply(*shared.contiguous_f32(dt, A, P, B, C), L)
