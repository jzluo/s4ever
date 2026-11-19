"""Fused fftconv.

Two batch rows form a pair, which the kernels transform as one complex sequence. A thread block
processes G pairs in each of its rounds. In a transform that is split over N1 thread blocks, each
block computes a share of every output (`kernels.fft`).
"""

import functools

import torch
import torch.nn.functional as F
from torch import Tensor

from s4ever._backends.shared import (
    ceil_div,
    contiguous_f32,
    dynamo_constant,
    refuse_double_backward,
    register_op,
    tracks_grad,
)

from .layout import FFT_POINTS_PER_THREAD
from .runtime import (
    blocks_per_row,
    launcher,
    lazy,
    on_device,
    persistent,
    registers,
    smem,
    unused,
)
from .transpose import features_last, sequences_last, transpose


@persistent
def twiddles(NF: int, device: torch.device) -> Tensor:
    from s4ever._backends.tilelang.kernels import fft as f

    tw = torch.tensor(f.layout(NF).twiddles(), dtype=torch.complex64, device=device)
    return torch.view_as_real(tw)


@persistent
def frequencies(NF: int, device: torch.device) -> Tensor:
    from s4ever._backends.tilelang.kernels import fft as f

    return torch.tensor(f.layout(NF).frequencies(), dtype=torch.int32, device=device)


@persistent
def split_twiddles(NF: int, N1: int, device: torch.device) -> Tensor:
    from s4ever._backends.tilelang.kernels import fft as f

    if N1 == 1:
        return unused(device, (1,), torch.uint64)
    tw = torch.tensor(f.split_twiddles(NF, N1), dtype=torch.complex64, device=device)
    return torch.view_as_real(tw).view(torch.uint64)[:, 0]


def pairs_per_block(pairs: int, H: int, device: torch.device, G: int, N1: int) -> int:
    """Return the rounds per thread block, so that H rows get about 8 N1 blocks per SM.

    For N1 > 2, each block runs one round. With more rounds, a block that combines the outputs of
    a round starts its next round later than the other blocks, and the shares of the other blocks
    stay in L2 until the combining block reads them.
    """
    if N1 > 2:
        return 1
    return max(1, pairs // (G * blocks_per_row(H, device)))


@dynamo_constant
def block_fft(device: torch.device) -> int:
    """Return the most points that one thread block transforms."""
    return cached_block_fft(device)


@functools.cache
def cached_block_fft(device: torch.device) -> int:
    from s4ever._backends.tilelang.kernels import fft as f

    return 8192 if f.shared_bytes(8192) <= smem(device) else 4096


def max_length(device: torch.device) -> int:
    """Return the longest sequence length that the fused convolutions support.

    At this length, each transform splits over `block_fft` thread blocks.
    """
    return block_fft(device) ** 2 // 2


def fft_config(L: int, device: torch.device) -> tuple[int, int]:
    """Return NF and N1 for sequences of length L.

    A transform has N1 NF >= 2 L points and is split over N1 thread blocks, each of which
    transforms NF points.
    """
    # NT is 2 L rounded up to a power of two, and at least 2 FFT_POINTS_PER_THREAD.
    NT = max(2 * FFT_POINTS_PER_THREAD, 2 << max(L - 1, 0).bit_length())
    N1 = max(1, NT // block_fft(device))
    return NT // N1, N1


def kernel_n1(N1: int) -> int | None:
    """Return the N1 to build `kernels.fftconv`'s kernels for, or None for three passes.

    The three-pass kernels, which run for N1 > MAX_SPLIT, read N1 from the shapes.
    """
    from s4ever._backends.tilelang.kernels import fft as f

    return N1 if N1 <= f.MAX_SPLIT else None


def shares_left(N1: int) -> bool:
    """Return whether the N1 thread blocks of a transform leave their output shares in memory.

    The blocks leave their shares for three passes, and for N1 > 1 when `kernels.fftconv.COMBINE`
    is off. A launch from `combiner` then combines the shares.
    """
    from s4ever._backends.tilelang.kernels import fft as f
    from s4ever._backends.tilelang.kernels import fftconv as conv

    return N1 > f.MAX_SPLIT or N1 > 1 and not conv.COMBINE


def buffers(u: Tensor, C: int) -> tuple[Tensor, Tensor]:
    """Allocate y and the spectra Kf of the convolution kernels, in the layout of the transforms."""
    Bsz, H, L = u.shape
    NF, N1 = fft_config(L, u.device)
    Kf = torch.empty(
        C,
        H,
        N1,
        FFT_POINTS_PER_THREAD,
        NF // FFT_POINTS_PER_THREAD,
        dtype=torch.uint64,
        device=u.device,
    )
    return u.new_empty(Bsz, C, H, L), Kf


def columns(N1: int, NF: int, dtype, device: torch.device):
    """Return a function that runs the first of three passes on x (B, R, L).

    The pass computes the N1-point transforms of the columns of each pair x[2 p] + i x[2 p + 1].
    """
    from s4ever._backends.tilelang.kernels import fftconv as conv

    kernel = lazy(conv.columns_kernel, N1, NF, "pairs", dtype)
    tables = twiddles(N1, device), frequencies(N1, device)

    def run(x):
        c = torch.empty(
            ceil_div(x.shape[0], 2), x.shape[1], N1, NF, dtype=torch.uint64, device=device
        )
        kernel(x, *tables, c)
        return c

    return run


def combiner(N1: int, NF: int, kind: str, dtype, addend: bool, device: torch.device):
    """Return a function that combines the shares that the N1 thread blocks of each transform left.

    For kind "pairs", the function writes y (B, R, L) from the shares w (P, R, N1, NF), and with
    addend it adds ad[c, b, h, t] to y[b, c H + h, t]. For kind "taps", it writes the taps'
    gradients from the shares w (R, N1, NF) of `kernels.fftconv.spectrum_grad_kernel`. Up to
    MAX_SPLIT blocks, `kernels.fftconv.combine_kernel` combines the shares, and for three passes,
    `kernels.fftconv.columns_inverse_kernel` runs the last pass.
    """
    from s4ever._backends.tilelang.kernels import fft as f
    from s4ever._backends.tilelang.kernels import fftconv as conv

    if N1 <= f.MAX_SPLIT:
        return lazy(conv.combine_kernel, N1, NF, kind, dtype, addend)
    kernel = lazy(conv.columns_inverse_kernel, N1, NF, kind, dtype, addend)
    tables = twiddles(N1, device), frequencies(N1, device)
    return lambda w, *outputs: kernel(w, *tables, *outputs)


def convolve(u: Tensor, Kf: Tensor, y: Tensor, ad: Tensor | None = None) -> None:
    """Write into y the convolution of u with the kernels whose spectra are Kf.

    With ad, y[b, c, h, t] += ad[c, b, h, t].
    """
    convolver(u.device, *y.shape, u.dtype, ad is not None)(u, Kf, y, ad)


@launcher
def convolver(device, Bsz, C, H, L, dtype, addend):
    from s4ever._backends.tilelang.kernels import fft as f
    from s4ever._backends.tilelang.kernels import fftconv as conv
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    NF, N1 = fft_config(L, device)
    tw, T1 = twiddles(NF, device), split_twiddles(NF, N1, device)
    pairs = ceil_div(Bsz, 2)
    rounds = pairs_per_block(pairs, H, device, f.layout(NF).G, N1)
    three_passes, leaves_shares = N1 > f.MAX_SPLIT, shares_left(N1)
    tl_dtype = DTYPES[dtype]
    kernel = lazy(
        conv.forward_kernel,
        NF,
        C,
        tl_dtype,
        kernel_n1(N1),
        addend and not leaves_shares,
        conv.COMBINE,
    )
    if three_passes:
        transform = columns(N1, NF, tl_dtype, device)
    if leaves_shares:
        combine = combiner(N1, NF, "pairs", tl_dtype, addend, device)
    no_ad = unused(device, (1, 1, 1, 1))
    # Without a split, every call can share the stand-ins.
    unsplit = combine_buffers(pairs, rounds, C, H, N1, NF, device) if N1 == 1 else None

    def launch(u, Kf, y, ad):
        w, cnt = unsplit or combine_buffers(pairs, rounds, C, H, N1, NF, device)
        src = transform(u) if three_passes else u
        ad = no_ad if ad is None else ad
        kernel(src, Kf, tw, T1, y, no_ad if leaves_shares else ad, w, cnt, rounds)
        if leaves_shares:
            combine(w[:pairs].flatten(1, 2), y.flatten(1, 2), ad)

    return launch


def conv_outputs(u, k, k_rev, D, ad, want_gu, want_gK):
    return buffers(u, (k if k is not None else k_rev).shape[0])


@register_op("fftconv", conv_outputs)
def fftconv_op(
    u: Tensor,
    k: Tensor | None,
    k_rev: Tensor | None,
    D: Tensor | None,
    ad: Tensor | None,
    want_gu: bool,
    want_gK: bool,
) -> tuple[Tensor, Tensor]:
    """Return y and the spectra Kf of the convolution kernels.

    `want_gu` and `want_gK` say whether a backward will compute the gradients of u and of the
    convolution kernels, so that the GPU kernels of that backward start compiling here.
    """
    y, Kf = conv_outputs(u, k, k_rev, D, ad, want_gu, want_gK)
    has = k is not None, k_rev is not None, D is not None
    spectrum(u.device, *y.shape, u.dtype, has, ad is not None, want_gu, want_gK)(k, k_rev, D, Kf)
    convolve(u, Kf, y, ad)
    return y, Kf


@launcher
def spectrum(device, Bsz, C, H, L, dtype, has, addend, want_gu, want_gK):
    """Return a function that writes the spectra Kf for `fftconv_op`.

    Building it also builds the other launchers of `fftconv_op` and those of the gradients that
    `fftconv_backward_op` will compute, so that all their GPU kernels compile in parallel.
    """
    from s4ever._backends.tilelang.kernels import fft as f
    from s4ever._backends.tilelang.kernels import fftconv as conv

    # The backward kernels start compiling first, because they take the longest to compile.
    if want_gu or want_gK:
        backward_runner(device, Bsz, C, H, L, dtype, want_gu, want_gK)
    if want_gK:
        spectrum_grad(device, C, H, L, has)
    convolver(device, Bsz, C, H, L, dtype, addend)
    NF, N1 = fft_config(L, device)
    tw, T1 = twiddles(NF, device), split_twiddles(NF, N1, device)
    flags = [int(h) for h in has]
    absent = unused(device, (1, 1)), unused(device, (1, 1)), unused(device, (1,))
    kernel = lazy(conv.spectrum_kernel, NF, kernel_n1(N1))
    three_passes = N1 > f.MAX_SPLIT
    if three_passes:
        circular = lazy(conv.columns_kernel, N1, NF, "circular", None)
        tables = twiddles(N1, device), frequencies(N1, device)

    def launch(k, k_rev, D, Kf):
        k, k_rev, D = (
            a if x is None else x.flatten(0, 1) for x, a in zip((k, k_rev, D), absent, strict=True)
        )
        if three_passes:
            c = torch.empty(C * H, N1, NF, dtype=torch.uint64, device=device)
            circular(k, k_rev, *tables, c, *flags[:2])
            k = c
        kernel(k, k_rev, D, tw, T1, Kf.view(C * H, N1, FFT_POINTS_PER_THREAD, -1), *flags)

    return launch


def takes_spectra(L: int, Lk: int, device: torch.device) -> bool:
    """Return whether `spectra_conv_op` accepts the spectra of length-Lk kernels at length L."""
    NF, N1 = fft_config(L, device)
    return N1 == 1 and 2 <= Lk <= NF and Lk & (Lk - 1) == 0


def spectra_outputs(u, k_f, D, bidirectional, want_gu, want_gK):
    return buffers(u, k_f.shape[0] // (1 + bidirectional))


@register_op("spectra_conv", spectra_outputs)
def spectra_conv_op(
    u: Tensor, k_f: Tensor, D: Tensor | None, bidirectional: bool, want_gu: bool, want_gK: bool
) -> tuple[Tensor, Tensor]:
    """Compute `fftconv_op` from the half spectra k_f (C, H, Lk / 2 + 1, 2) of the kernels.

    The spectra are scaled by 1 / Lk, as `kernels.cauchy` writes them. With `bidirectional`, k_f has
    2 C channels, and the last C are the spectra of k_rev.
    """
    y, Kf = spectra_outputs(u, k_f, D, bidirectional, want_gu, want_gK)
    args = (u.dtype, D is not None, bidirectional, want_gu, want_gK)
    spectra(u.device, *y.shape, k_f.shape[2], *args)(k_f, D, Kf)
    convolve(u, Kf, y)
    return y, Kf


@launcher
def spectra(device, Bsz, C, H, L, bins, dtype, has_D, bidirectional, want_gu, want_gK):
    """Return a function that writes the spectra Kf for `spectra_conv_op`, as `spectrum` does."""
    from s4ever._backends.tilelang.kernels import fftconv as conv

    Lk = 2 * bins - 2
    if want_gu or want_gK:
        backward_runner(device, Bsz, C, H, L, dtype, want_gu, want_gK)
    if want_gK:
        spectra_grad(device, C, H, L, Lk, bidirectional, has_D)
    convolver(device, Bsz, C, H, L, dtype, False)
    NF = fft_config(L, device)[0]
    kernel = lazy(conv.spectra_kernel, NF, NF // Lk, bidirectional)
    tw, freqs, no_D = twiddles(NF, device), frequencies(NF, device), unused(device, (1,))

    def launch(k_f, D, Kf):
        rows = k_f.flatten(0, 1)
        kf, kfr = (rows[: C * H], rows[C * H :]) if bidirectional else (rows, rows)
        D = no_D if D is None else D.flatten()
        kernel(kf, kfr, D, tw, freqs, Kf.view(C * H, 1, FFT_POINTS_PER_THREAD, -1), L, int(has_D))

    return launch


def conv_backward(
    u: Tensor, gy: Tensor, Kf: Tensor, gu: Tensor | None, want_gK: bool
) -> Tensor | None:
    """Write the gradient of u into gu unless gu is None, and return gK for the gradient of Kf.

    With `kernels.fftconv.COMBINE`, the gradient of Kf is gK[0], and without it, the gradient is
    gK.sum(0). Without want_gK, the gradient of Kf is not computed, and the result is None.
    """
    Bsz, H, L = u.shape
    runner = backward_runner(u.device, Bsz, Kf.shape[0], H, L, u.dtype, gu is not None, want_gK)
    # Without gu, the kernels do not write the gu argument, and u stands in for it.
    return runner(u, gy, Kf, u if gu is None else gu)


@launcher
def backward_runner(device, Bsz, C, H, L, dtype, want_gu, want_gK):
    from s4ever._backends.tilelang.kernels import fft as f
    from s4ever._backends.tilelang.kernels import fftconv as conv
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    NF, N1 = fft_config(L, device)
    lay = f.layout(NF)
    tw, T1 = twiddles(NF, device), split_twiddles(NF, N1, device)
    pairs = ceil_div(Bsz, 2)
    three_passes, leaves_shares = N1 > f.MAX_SPLIT, shares_left(N1)
    tl_dtype = DTYPES[dtype]
    built_n1 = kernel_n1(N1)
    gu_rounds = pairs_per_block(pairs, H, device, lay.G, N1)
    gu_kernel = gK_kernel = None
    # One kernel computes both gradients when each thread of a block can have 256 registers. In a
    # larger block, each thread has registers for only one gradient.
    fused = want_gu and want_gK and lay.threads <= registers(device) // 256
    # A kernel that computes only gK does not combine shares, so its rounds are planned as for N1 H
    # rows of unsplit transforms.
    rounds = gu_rounds if fused else pairs_per_block(pairs, N1 * H, device, lay.G, 1)
    if fused:
        gK_kernel = lazy(
            conv.backward_kernel, NF, C, tl_dtype, C == 1, "gu gK", built_n1, conv.COMBINE
        )
    elif want_gK:
        # With one round, sums in registers would only add register pressure.
        acc = C == 1 and rounds > 1
        gK_kernel = lazy(conv.backward_kernel, NF, C, tl_dtype, acc, "gK", built_n1, conv.COMBINE)
    if want_gu and not fused:
        gu_kernel = lazy(conv.backward_kernel, NF, C, tl_dtype, False, "gu", built_n1, conv.COMBINE)
    if three_passes:
        transform = columns(N1, NF, tl_dtype, device)
    if leaves_shares and want_gu:
        combine = combiner(N1, NF, "pairs", tl_dtype, False, device)
    no_ad = unused(device, (1, 1, 1, 1))
    no_cntK = unused(device, (1, N1), torch.int32)
    no_cnt = unused(device, (1, 1, 1, 2), torch.int32)
    S = ceil_div(pairs, lay.G * rounds) * lay.G  # groups of pairs, each with its own sum in gK
    # With COMBINE and more than one sum, the kernels count the finished blocks in cntK from zero.
    zero_cntK = S > 1 and conv.COMBINE
    unsplit = combine_buffers(pairs, rounds, 1, H, N1, NF, device) if N1 == 1 else None

    def unread(*shape):
        """Allocate a stand-in for an argument that the launched kernel does not read.

        Unlike the stand-ins from `unused`, it is freed after the launch, because its size grows
        with the length of the transform.
        """
        return torch.empty(shape, dtype=torch.uint64, device=device)

    def backward(u, gy, Kf, gu):
        if three_passes:
            gy_c = transform(gy.flatten(1, 2)).view(pairs, C, H, N1, NF)
            ins = (transform(u), gy_c, Kf, tw, T1, gu)
        else:
            ins = (u, gy, Kf, tw, T1, gu)
        if gu_kernel is not None:
            scratch = unsplit or combine_buffers(pairs, gu_rounds, 1, H, N1, NF, device)
            gu_kernel(
                *ins,
                unread(1, C, 1, N1, FFT_POINTS_PER_THREAD, NF // FFT_POINTS_PER_THREAD),
                no_cntK,
                *scratch,
                gu_rounds,
            )
            if leaves_shares:
                combine(scratch[0][:pairs, 0], gu, no_ad)
            del scratch  # so that the allocation of gK can reuse the memory
        if gK_kernel is None:
            return None
        gK = torch.empty(S, *Kf.shape, dtype=torch.uint64, device=device)
        cntK = (torch.zeros if zero_cntK else torch.empty)(H, N1, dtype=torch.int32, device=device)
        if fused:
            scratch = unsplit or combine_buffers(pairs, rounds, 1, H, N1, NF, device)
            gK_kernel(*ins, gK, cntK, *scratch, rounds)
            if leaves_shares:
                combine(scratch[0][:pairs, 0], gu, no_ad)
        else:
            gK_kernel(*ins, gK, cntK, unread(1, 1, 1, N1, NF), no_cnt, rounds)
        return gK

    return backward


def conv_grads(u, gy, Kf, Lk, Lk_rev, has_D, want_gu, want_gK):
    C, H = Kf.shape[:2]
    shapes = [(C, H, n) if n is not None else 0 for n in (Lk, Lk_rev)] + [(C, H) if has_D else 0]
    gu = torch.empty_like(u) if want_gu else u.new_empty(0)
    return [gu, *(u.new_empty(s if want_gK else 0, dtype=torch.float32) for s in shapes)]


@register_op("fftconv_backward", conv_grads)
def fftconv_backward_op(
    u: Tensor,
    gy: Tensor,
    Kf: Tensor,
    Lk: int | None,
    Lk_rev: int | None,
    has_D: bool,
    want_gu: bool,
    want_gK: bool,
) -> list[Tensor]:
    """Return the gradients of u, k, k_rev and D.

    A gradient that is not wanted, or whose input is absent, is an empty tensor.
    """
    grads = conv_grads(u, gy, Kf, Lk, Lk_rev, has_D, want_gu, want_gK)
    gK = conv_backward(u, gy, Kf, grads[0] if want_gu else None, want_gK)
    if want_gK:
        has = Lk is not None, Lk_rev is not None, has_D
        spectrum_grad(u.device, *gy.shape[1:], has)(gK, *grads[1:])
    return grads


@launcher
def spectrum_grad(device, C, H, L, has):
    from s4ever._backends.tilelang.kernels import fftconv as conv

    NF, N1 = fft_config(L, device)
    tw, T1 = twiddles(NF, device), split_twiddles(NF, N1, device)
    flags = [int(h) for h in has]
    absent = unused(device, (1, 1)), unused(device, (1, 1)), unused(device, (1,))
    kernel = lazy(conv.spectrum_grad_kernel, NF, kernel_n1(N1), conv.COMBINE)
    leaves_shares = shares_left(N1)
    if leaves_shares:
        combine = combiner(N1, NF, "taps", None, False, device)

    def scratch():
        w, cnt = combine_buffers(1, 1, C, H, N1, NF, device)
        return w.view(-1, N1, NF), cnt.view(-1, 2)

    unsplit = scratch() if N1 == 1 else None

    def launch(gK, gk, gk_rev, gD):
        taps = tuple(
            g.flatten(0, 1) if h else a
            for g, h, a in zip((gk, gk_rev, gD), has, absent, strict=True)
        )
        w, cnt = unsplit or scratch()
        kernel(
            gK.view(gK.shape[0], C * H, N1, FFT_POINTS_PER_THREAD, -1),
            tw,
            T1,
            *taps,
            w,
            cnt,
            *flags,
        )
        if leaves_shares:
            combine(w, *taps, *flags)

    return launch


def spectra_grads(u, gy, Kf, Lk, bidirectional, has_D, want_gu, want_gK):
    C, H = Kf.shape[:2]
    shapes = [(C * (1 + bidirectional), H, Lk // 2 + 1, 2), (C, H) if has_D else 0]
    gu = torch.empty_like(u) if want_gu else u.new_empty(0)
    return [gu, *(u.new_empty(s if want_gK else 0, dtype=torch.float32) for s in shapes)]


@register_op("spectra_conv_backward", spectra_grads)
def spectra_conv_backward_op(
    u: Tensor,
    gy: Tensor,
    Kf: Tensor,
    Lk: int,
    bidirectional: bool,
    has_D: bool,
    want_gu: bool,
    want_gK: bool,
) -> list[Tensor]:
    """Return the gradients of u, k_f and D.

    A gradient that is not wanted, or whose input is absent, is an empty tensor.
    """
    grads = spectra_grads(u, gy, Kf, Lk, bidirectional, has_D, want_gu, want_gK)
    gu, g_f, gD = grads
    gK = conv_backward(u, gy, Kf, gu if want_gu else None, want_gK)
    if want_gK:
        spectra_grad(u.device, *gy.shape[1:], Lk, bidirectional, has_D)(gK, g_f, gD)
    return grads


@launcher
def spectra_grad(device, C, H, L, Lk, bidirectional, has_D):
    from s4ever._backends.tilelang.kernels import fftconv as conv

    NF = fft_config(L, device)[0]
    kernel = lazy(conv.spectra_grad_kernel, NF, NF // Lk, bidirectional, conv.COMBINE)
    tw, freqs, no_D = twiddles(NF, device), frequencies(NF, device), unused(device, (1,))

    def launch(gK, g_f, gD):
        rows = g_f.flatten(0, 1)
        gkf, gkfr = (rows[: C * H], rows[C * H :]) if bidirectional else (rows, rows)
        gD = gD.flatten() if has_D else no_D
        kernel(
            gK.view(gK.shape[0], C * H, 1, FFT_POINTS_PER_THREAD, -1),
            tw,
            freqs,
            gkf,
            gkfr,
            gD,
            L,
            int(has_D),
        )

    return launch


class FFTConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k, k_rev, D, ad, want_gu, want_gK):
        u = transpose(u)
        y, Kf = fftconv_op(u, k, k_rev, D, ad, want_gu, want_gK)
        ctx.save_for_backward(u, Kf)
        ctx.args = [None if x is None else x.shape[-1] for x in (k, k_rev)] + [D is not None]
        ctx.pad = None if ad is None else ad.shape[-1] - u.shape[-1]
        return features_last(y)

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        u, Kf = ctx.saved_tensors
        gy = sequences_last(grads[0])
        want_gu, want_gK = ctx.needs_input_grad[0], any(ctx.needs_input_grad[1:4])
        gu, *grads = fftconv_backward_op(u, gy, Kf, *ctx.args, want_gu, want_gK)
        has = [n is not None for n in ctx.args[:2]] + ctx.args[2:]
        gad = None
        if ctx.needs_input_grad[4]:
            gad = gy.transpose(0, 1)
            if ctx.pad:
                gad = F.pad(gad, (0, ctx.pad))
        grads = [g if h and want_gK else None for g, h in zip(grads, has, strict=True)]
        return transpose(gu) if want_gu else None, *grads, gad, None, None


def combine_buffers(
    pairs: int, rounds: int, C: int, H: int, N1: int, NF: int, device: torch.device
):
    """Allocate the shares w and the counters cnt with which the N1 blocks of a transform combine.

    For N1 == 1, the blocks do not combine, and the buffers are stand-ins.
    """
    from s4ever._backends.tilelang.kernels import fftconv as conv

    if N1 == 1:
        return (
            unused(device, (1, C, 1, 1, NF), torch.uint64),
            unused(device, (1, C, 1, 2), torch.int32),
        )
    P = ceil_div(pairs, rounds) * rounds  # pairs rounded up to a multiple of rounds
    w = torch.empty(P, C, H, N1, NF, dtype=torch.uint64, device=device)
    # With COMBINE, the kernels count the finished blocks of each transform in cnt from zero.
    alloc = torch.zeros if conv.COMBINE else torch.empty
    return w, alloc(P, C, H, 2, dtype=torch.int32, device=device)


@on_device
def fftconv(
    u: Tensor, k: Tensor | None, D: Tensor | None, k_rev: Tensor | None, ad: Tensor | None
) -> Tensor:
    """Compute `s4ever.ops.fftconv` for float32 convolution kernels k and k_rev (C, H, <= L).

    With ad (C, B, H, >= L), y[b, t, c, h] += ad[c, b, h, t].
    """
    k, k_rev, ad = contiguous_f32(k, k_rev, ad)
    if D is not None:
        # pyrefly: ignore[missing-attribute]  ops.fftconv passes k or k_rev, or both
        D = torch.broadcast_to(D.float(), (k if k is not None else k_rev).shape[:2]).contiguous()
    return FFTConv.apply(u, k, k_rev, D, ad, tracks_grad(u), tracks_grad(k, k_rev, D))
