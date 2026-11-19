"""FFT convolution of real sequences with per-feature kernels on Metal.

Two batch rows u_a and u_b form one complex sequence z = u_a + i u_b. Because the two rows have
the same real kernel, the inverse transform of Z K is y_a + i y_b. The forward stages leave the
frequencies at digit-reversed positions, and the spectra of the convolution kernels are stored in
the same order. The inverse stages start from this order, so the transforms never reorder values.

A transform of n = N1 NF > NF_MAX points uses the four-step FFT, in which the threadgroups of
slice k1 transform Z_k1 into the frequencies k1 + N1 k2 of z:

    w = exp(-2 pi i / n)
    Z_k1[m] = sum_n1 z[m + NF n1] w^((m + NF n1) k1)    for m < NF

For N1 <= SPLIT, each threadgroup sums the N1 segments z[m + NF n1] of its rows itself, so the
rows are read once per slice. The inverse transform of each slice is then a share of the outputs,
and a second dispatch (`combine_pairs`) sums the shares. The convolution at N1 = 2 does not need a
second dispatch, because one threadgroup computes both slices (`conv_halves`). For N1 > SPLIT,
Z_k1 are the twiddled N1-point transforms of the columns z[m + NF n1]. A column dispatch computes
these transforms before the row kernels, and another computes the inverse transforms after them.
"""

import cmath
import functools
import math

import torch
from torch import Tensor

from s4ever._backends import shared
from s4ever._backends.shared import ceil_div

from . import msl
from .transpose import transpose, transposes

E = 8  # points per thread
# A threadgroup transforms at most NF_MAX points, because an exchange buffer of NF_MAX float2
# values fills the 32 KB of threadgroup memory that Metal allows.
NF_MAX = 4096
# The transforms have at least 2 L points, and at most NF_MAX**2 points, because the column
# transforms have at most NF_MAX points.
MAX_LENGTH = NF_MAX**2 // 2
SPLIT = 4  # the largest N1 at which the row kernels form the slices from the samples
GROUP = 256  # threads per threadgroup of `combine_pairs`, and the target for the row kernels
COLUMN_GROUP = 512  # the target for the threads per threadgroup of the column kernels

ROW_NAMES = (
    "spectrum",
    "conv",
    "conv_halves",
    "conv_grads",
    "spectrum_grads",
    "spectrum_grads_whole",
    "combine_pairs",
    "real_inverse",
)
COLUMN_NAMES = ("columns_pairs", "columns_inverse_pairs")


def dft(points: int) -> str:
    """Return the MSL specialization of `dft` for a DFT of `points` points.

    The DFT runs in registers and leaves its outputs in natural order. It uses radix-2 decimation
    in frequency and then the bit-reversal permutation.
    """
    lines = [f"template <> inline void dft<{points}>(thread float2* v) {{"]
    s = points // 2
    while s >= 1:
        for i in range(points):
            if i & s == 0:
                k = (i % s) * (points // (2 * s))
                root = cmath.exp(-2j * math.pi * k / points)
                d = f"cmul(a - b, float2({root.real!r}f, {root.imag!r}f))"
                if k == 0:
                    d = "a - b"
                elif 4 * k == points:  # times -i
                    d = "float2(a.y - b.y, b.x - a.x)"
                a, b = f"v[{i}]", f"v[{i + s}]"
                lines.append(f"    {{ float2 a = {a}, b = {b}; {a} = a + b; {b} = {d}; }}")
        s //= 2
    bits = points.bit_length() - 1
    for i in range(points):
        r = int(f"{i:0{bits}b}"[::-1], 2) if bits else 0
        if i < r:
            lines.append(f"    {{ float2 t = v[{i}]; v[{i}] = v[{r}]; v[{r}] = t; }}")
    return "\n".join([*lines, "}"])


def radix_stages(NF: int) -> list[tuple[int, int]]:
    """Return the stages (radix, M) of an NF-point transform.

    Stage (radix, M) computes DFTs of `radix` points at stride M / radix. A radix smaller than E
    comes first.
    """
    radices, m = [], NF
    while m >= E:
        radices.append(E)
        m //= E
    radices = ([m] if m > 1 else []) + radices
    stages, M = [], NF
    for radix in radices:
        stages.append((radix, M))
        M //= radix
    return stages


def source(NF: int, C: int, dtype: torch.dtype, columns: bool) -> tuple[str, int, int]:
    """Return the MSL source, the threadgroup width and the number of transforms per threadgroup.

    A threadgroup runs as many transforms as fit in its target width and in NF_MAX points.
    """
    stages = radix_stages(NF)
    radices = [radix for radix, _ in stages]
    e = min(E, NF)
    fft_threads = NF // e
    group_ffts = max(1, min((COLUMN_GROUP if columns else GROUP) // fft_threads, NF_MAX // NF))

    def move(a, b):
        return f"exchange<{a[0]}, {a[1]}, {b[0]}, {b[1]}>(v, buf, off, t);"

    fwd, inv = [], []
    for i, (radix, M) in enumerate(stages):
        fwd += [move(stages[i - 1], (radix, M))] if i else []
        fwd.append(f"forward<{radix}, {M}>(v, t, tw);")
    for i in reversed(range(len(stages))):
        inv += [move(stages[i + 1], stages[i])] if i < len(stages) - 1 else []
        inv.append(f"inverse<{stages[i][0]}, {stages[i][1]}>(v, t, tw);")
    # The forward stages leave the frequency k at the position p, where the digit of stage s is
    # e_s < radix_s:
    #
    #     p = sum_s e_s M_s / radix_s
    #     k = sum_s e_s radix_0 ... radix_{s - 1}
    digits, place = [], 1
    for radix, M in stages:
        digits.append(f"k += p / {M // radix} * {place}; p %= {M // radix};")
        place *= radix
    slice_counts = [2**i for i in range(1, SPLIT.bit_length())]
    cases = " ".join(f"case {count}: idft<{count}>(u); break;" for count in slice_counts)
    prelude = f"""
constant constexpr uint NF = {NF}, E = {e}, C = {C};
constant constexpr uint FFT_THREADS = {fft_threads}, GROUP_FFTS = {group_ffts};
constant constexpr uint NF_MAX = {NF_MAX}, SPLIT = {SPLIT};
constant constexpr uint FIRST_RADIX = {radices[0]}, LAST_RADIX = {radices[-1]};
typedef {msl.TYPES[dtype]} IO;
inline uint rev(uint p) {{ uint k = 0; {" ".join(digits)} return k; }}
template <uint R> inline void dft(thread float2* v);
"""
    dfts = "\n".join(dft(points) for points in sorted({*radices, *slice_counts}))
    args = "thread float2* v, threadgroup float2* buf, uint off, uint t, device const float2* tw"
    stage_functions = f"""
inline void forward_stages({args}) {{ {" ".join(fwd)} }}
inline void inverse_stages({args}) {{ {" ".join(inv)} }}
inline void inverse_dft(thread float2* u, uint N1) {{ switch (N1) {{ {cases} }} }}
"""
    kernels = "fft_columns.metal" if columns else "fft_rows.metal"
    src = msl.source(prelude + dfts, "fft.metal", stage_functions, kernels)
    return src, fft_threads * group_ffts, group_ffts


@functools.cache
def library(NF: int, C: int, dtype: torch.dtype, columns: bool = False):
    src, width, group_ffts = source(NF, C, dtype, columns)
    names = COLUMN_NAMES if columns else ROW_NAMES
    threads = {name: GROUP if name.startswith("combine") else width for name in names}
    return msl.library(src, threads), width, group_ffts


@functools.cache
def roots(n: int, count: int) -> Tensor:
    """Return w^j = exp(-2 pi i j / n) for j < count as (count, 2) on MPS."""
    j = torch.arange(count, dtype=torch.float64)
    return (
        torch.view_as_real(torch.polar(torch.ones_like(j), -2 * math.pi / n * j)).float().to("mps")
    )


def split(n: int) -> tuple[int, int]:
    NF = min(n, NF_MAX)
    return NF, n // NF


def twiddles(n: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return the twiddles of the NF-point transforms, and the coarse and fine roots of n points.

    With w = exp(-2 pi i / n), the coarse roots are w^(NF i) for i < N1 and the fine roots are w^i
    for i < NF, so that w^j = coarse[j // NF] fine[j % NF].
    """
    NF, N1 = split(n)
    return roots(NF, NF), roots(N1, N1), roots(n, NF)


def fits(n: int, dtype: torch.dtype) -> bool:
    """Return whether the kernels support transforms of n points for `dtype` inputs.

    The kernels transform powers of two only. They must build for float32 as well as for dtype,
    because they compute the spectra of the convolution kernels in float32.
    """
    if n & (n - 1):
        return False
    NF, N1 = split(n)
    return N1 <= NF_MAX and builds(NF, N1, torch.float32) and builds(NF, N1, dtype)


@msl.constant
def builds(NF: int, N1: int, dtype: torch.dtype) -> bool:
    """Return whether the kernels build for the most channels, and warn if they do not.

    The kernels use the most registers at this channel count.
    """

    def build():
        library(NF, shared.MAX_CHANNELS, dtype)
        transposes(dtype)
        if N1 > SPLIT:
            library(N1, 1, dtype, True)

    return msl.builds(f"the FFT convolution kernels for {dtype}", build)


def launch_columns(
    name: str, out: Tensor, src: Tensor, *args, n: int, rows: int, dtype: torch.dtype
):
    NF, N1 = split(n)
    lib, width, group_ffts = library(N1, 1, dtype, True)
    _, coarse, fine = twiddles(n)
    threads = [NF // group_ffts * rows * width, 1, 1]
    getattr(lib, name)(out, src, coarse, fine, *args, threads=threads, group_size=[width, 1, 1])


def shares(like: Tensor, n: int, rows: int) -> Tensor:
    """Allocate the partial outputs (shares) of the threadgroups as (rows, n, 2).

    For n <= NF_MAX, the kernels write the outputs directly, and the result is a one-element
    stand-in.
    """
    return like.new_empty(rows, n, 2, dtype=torch.float32) if n > NF_MAX else like.new_empty(1)


def pairs(x: Tensor, R: int, n: int) -> Tensor:
    """Return the column transforms of the row pairs of x (B, R, H, L).

    For N1 <= SPLIT, the row kernels read the samples instead, and the result is a one-element
    stand-in.
    """
    if split(n)[1] <= SPLIT:
        return x.new_empty(1)
    B, H, L = x.shape[0], x.shape[-2], x.shape[-1]
    rows = (B + 1) // 2 * R * H
    A = x.new_empty(rows, n, 2, dtype=torch.float32)
    launch_columns("columns_pairs", A, x, B, R, H, L, n=n, rows=rows, dtype=x.dtype)
    return A


def combine(lib, y: Tensor, U: Tensor, R: int, n: int, ad: Tensor | None):
    """Write y (B, R, H, L) from the shares U of its row pairs, and add ad if it is given."""
    B, H, L = y.shape[0], y.shape[-2], y.shape[-1]
    NF, N1 = split(n)
    rows = (B + 1) // 2 * R * H
    add = (y if ad is None else ad, int(ad is not None))
    if N1 > SPLIT:
        args = (y, U, add[0], B, R, H, L, add[1])
        launch_columns("columns_inverse_pairs", *args, n=n, rows=rows, dtype=y.dtype)
    elif N1 > 1:
        args = (y, add[0], U, B, R, H, L, N1, add[1])
        lib.combine_pairs(*args, threads=[NF, rows, 1], group_size=[GROUP, 1, 1])


@shared.register_op("metal_spectrum", lambda k: k.new_empty(*k.shape, 2), "MPS")
def spectrum_op(k: Tensor) -> Tensor:
    """Return the spectra / n (C, H, n, 2) of the real kernels k (C, H, n).

    The frequencies are in the order in which the forward transforms leave them.
    """
    C, H, n = k.shape
    NF, N1 = split(n)
    lib, width, group_ffts = library(NF, C, torch.float32)
    Kf = k.new_empty(C, H, n, 2)
    rows = C * H
    row_pairs = (rows + 1) // 2
    threads = [(N1 // 2 + 1) * ceil_div(row_pairs, group_ffts) * width, 1, 1]
    args = (Kf, k, pairs(k.view(rows, 1, n), 1, n), positions(N1), *twiddles(n), rows, N1)
    lib.spectrum(*args, threads=threads, group_size=[width, 1, 1])
    return Kf


@shared.register_op(
    "metal_conv", lambda x, Kf, ad: x.new_empty(x.shape[0], Kf.shape[0], *x.shape[1:]), "MPS"
)
def conv_op(x: Tensor, Kf: Tensor, ad: Tensor | None) -> Tensor:
    """Return the convolutions y (B, C, H, L) of x (B, H, L) with the kernels whose spectra are Kf.

    Kf is the output of `spectrum_op`. With ad (B, C, H, L), y += ad.
    """
    (B, H, L), (C, _, n, _) = x.shape, Kf.shape
    NF, N1 = split(n)
    lib, width, group_ffts = library(NF, C, x.dtype)
    P = (B + 1) // 2
    y = x.new_empty(B, C, H, L)
    groups = ceil_div(P * H, group_ffts)
    add = (x if ad is None else ad, int(ad is not None))
    if N1 == 2:
        args = (y, add[0], x, Kf, *twiddles(n), B, H, L, add[1])
        lib.conv_halves(*args, threads=[groups * width, 1, 1], group_size=[width, 1, 1])
        return y
    U = shares(x, n, P * C * H)
    args = (y, add[0], U, x, pairs(x, 1, n), Kf, *twiddles(n), B, H, L, N1, add[1])
    lib.conv(*args, threads=[N1 * groups * width, 1, 1], group_size=[width, 1, 1])
    combine(lib, y, U, C, n, ad)
    return y


def grads_outputs(x: Tensor, gy: Tensor, Kf: Tensor, want_gx: bool) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x) if want_gx else x.new_empty(0), Kf.new_empty(Kf.shape[:3])


@shared.register_op("metal_conv_grads", grads_outputs, "MPS")
def conv_grads_op(x: Tensor, gy: Tensor, Kf: Tensor, want_gx: bool) -> tuple[Tensor, Tensor]:
    """Return the gradients of x and of the real kernels for gy (B, C, H, L).

    Without want_gx, the gradient of x is an empty tensor.
    """
    (B, H, L), (C, _, n, _) = x.shape, Kf.shape
    NF, N1 = split(n)
    lib, width, group_ffts = library(NF, C, x.dtype)
    P, heads = (B + 1) // 2, ceil_div(H, group_ffts)
    # Each threadgroup sums the gradients of the kernels over a chunk of `chunk_pairs` row pairs,
    # so that the dispatch has about msl.GROUPS threadgroups.
    chunk_pairs = ceil_div(P, min(P, max(1, msl.GROUPS // (heads * N1))))
    chunks = ceil_div(P, chunk_pairs)
    gx, gk = grads_outputs(x, gy, Kf, want_gx)
    gK = Kf.new_empty(chunks, C, H, n, 2)
    tables = twiddles(n)
    # Without want_gx, the kernel writes neither gx nor U, and x stands in for gx.
    out, U = (gx, shares(x, n, P * H)) if want_gx else (x, x.new_empty(1))
    ins = (x, gy, pairs(x, 1, n), pairs(gy, C, n), Kf, *tables, B, H, L, N1, chunk_pairs)
    threads = [N1 * heads * chunks * width, 1, 1]
    lib.conv_grads(out, U, gK, *ins, int(want_gx), threads=threads, group_size=[width, 1, 1])
    if want_gx:
        combine(lib, gx, U, 1, n, None)
    lib = library(NF, C, torch.float32)[0]
    rows = C * H
    if N1 == 1:
        args = (gk, gK, *tables, rows, chunks)
        threads = [ceil_div(rows, group_ffts) * width, 1, 1]
        lib.spectrum_grads_whole(*args, threads=threads, group_size=[width, 1, 1])
        return gx, gk
    row_pairs = (rows + 1) // 2
    U = shares(x, n, row_pairs)
    threads = [(N1 // 2 + 1) * ceil_div(row_pairs, group_ffts) * width, 1, 1]
    args = (U, gK, positions(N1), *tables, rows, chunks, N1)
    lib.spectrum_grads(*args, threads=threads, group_size=[width, 1, 1])
    combine(lib, gk.view(rows, 1, n), U, 1, n, None)
    return gx, gk


@functools.cache
def positions(N1: int) -> Tensor:
    """Return the positions (N1,) at which the kernels store the slices of the frequencies k1 < N1.

    For N1 > SPLIT, the slice of frequency k1 is at the position p with rev(p) = k1 in the column
    transforms. Otherwise it is at position k1.
    """
    k1 = torch.arange(N1)
    if N1 > SPLIT:
        p, f, place = k1, torch.zeros_like(k1), 1
        for radix, M in radix_stages(N1):
            f += p // (M // radix) * place
            p, place = p % (M // radix), place * radix
        k1 = torch.argsort(f)
    return k1.int().to("mps")


def real_fits(n: int) -> bool:
    return n >= 2 and fits(n, torch.float32)


def real_inverse(f: Tensor, n: int) -> Tensor:
    """Return `torch.fft.irfft(f, n, norm="forward")` of the half spectra f (..., n / 2 + 1, 2).

    Callers check `real_fits` first.
    """
    NF, N1 = split(n)
    lib, width, group_ffts = library(NF, 1, torch.float32)
    rows = math.prod(f.shape[:-2])
    P = (rows + 1) // 2
    k, U = f.new_empty(rows, 1, n), shares(f, n, P)
    args = (k, U, f.contiguous(), positions(N1), *twiddles(n), rows, N1)
    threads = [N1 * ceil_div(P, group_ffts) * width, 1, 1]
    lib.real_inverse(*args, threads=threads, group_size=[width, 1, 1])
    combine(lib, k, U, 1, n, None)
    return k.view(*f.shape[:-2], n)


class FFTConv(torch.autograd.Function):
    """Convolution y (B, L, C, H) of x (B, L, H) with kernel (C, H, n), in x's dtype.

    The length n of the kernel is a power of two and at least 2 L. The kernels zero-pad x to n
    points and return the first L points of the circular convolution. With ad (B, C, H, L),
    y[b, t, c, h] += ad[b, c, h, t].
    """

    @staticmethod
    def forward(ctx, x, kernel, ad, keep):
        x, Kf = transpose(x), spectrum_op(kernel)
        if keep:
            ctx.save_for_backward(x, Kf)
        return transpose(conv_op(x, Kf, ad).flatten(1, 2)).unflatten(2, Kf.shape[:2])

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        g = grads[0]
        gx, gk = None, None
        gy = transpose(g.flatten(2)).unflatten(1, g.shape[2:])
        if ctx.saved_tensors:
            x, Kf = ctx.saved_tensors
            want_gx = ctx.needs_input_grad[0]
            gx, gk = conv_grads_op(x, gy, Kf, want_gx)
            gx = transpose(gx) if want_gx else None
        return gx, gk, gy if ctx.needs_input_grad[2] else None, None


def conv(x: Tensor, kernel: Tensor, keep: bool, ad: Tensor | None = None) -> Tensor:
    """Return `FFTConv` of x (B, L, H) as y (B, L, C, H), and add ad (B, L, C, H) if it is given."""
    ad = None if ad is None else ad.permute(0, 2, 3, 1).contiguous()
    return FFTConv.apply(x, kernel.contiguous(), ad, keep)
