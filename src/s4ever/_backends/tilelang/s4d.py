import functools

import torch
from torch import Tensor

from s4ever._backends.shared import (
    ceil_div,
    contiguous_f32,
    refuse_double_backward,
    register_op,
    tracks_grad,
)

from .layout import CHUNK_COLUMN_STRIDE, CHUNK_ROWS
from .runtime import launcher, lazy, on_device, or_unused, precision, smem, unused
from .transpose import features_last, sequences_last, transpose

BLOCK_MODES = 32  # the most modes per thread block, so that its chunk matrices stay in registers
CHUNK = 32  # steps per chunk


def blocks(N: int) -> tuple[int, int]:
    """Return the modes per thread block of the chunk scans and the padded mode count.

    The padded count is 8, 16 or a multiple of BLOCK_MODES, because these sizes split evenly over
    the kernels' warps. It is at least CHUNK_COLUMN_STRIDE, because `kernels.chunk` stores the
    states in runs of that many real parts, each followed by as many imaginary parts.
    """
    Np = max(
        8 if N <= 8 else 16 if N <= 16 else ceil_div(N, BLOCK_MODES) * BLOCK_MODES,
        CHUNK_COLUMN_STRIDE,
    )
    return min(Np, BLOCK_MODES), Np


def scan_groups(Cc: int, block_modes: int) -> list[tuple[int, int]]:
    """Return the groups of channels that one chunk-scan launch handles, as (first, count)."""
    # A thread block of w warps (`kernels.chunk.threads`) scans max(1, w // 2) channels. Kernels for
    # more channels spill registers and run slower, and on some GPUs their stack frames exceed 1 KB.
    # A kernel with such a frame raises the stack size of the CUDA context, which makes every
    # spilling kernel that is launched later run slower.
    group_size = max(1, block_modes // 16)
    return [(c0, min(group_size, Cc - c0)) for c0 in range(0, Cc, group_size)]


def sums_dtype(dtype: torch.dtype, launches: int) -> torch.dtype:
    """Return the dtype in which the scans accumulate an output for a `dtype` input.

    When more than one kernel launch adds to the output, the output is float32, so that a
    half-precision result is rounded once.
    """
    return torch.float32 if launches > 1 else dtype


def scan_outputs(u, dt, A, B, C, D, x0, disc, reverse, final, save):
    """Allocate y, the final state xN and the states Xs at the chunks' starts."""
    Bsz, H, L = u.shape
    block_modes, Np = blocks(A.shape[1])
    xN = A.new_empty((Bsz, H, A.shape[1], 2) if final else 0)
    Xs = A.new_empty((Bsz, H, ceil_div(L, CHUNK), 2 * Np) if save else 0)
    return u.new_empty(Bsz, C.shape[0], H, L, dtype=sums_dtype(u.dtype, Np // block_modes)), xN, Xs


@launcher
def scan_kernels(device, N, Cc, dtype, prec, reverse, disc, final, save, sums):
    """Return functions that launch the kernels of `chunk_scan_op`, keyed by channel count.

    With save, the launcher also builds `scan_backward_kernels`, so that the forward and backward
    kernels compile in parallel. The backward gets a gradient of the final state only when the loss
    depends on that state, so with `final` the launcher builds the backward kernels for both cases.
    """
    from s4ever._backends.tilelang.kernels import chunk as k
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    if save:
        for has_final in {False, final}:
            scan_backward_kernels(device, N, Cc, dtype, prec, reverse, disc, has_final, sums)
    block_modes, Np = blocks(N)
    groups = scan_groups(Cc, block_modes)
    args = (DTYPES[dtype], prec, reverse, disc, len(groups) > 1, save)
    y_dtype = DTYPES[sums_dtype(dtype, Np // block_modes)]
    counts = {gc for _, gc in groups}
    return {gc: lazy(k.forward_kernel, CHUNK, block_modes, gc, *args, y_dtype) for gc in counts}


@launcher
def scan_backward_kernels(device, N, Cc, dtype, prec, reverse, disc, has_final, sums):
    """Return functions that launch the kernels of `chunk_scan_backward_op`.

    The first function launches the kernel of the final state's gradient, and it is None without
    has_final. The second item maps a channel count to its backward kernel. The last function adds
    the sums of the batch blocks, and it is None without sums.
    """
    from s4ever._backends.tilelang.kernels import chunk as k
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    block_modes, Np = blocks(N)
    groups = scan_groups(Cc, block_modes)
    args = (DTYPES[dtype], prec, reverse, disc, len(groups) > 1)
    gu_dtype = DTYPES[sums_dtype(dtype, Np // block_modes * len(groups) + has_final)]
    final = (
        lazy(k.final_kernel, CHUNK, block_modes, DTYPES[dtype], reverse, disc)
        if has_final
        else None
    )
    counts = {gc for _, gc in groups}
    kernels = {
        gc: lazy(k.backward_kernel, CHUNK, block_modes, gc, *args, gu_dtype) for gc in counts
    }
    return final, kernels, lazy(k.sums_kernel, N) if sums else None


@register_op("chunk_scan", scan_outputs)
def chunk_scan_op(
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
) -> tuple[Tensor, Tensor, Tensor]:
    y, xN, Xs = scan_outputs(u, dt, A, B, C, D, x0, disc, reverse, final, save)
    block_modes, Np = blocks(A.shape[1])
    # The last argument says whether the backward needs `kernels.chunk.sums_kernel`, which it does
    # for more than one block of batch rows.
    args = (u.dtype, precision(), reverse, disc, final, save, save and u.shape[0] > CHUNK_ROWS)
    kernels = scan_kernels(u.device, A.shape[1], C.shape[0], *args)
    Xb = Xs if save else unused(u.device, (1,) * 4)
    for c0, gc in scan_groups(C.shape[0], block_modes):
        fn = kernels[gc]
        Dg = None if D is None else D[c0 : c0 + gc]
        xN_ = xN if final and c0 == 0 else None
        # Each launch scans one block of modes. The launches after the first add to y, and only the
        # first adds the D term.
        for n0 in range(0, Np, block_modes):
            fn(
                u,
                dt,
                A,
                B,
                C,
                or_unused(Dg, u, 1, 1),
                or_unused(x0, u, 1, 1, 1, 2),
                y,
                or_unused(xN_, u, 1, 1, 1, 2),
                Xb,
                n0,
                c0,
                int(n0 > 0),
                int(n0 == 0 and D is not None),
                int(x0 is not None),
                int(xN_ is not None),
            )
    return y, xN, Xs


def scan_grads(u, gy, dt, A, B, C, D, Xs, g_final, disc, reverse, want_dx0):
    """Allocate gu and the gradients of dt, A, B, C, D and x0."""
    block_modes, Np = blocks(A.shape[1])
    launches = Np // block_modes * len(scan_groups(C.shape[0], block_modes)) + (g_final is not None)
    gu = torch.empty_like(u, dtype=sums_dtype(u.dtype, launches))
    dD = A.new_empty(0) if D is None else torch.empty_like(D)
    dx0 = A.new_empty((u.shape[0], *A.shape[:2], 2) if want_dx0 else 0)
    return [gu, *(torch.empty_like(t) for t in (dt, A, B, C)), dD, dx0]


@register_op("chunk_scan_backward", scan_grads)
def chunk_scan_backward_op(
    u: Tensor,
    gy: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    Xs: Tensor,
    g_final: Tensor | None,
    disc: str,
    reverse: bool,
    want_dx0: bool,
) -> list[Tensor]:
    grads = scan_grads(u, gy, dt, A, B, C, D, Xs, g_final, disc, reverse, want_dx0)
    gu, ddt, dA, dB, dC, dD, dx0 = grads
    # Each block of CHUNK_ROWS batch rows adds to its own sums of the parameters' gradients. After
    # the launches, `kernels.chunk.sums_kernel` adds these sums in order, so that the gradients are
    # the same in every run.
    batch_blocks = ceil_div(u.shape[0], CHUNK_ROWS)
    sums = [ddt, dA, dB, dC]
    if batch_blocks > 1:
        sums = [t.new_empty(batch_blocks * t.shape[0], *t.shape[1:]) for t in sums]
    if D is None:
        sD = unused(u.device, (1, batch_blocks))
    else:
        sD = dD if batch_blocks == 1 else dD.new_empty(dD.shape[0], batch_blocks * dD.shape[1])
    torch._foreach_zero_(sums if D is None else [*sums, sD])
    sdt, sA, sB, sC = sums
    block_modes, Np = blocks(A.shape[1])
    args = (u.dtype, precision(), reverse, disc, g_final is not None, batch_blocks > 1)
    kernels = scan_backward_kernels(u.device, A.shape[1], C.shape[0], *args)
    final_kernel, kernels, sums_kernel = kernels
    g_last = None
    if g_final is not None:
        g_last = torch.empty_like(g_final)
        gt = u.new_empty(*u.shape[:2], CHUNK, dtype=torch.float32)
        for n0 in range(0, Np, block_modes):
            final_kernel(u, dt, A, B, Xs, g_final, g_last, gt, sdt, sA, sB, n0, int(n0 > 0))
    for i, (c0, gc) in enumerate(scan_groups(C.shape[0], block_modes)):
        fn = kernels[gc]
        Dg, sDg = (None, sD) if D is None else (D[c0 : c0 + gc], sD[c0 : c0 + gc])
        dx0g = dx0 if i == 0 or not want_dx0 else torch.empty_like(dx0)
        g_ = g_last if i == 0 else None
        dx0_ = dx0g if want_dx0 else None
        for n0 in range(0, Np, block_modes):
            fn(
                u,
                gy,
                dt,
                A,
                B,
                C,
                or_unused(Dg, u, 1, 1),
                Xs,
                or_unused(g_, u, 1, 1, 1, 2),
                gu,
                or_unused(dx0_, u, 1, 1, 1, 2),
                sdt,
                sA,
                sB,
                sC,
                sDg,
                n0,
                c0,
                int(n0 > 0 or i > 0),
                int(n0 == 0 and D is not None),
                int(g_ is not None),
                int(dx0_ is not None),
            )
        if dx0g is not dx0:
            dx0 += dx0g
    if batch_blocks > 1:
        dD_ = dD if D is not None else unused(u.device, (1, 1))
        sums_kernel(sdt, sA, sB, sC, sD, ddt, dA, dB, dC, dD_, int(D is not None))
    if g_final is not None:
        # gt holds the gradient that flows from g_final to the inputs of the last chunk.
        L = u.shape[-1]
        last = L - (Xs.shape[2] - 1) * CHUNK  # steps in the last chunk
        # pyrefly: ignore[unbound-name]  gt is set under the same condition
        gu[..., slice(None, last) if reverse else slice(L - last, None)] += gt[..., :last]
    return grads


class ChunkScan(torch.autograd.Function):
    """Chunked scans of u (B, L, H) into y (B, L, C, H) over blocks of modes."""

    @staticmethod
    def forward(ctx, u, dt, A, B, C, D, x0, config, grad):
        ctx.set_materialize_grads(False)
        u = transpose(u)
        y, xN, Xs = chunk_scan_op(u, dt, A, B, C, D, x0, *config, grad)
        ctx.save_for_backward(u, dt, A, B, C, D, Xs)
        ctx.config = config
        y = features_last(y, u.dtype)
        return y, xN if config[-1] else None

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        gy, g_final = grads
        u, dt, A, B, C, D, Xs = ctx.saved_tensors
        disc, reverse, _ = ctx.config
        Bsz, H, L = u.shape
        if gy is None:
            gy = u.new_zeros(Bsz, C.shape[0], H, L)
        else:
            gy = sequences_last(gy)
        g_final = None if g_final is None else g_final.contiguous()
        want_dx0 = ctx.needs_input_grad[6]
        gu, ddt, dA, dB, dC, dD, dx0 = chunk_scan_backward_op(
            u, gy, dt, A, B, C, D, Xs, g_final, disc, reverse, want_dx0
        )
        dD = dD if D is not None else None
        return transpose(gu, u.dtype), ddt, dA, dB, dC, dD, dx0 if want_dx0 else None, None, None


@on_device
def s4d_chunk(
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
    """Compute `s4ever.ops.s4d` in chunk mode."""
    config = (disc, reverse, output_final_state)
    grad = tracks_grad(u, dt, A, B, C, D, initial_state)
    return ChunkScan.apply(u, *contiguous_f32(dt, A, B, C, D, initial_state), config, grad)


def channel_groups(Cg: int, *tensors: Tensor):
    if Cg == tensors[0].shape[0]:
        return [tensors]
    return zip(*(t.split(Cg) for t in tensors), strict=True)


@launcher
def vandermonde_kernels(device, N, Cc, L, disc, grad):
    """Return the channels per launch of `s4d_conv_kernel_op` and functions that launch its kernels.

    The functions are keyed by channel count, because the last launch can have fewer channels. With
    grad, the launcher also builds `vandermonde_backward_kernels`, so that the forward and backward
    kernels compile in parallel.
    """
    from s4ever._backends.tilelang.kernels import vandermonde as v

    if grad:
        vandermonde_backward_kernels(device, N, Cc, L, disc)
    Cg, NA = v.config(N, Cc, L, smem(device), False)
    return Cg, {n: lazy(v.forward_kernel, N, n, disc, NA) for n in {Cg, Cc % Cg} - {0}}


@launcher
def vandermonde_backward_kernels(device, N, Cc, L, disc):
    """Return what `vandermonde_kernels` returns, for `s4d_conv_kernel_backward_op`."""
    from s4ever._backends.tilelang.kernels import vandermonde as v

    Cg, NA = v.config(N, Cc, L, smem(device), True)
    return Cg, {n: lazy(v.backward_kernel, N, n, disc, NA) for n in {Cg, Cc % Cg} - {0}}


@register_op("s4d_conv_kernel", lambda dt, A, B, C, L, disc, grad: C.new_empty(*C.shape[:2], L))
def s4d_conv_kernel_op(
    dt: Tensor, A: Tensor, B: Tensor, C: Tensor, L: int, disc: str, grad: bool
) -> Tensor:
    Cc = C.shape[0]
    k = C.new_empty(Cc, C.shape[1], L)
    Cg, kernels = vandermonde_kernels(C.device, A.shape[1], Cc, L, disc, grad)
    for Cp, kp in channel_groups(Cg, C, k):
        kernels[Cp.shape[0]](dt, A, B, Cp, kp)
    return k


@register_op("s4d_conv_kernel_backward", lambda *args: [torch.empty_like(x) for x in args[:4]])
def s4d_conv_kernel_backward_op(
    dt: Tensor, A: Tensor, B: Tensor, C: Tensor, g: Tensor, disc: str
) -> list[Tensor]:
    gC = torch.empty_like(C)
    Cg, kernels = vandermonde_backward_kernels(C.device, A.shape[1], C.shape[0], g.shape[-1], disc)
    parts = []
    for Cp, gp, gCp in channel_groups(Cg, C, g, gC):
        parts.append([torch.empty_like(x) for x in (dt, A, B)])
        kernels[Cp.shape[0]](dt, A, B, Cp, gp, *parts[-1], gCp)
    return [functools.reduce(torch.add, x) for x in zip(*parts, strict=True)] + [gC]


class S4DConvKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, B, C, L, disc, grad):
        ctx.save_for_backward(dt, A, B, C)
        ctx.disc = disc
        return s4d_conv_kernel_op(dt, A, B, C, L, disc, grad)

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        g = grads[0].contiguous()
        return *s4d_conv_kernel_backward_op(*ctx.saved_tensors, g, ctx.disc), None, None, None


@on_device
def s4d_conv_kernel(dt: Tensor, A: Tensor, B: Tensor, C: Tensor, L: int, disc: str) -> Tensor:
    """Compute `s4ever.ops.s4d_conv_kernel` for float32 parameters.

    The parameters have at most MAX_CONV_KERNEL_MODES modes.
    """
    return S4DConvKernel.apply(*contiguous_f32(dt, A, B, C), L, disc, tracks_grad(dt, A, B, C))
