from torch import Tensor

from s4ever._backends.shared import contiguous_f32, recurrent_outputs, register_op

from .runtime import launcher, on_device, unused


def recurrent_blocks(N: int, step: bool) -> tuple[int, int]:
    """Return the padded mode count and the batch rows per thread block of `kernels.recurrent`.

    The padded count is a power of two of at least 32, because the kernels spread the modes of each
    feature over the 32 lanes of a warp.
    """
    from s4ever._backends.tilelang.kernels import recurrent

    Np = max(32, 1 << (N - 1).bit_length())
    # A single step spends most of its time setting up the parameters, and a block of two rows sets
    # them up once for both rows. Longer scans run faster with one row per block, because there are
    # then more blocks.
    return Np, 2 if step and Np <= recurrent.REG_MODES else 1


@register_op("s4d_recurrent", lambda u, dt, A, B, C, *_: recurrent_outputs(u, A, C))
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
    y, xN = recurrent_outputs(u, A, C)
    step = u.shape[1] == 1 and u.shape[0] > 1
    scan = s4d_scan(u.device, A.shape[1], C.shape[0], u.dtype, disc, reverse, step)
    scan(u, dt, A, B, C, D, x0, y, xN)
    return y, xN


@launcher
def s4d_scan(device, N, Cc, dtype, disc, reverse, step):
    from s4ever._backends.tilelang.kernels import recurrent
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    Np, rows = recurrent_blocks(N, step)
    kernel = recurrent.s4d_kernel(Np, Cc, DTYPES[dtype], disc, reverse, rows)
    no_D, no_x0 = unused(device, (1, 1)), unused(device, (1, 1, 1, 2))

    def scan(u, dt, A, B, C, D, x0, y, xN):
        flags = (int(D is not None), int(x0 is not None))
        kernel(u, dt, A, B, C, no_D if D is None else D, no_x0 if x0 is None else x0, y, xN, *flags)

    return scan


@on_device
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
    """Compute `s4ever.ops.s4d` in recurrent mode, without gradients."""
    params = contiguous_f32(dt, A, B, C, D, initial_state)
    u = u.contiguous()
    y, xN = s4d_recurrent_op(u, *params, disc, reverse)
    return y, xN if output_final_state else None


@register_op("s4_recurrent", lambda u, dt, A, P, B, c, *_: recurrent_outputs(u, A, c))
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
    y, xN = recurrent_outputs(u, A, c)
    step = u.shape[1] == 1 and u.shape[0] > 1
    s4_scan(u.device, A.shape[1], c.shape[0], u.dtype, reverse, step)(
        u, dt, A, P, B, c, D, x0, y, xN
    )
    return y, xN


@launcher
def s4_scan(device, N, Cc, dtype, reverse, step):
    from s4ever._backends.tilelang.kernels import recurrent
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    Np, rows = recurrent_blocks(N, step)
    kernel = recurrent.s4_kernel(Np, Cc, DTYPES[dtype], reverse, rows)
    no_D, no_x0 = unused(device, (1, 1)), unused(device, (1, 1, 1, 2))

    def scan(u, dt, A, P, B, c, D, x0, y, xN):
        flags = (int(D is not None), int(x0 is not None))
        kernel(
            u, dt, A, P, B, c, no_D if D is None else D, no_x0 if x0 is None else x0, y, xN, *flags
        )

    return scan


@on_device
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
    """Compute `s4ever.ops.s4` in recurrent mode, without gradients.

    `cvec` (C, H, 2N) holds 2 conj(c) for the readout c, as in the first output of
    `s4ever.ops._s4.s4_recurrence`.
    """
    cvec = cvec.unflatten(-1, (-1, 2))
    u = u.contiguous()
    y, xN = s4_recurrent_op(u, *contiguous_f32(dt, A, P, B, cvec, D, initial_state), reverse)
    return y, xN if output_final_state else None
