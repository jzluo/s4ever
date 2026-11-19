import torch
from torch import Tensor

from s4ever._backends.shared import (
    ceil_div,
    contiguous_f32,
    refuse_double_backward,
    register_op,
    tracks_grad,
)

from .layout import FINAL_STATE_ROWS
from .runtime import blocks_per_row, launcher, lazy, on_device, precision
from .transpose import transpose


def irfft(k_f: Tensor, L: int) -> Tensor:
    """Return the convolution kernels of length L from half spectra that `kernels.cauchy` wrote.

    `kernels.cauchy` scales the spectra by 1 / L, so the inverse transform does not normalize. The
    transform runs inside the registered ops, because Inductor does not generate code for complex
    dtypes.
    """
    return torch.fft.irfft(torch.view_as_complex(k_f), n=L, norm="forward")


def rfft(g: Tensor, L: int) -> Tensor:
    return torch.view_as_real(torch.fft.rfft(g.contiguous(), n=L))


@launcher
def cauchy_kernel(device, N, Cc, L, grad):
    """Return a function that launches the kernel of `s4_spectrum_op`.

    With grad, the launcher also builds `cauchy_backward_kernels`, so that the forward and backward
    kernels compile in parallel.
    """
    from s4ever._backends.tilelang.kernels import cauchy

    if grad:
        cauchy_backward_kernels(device, N, Cc, L)
    return lazy(cauchy.forward_kernel, N, Cc)


@launcher
def cauchy_backward_kernels(device, N, Cc, L):
    """Return functions that launch the kernels of `s4_spectrum_backward_op`.

    The first function is None when L has no frequency tiles.
    """
    from s4ever._backends.tilelang.kernels import cauchy

    tile_sums = lazy(cauchy.backward_kernel, N, Cc) if frequency_tiles(L) else None
    return tile_sums, lazy(cauchy.finalize_kernel, N, Cc)


@register_op(
    "s4_spectrum", lambda dt, A, P, B, C, L, grad: C.new_empty(*C.shape[:2], L // 2 + 1, 2)
)
def s4_spectrum_op(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, L: int, grad: bool
) -> Tensor:
    k_f = C.new_empty(*C.shape[:2], L // 2 + 1, 2)
    cauchy_kernel(C.device, A.shape[1], C.shape[0], L, grad)(dt, A, P, B, C, k_f, L)
    return k_f


@register_op("s4_conv_kernel", lambda dt, A, P, B, C, L, grad: C.new_empty(*C.shape[:2], L))
def s4_conv_kernel_op(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, L: int, grad: bool
) -> Tensor:
    return irfft(s4_spectrum_op(dt, A, P, B, C, L, grad), L)


@register_op("s4_conv_kernel_backward", lambda *args: [torch.empty_like(x) for x in args[:5]])
def s4_conv_kernel_backward_op(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, gK: Tensor, L: int
) -> list[Tensor]:
    return s4_spectrum_backward_op(dt, A, P, B, C, rfft(gK, L), L)


def frequency_tiles(L: int) -> int:
    """Return the number of tiles of the frequencies 0 < k < L / 2 in the Cauchy backward."""
    from s4ever._backends.tilelang.kernels import cauchy

    return ceil_div((L + 1) // 2 - 1, cauchy.BK)  # (L + 1) // 2 - 1 frequencies, BK per tile


def cauchy_parts(g: Tensor, H: int, Cc: int, N: int, L: int) -> tuple[Tensor, int]:
    """Return the buffer of partial sums and the tiles per thread block for the Cauchy backward.

    The partial sums are those of `kernels.cauchy.backward_kernel`. When L has no tiles, the tile
    count is 0, and the buffer holds zeros for `kernels.cauchy.finalize_kernel` to add.
    """
    tiles = frequency_tiles(L)
    # The tiles of a feature are spread over at most blocks_per_row thread blocks.
    tiles_per_block = ceil_div(tiles, max(1, min(tiles, blocks_per_row(H, g.device))))
    # Each block stores the 4 (Cc + 1) coefficient rows of `kernels.cauchy`, then the rows of a and
    # |w|^2.
    rows = 4 * (Cc + 1) + 2
    return (
        g.new_empty(H, ceil_div(tiles, tiles_per_block), rows, N)
        if tiles_per_block
        else g.new_zeros(H, 1, rows, N)
    ), tiles_per_block


@register_op("s4_spectrum_backward", lambda *args: [torch.empty_like(x) for x in args[:5]])
def s4_spectrum_backward_op(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, g: Tensor, L: int
) -> list[Tensor]:
    params = (dt, A, P, B, C)
    Cc, H, N = C.shape[:3]
    tile_sums, finalize = cauchy_backward_kernels(C.device, N, Cc, L)
    part, tiles_per_block = cauchy_parts(g, H, Cc, N, L)
    if tiles_per_block:
        tile_sums(*params, g, part, L, tiles_per_block)
    grads = [torch.empty_like(x) for x in params]
    finalize(*params, g, part, *grads, L)
    return grads


class S4ConvKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dt, A, P, B, C, L, grad):
        ctx.save_for_backward(dt, A, P, B, C)
        ctx.L = L
        return s4_conv_kernel_op(dt, A, P, B, C, L, grad)

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        return *s4_conv_kernel_backward_op(*ctx.saved_tensors, grads[0], ctx.L), None, None


@on_device
def s4_conv_kernel(dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, L: int) -> Tensor:
    """Compute `s4ever.ops.s4_conv_kernel` for float32 parameters.

    The parameters have at most MAX_CONV_KERNEL_MODES modes.
    """
    return S4ConvKernel.apply(*contiguous_f32(dt, A, P, B, C), L, tracks_grad(dt, A, P, B, C))


STATES = 8  # the most states that one launch of `s4_state_response`'s Cauchy kernel handles


def state_launches(Bsz: int) -> tuple[int, int]:
    """Return the states per launch and the number of launches for Bsz states."""
    per_launch = min(STATES, 1 << (Bsz - 1).bit_length())  # Bsz rounded up to a power of two
    return per_launch, ceil_div(Bsz, per_launch)


def state_outputs(dt, A, P, C, x0, L):
    return x0.new_empty(C.shape[0], *x0.shape[:2], L)


@launcher
def state_kernels(device, N, per_launch):
    """Return functions that launch the kernels of `s4_state_response_op`."""
    from s4ever._backends.tilelang.kernels import cauchy, state_inputs

    return lazy(state_inputs.forward_kernel, N), lazy(cauchy.forward_kernel, N, per_launch)


@register_op("s4_state_response", state_outputs)
def s4_state_response_op(dt: Tensor, A: Tensor, P: Tensor, C: Tensor, x0: Tensor, L: int) -> Tensor:
    (Bsz, H, N), Cc = x0.shape[:3], C.shape[0]
    per_launch, launches = state_launches(Bsz)
    inputs, kernel = state_kernels(x0.device, N, per_launch)
    out = x0.new_empty(Cc, launches * per_launch, H, L // 2 + 1, 2)
    X, Q = x0.new_empty(launches * per_launch, H, N, 2), torch.empty_like(P)
    inputs(x0, dt, A, P, X, Q)
    for i in range(Cc):
        for t in range(0, launches * per_launch, per_launch):
            kernel(dt, A, Q, C[i], X[t : t + per_launch], out[i, t : t + per_launch], L)
    return irfft(out[:, :Bsz], L)


@on_device
def s4_state_response(dt: Tensor, A: Tensor, P: Tensor, C: Tensor, x0: Tensor, L: int) -> Tensor:
    """Compute `s4ever.ops._s4.state_response` without gradients.

    The parameters are float32 tensors with at most MAX_CONV_KERNEL_MODES modes. The Cauchy kernel
    treats each state as a channel of the transposed system that `kernels.state_inputs` sets up.
    """
    return s4_state_response_op(*contiguous_f32(dt, A, P, C, x0), L)


# `s4_state_powers` supports at most this many modes, the most for which one thread block can
# square the state matrix.
MAX_POWER_MODES = 32


def powers_outputs(dt, A, P, B, T, K):
    S = 2 * A.shape[1]
    return [dt.new_empty(dt.shape[0], T, S), dt.new_empty(K, dt.shape[0], S, S)]


@register_op("s4_state_powers", powers_outputs)
def s4_state_powers_op(dt: Tensor, A: Tensor, P: Tensor, B: Tensor, T: int, K: int) -> list[Tensor]:
    from s4ever._backends.tilelang.kernels import powers

    In, Es = powers_outputs(dt, A, P, B, T, K)
    powers.forward_kernel(A.shape[1], T)(dt, A, P, B, In, Es)
    return [In, Es]


@on_device
def s4_state_powers(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, T: int, K: int
) -> tuple[Tensor, list[Tensor]]:
    """Compute `s4ever.ops._s4.state_powers` without gradients.

    The parameters are float32 tensors with at most MAX_POWER_MODES modes.
    """
    In, Es = s4_state_powers_op(*contiguous_f32(dt, A, P, B), T, K)
    return In, list(Es.unbind(0))


CHUNK = 64  # steps per chunk of `s4_group_states`, which keeps In in registers
MIN_GROUP = 16  # the fewest chunks in a group when a batch row splits into several groups


def groups(Bsz: int, L: int) -> tuple[int, int]:
    """Return the group count G and the chunks Cg per group for `s4_group_states`.

    When Bsz < FINAL_STATE_ROWS, each batch row splits into up to FINAL_STATE_ROWS // Bsz groups of
    at least MIN_GROUP chunks, so that the groups fill the rows of a thread block. With more than
    one group, G Cg is the smallest power of two at or above the chunk count, because
    `ops._s4.final_state` combines the states of the groups in halves.
    """
    chunks = ceil_div(L, CHUNK)
    # G is the largest power of two at or below the group count that the limits allow, or 1.
    G = 1 << max(0, min(FINAL_STATE_ROWS // Bsz, chunks // MIN_GROUP).bit_length() - 1)
    # With more than one group, Cg is chunks / G rounded up to a power of two.
    return G, chunks if G == 1 else 1 << (ceil_div(chunks, G) - 1).bit_length()


def group_outputs(u, In, E, G, Cg):
    return In.new_empty(u.shape[1], G * u.shape[0], In.shape[-1])


@register_op("s4_group_states", group_outputs)
def s4_group_states_op(u: Tensor, In: Tensor, E: Tensor, G: int, Cg: int) -> Tensor:
    from s4ever._backends.tilelang.kernels import final_state
    from s4ever._backends.tilelang.kernels.dsl import DTYPES

    X = group_outputs(u, In, E, G, Cg)
    final_state.forward_kernel(CHUNK, In.shape[-1], DTYPES[u.dtype], precision())(u, In, E, X, Cg)
    return X


@on_device
def s4_group_states(u: Tensor, In: Tensor, E: Tensor) -> Tensor:
    """Return the states (H, G B, 2N) at the ends of G groups of chunks of u (B, L, H).

    The kernel scans each group of consecutive chunks from a zero state, without gradients, and
    orders the rows by group, then by batch row. It pads the sequences with zeros in front, so that
    the last group ends at the last step of u. `In` (H, CHUNK, 2N) is the Krylov block from
    `s4_state_powers`, and E is dA^CHUNK - I.
    """
    u = transpose(u)
    return s4_group_states_op(u, In, E, *groups(u.shape[0], u.shape[-1]))
