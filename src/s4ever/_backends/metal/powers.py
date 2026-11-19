"""S4's state powers on MPS, computed with the method of `_backends.tilelang.kernels.powers`."""

import functools

from torch import Tensor

from s4ever._backends import shared

from . import msl

# The kernels support at most 32 modes, because `squares` pads the matrix E (2N, 2N) to 64 x 64 and
# `krylov` puts mode n in lane n of one SIMD group.
MAX_MODES = 32
# A `squares` threadgroup has 8 SIMD groups, and each computes one row of 8 x 8 tiles of E.
THREADS = 256


@functools.cache
def library():
    prelude = f"constant constexpr uint NP = {MAX_MODES}, THREADS = {THREADS};\n"
    return msl.library(msl.source(prelude, "powers.metal"), {"squares": THREADS})


def fits(N: int) -> bool:
    return N <= MAX_MODES and builds()


@msl.constant
def builds() -> bool:
    return msl.builds("S4's state kernels", library)


def powers_outputs(dt, A, P, B, T, K):
    H, S = len(dt), 2 * A.shape[1]
    return [dt.new_empty(H, T, S), dt.new_empty(K, H, S, S)]


@shared.register_op("metal_s4_state_powers", powers_outputs, "MPS")
def s4_state_powers_op(dt: Tensor, A: Tensor, P: Tensor, B: Tensor, T: int, K: int) -> list[Tensor]:
    In, Es = powers_outputs(dt, A, P, B, T, K)
    H, N = A.shape[:2]
    lib = library()
    lib.squares(Es, dt, A, P, B, H, N, K, threads=[H * THREADS, 1, 1], group_size=[THREADS, 1, 1])
    lib.krylov(In, dt, A, P, B, N, T, threads=[H * 32, 1, 1], group_size=[32, 1, 1])
    return [In, Es]


def state_powers(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, T: int, K: int
) -> tuple[Tensor, list[Tensor]]:
    """Compute `s4ever.ops._s4.state_powers` on MPS without gradients, for N that `fits` accepts."""
    In, Es = s4_state_powers_op(*shared.contiguous_f32(dt, A, P, B), T, K)
    return In, list(Es.unbind(0))
