"""Kernel that computes the inputs of the Cauchy kernels for S4's responses to initial states.

For the readout c, the response to a state x0 at step t is 2 Re(c dA^(t+1) x0). This response is
also the convolution kernel of the transposed system, which has the input matrix C and the
readout X:

    X = (2/dt + M) x0 / 2
    M = Lambda - P P^*

The Cauchy kernels therefore take C in place of B, Q = conj(P) in place of P, and X in place of C.
"""

from typing import Any, NamedTuple

import tilelang.language as T

from s4ever._backends.shared import ceil_div

from ..runtime import compiled
from .dsl import add, cmul, conj, f32, keep, scale, var


class Mode(NamedTuple):
    """Mode n of a lane, with its values of Lambda and P.

    The loads read index m = min(n, N - 1), and they return zeros unless `live` (n < N) is true.
    """

    n: Any
    live: Any
    m: Any
    lam: tuple
    p: tuple


@T.macro
def put(X, idx, value, cond):
    if cond:
        T.buffer_store(X, value, idx)


def put2(X, idx, z, cond):
    for j in range(2):
        put(X, [*idx, j], z[j], cond)


def load(X, idx, live):
    return tuple(T.if_then_else(live, X[(*idx, j)], 0.0) for j in range(2))


def lane_modes(Lam, P, h, N):
    lane = T.get_thread_binding()
    modes = []
    for j in range(ceil_div(N, 32)):
        n = lane + 32 * j
        live, m = n < N, T.min(n, N - 1)
        lam, p = (keep(load(X, (h, m), live)) for X in (Lam, P))
        modes.append(Mode(n, live, m, lam, p))
    return modes


def warp_sum(values):
    return var(T.warp_reduce_sum(sum(values)))


def state(x0, modes, idx, live):
    """Return this lane's values of x0[idx] and the sum s = 2 Re sum_n conj(P_n) x0_n."""
    xs = [keep(load(x0, (*idx, md.m), T.And(md.live, live))) for md in modes]
    return xs, warp_sum(2 * cmul(conj(md.p), x)[0] for md, x in zip(modes, xs, strict=True))


def forward(x0, dt, Lam, P, X, Q, h, b, N):
    """Store Q = conj(P) and X[b, h] = ((2/dt + Lambda) x0 - P s) / 2."""
    B = x0.shape[0]
    modes = lane_modes(Lam, P, h, N)
    tau = var(2 / dt[h])
    xs, s = state(x0, modes, (T.min(b, B - 1), h), b < B)
    for md, x in zip(modes, xs, strict=True):
        Xn = add(cmul((tau + md.lam[0], md.lam[1]), x), scale(-s, md.p))
        put2(X, (b, h, md.n), scale(0.5, Xn), md.live)
        put2(Q, (h, md.n), conj(md.p), T.And(md.live, b == 0))


@compiled
def forward_kernel(N):
    """Build the kernel that computes Q = conj(P) and X from x0.

    X has Bp >= B rows, and its rows b >= B are zero.
    """
    H, B, Bp = T.dynamic("H"), T.dynamic("B"), T.dynamic("Bp")

    @T.prim_func
    def state_inputs_forward(
        x0: T.Tensor((B, H, N, 2), f32),
        dt: T.Tensor((H,), f32),
        Lam: T.Tensor((H, N, 2), f32),
        P: T.Tensor((H, N, 2), f32),
        X: T.Tensor((Bp, H, N, 2), f32),
        Q: T.Tensor((H, N, 2), f32),
    ):
        with T.Kernel(H, Bp, threads=32) as (h, b):
            forward(x0, dt, Lam, P, X, Q, h, b, N)

    return state_inputs_forward
