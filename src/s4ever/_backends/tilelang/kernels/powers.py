"""Kernel for S4's state powers (`ops._s4.state_powers`).

`forward_kernel` computes the Krylov block of dB and the squares dA^(2^k) - I from the parameters,
with dA in the form

    J = (2/dt - Lambda)^-1
    s = 1 + 2 Re(conj(P) J P)
    c = 4 / (dt s)
    m = 2 J Lambda
    u = J P
    v = 2 conj(conj(P) J)
    E = dA - I = blockdiag(m) - c u v^T

where u and v are vectors of (re, im) pairs. CHAIN threads square E as E <- 2 E + E^2 on 8 x 8
register tiles. Shared memory also holds a swizzled transpose Et of E, so that both factors of a
product are loaded as vectors. The swizzle spreads the stores into Et over the memory banks.
"""

import tilelang.language as T

from ..runtime import compiled
from .dsl import cmul, conj, f32, keep, scale, var

S2 = 64  # the padded state dimension
NP = S2 // 2  # the padded mode count
CHAIN = 128  # threads squaring E
# The barriers are placed by hand. TileLang 0.1.15's barrier inference puts a barrier between two
# stores of one thread to shared memory when it cannot prove that they go to different addresses,
# and here it would add a barrier before each of the 8 rounds of stores into Et.
CONFIG = {"tl.disable_thread_storage_sync": True}


def tile(t):
    """Return the tile (ti, tj) that thread t computes in a product, and the half kh that it sums.

    Lanes 16 apart compute the same tile over the two halves of the inner dimension.
    """
    index = 16 * (t // 32) + t % 16
    return index // 8, index % 8, t % 32 // 16


def rc(t, a):
    """Return the matrix row that holds row a < 8 of the tiles in tile row t.

    Columns use the same map.
    """
    return 4 * t + a % 4 + S2 // 2 * (a // 4)


def sw(r, c):
    """Return the column in which the swizzled transpose of M stores M[r, c]."""
    return (((r // 4) ^ ((c // 4) % 8)) * 4) + r % 4


def mode(dt, Lam, P, B, h, n, N):
    """Return dt, Lambda, P, B, J and `live` for mode n of feature h.

    `live` is 1.0 for n < N and 0.0 otherwise, and P and B are zero for the modes n >= N.
    """
    live = T.cast(n < N, f32)
    clamped = T.min(n, N - 1)
    d = dt[h]
    lam = (Lam[h, clamped, 0], Lam[h, clamped, 1])
    p = (P[h, clamped, 0] * live, P[h, clamped, 1] * live)
    b = (B[h, clamped, 0] * live, B[h, clamped, 1] * live)
    zr = var(2 / d - lam[0])
    den = var(zr * zr + lam[1] * lam[1])
    return d, lam, p, b, (var(zr / den), var(lam[1] / den)), live


@T.macro
def prologue(dt, Lam, P, B, h, N: int, md, uv, terms, scalars, JB, cu):
    """Store m in md, (u, v) in uv, J B in JB, (s, beta, c) in scalars and c u in cu.

    beta is 2 Re(conj(P) J B). Thread n first stores its mode's terms of the sums s and beta in
    terms[:, n].
    """
    tx = T.get_thread_binding()
    if tx < NP:
        _, lam, p, b, J, live = mode(dt, Lam, P, B, h, tx, N)
        m, u, db = scale(2, cmul(J, lam)), cmul(J, p), cmul(J, b)
        pJ = keep(cmul(conj(p), J))
        for j in T.unroll(2):
            md[tx, j] = T.if_then_else(j == 0, m[0], m[1]) * live
            uv[0, 2 * tx + j] = T.if_then_else(j == 0, u[0], u[1])
            uv[1, 2 * tx + j] = T.if_then_else(j == 0, 2 * pJ[0], -2 * pJ[1])
            JB[tx, j] = T.if_then_else(j == 0, db[0], db[1])
        terms[0, tx] = 2 * cmul(pJ, p)[0]
        terms[1, tx] = 2 * cmul(pJ, b)[0]
    T.sync_threads()
    if tx == 0:
        scalars[0] = 1.0
        scalars[1] = 0.0
        for n in T.serial(NP):
            scalars[0] = scalars[0] + terms[0, n]
            scalars[1] = scalars[1] + terms[1, n]
        scalars[2] = 4 / (dt[h] * scalars[0])
    T.sync_threads()
    if tx < S2:
        cu[tx] = scalars[2] * uv[0, tx]
    T.sync_threads()


@T.macro
def mm(acc, At, Bm, ti, tj, kh):
    """Add this thread's tile of A Bm, over half kh of the inner dimension, to acc.

    At is the swizzled transpose of A.
    """
    for k4 in T.serial(S2 // 8):
        for kk in T.unroll(4):
            k = S2 // 2 * kh + 4 * k4 + kk
            for a in T.unroll(8):
                for b in T.unroll(8):
                    i, j = rc(ti, a), rc(tj, b)
                    acc[8 * a + b] += At[k, sw(i, k)] * Bm[k, j]


@T.macro
def fold(acc, low):
    """Add to acc the tile that the lane 16 away summed over the other half of the inner dimension.

    With `low`, this lane keeps rows 0-3 of the sum in acc[:32], and otherwise it keeps rows 4-7 in
    acc[32:].
    """
    for q in T.unroll(32):
        other = T.shfl_xor(T.if_then_else(low, acc[32 + q], acc[q]), 16)
        if low:
            acc[q] = acc[q] + other
        else:
            acc[32 + q] = acc[32 + q] + other


@T.macro
def put_t(Mt, acc, ti, tj, kh):
    """Store half kh of the rows of this thread's tile acc into Mt, the swizzled transpose of M."""
    for b in T.unroll(8):
        for a in T.unroll(8):
            if a // 4 == kh:
                Mt[rc(tj, b), sw(rc(ti, a), rc(tj, b))] = acc[8 * a + b]


@T.macro
def put(M, Mt, acc, ti, tj, kh):
    for a in T.unroll(8):
        if a // 4 == kh:
            for b in T.unroll(8):
                M[rc(ti, a), rc(tj, b)] = acc[8 * a + b]
    put_t(Mt, acc, ti, tj, kh)


@T.macro
def store(X, acc, k, h, ti, tj, kh, S: int):
    """Store half kh of the rows of this thread's tile acc into the S x S matrix X[k, h]."""
    for a in T.unroll(8):
        for c in T.unroll(2):
            i, j = rc(ti, a), rc(tj, 4 * c)
            if a // 4 == kh and i < S and j < S:
                if S % 4 == 0:
                    for b in T.vectorized(4):
                        X[k, h, i, j + b] = acc[8 * a + 4 * c + b]
                else:
                    for b in T.unroll(4):
                        if j + b < S:
                            X[k, h, i, j + b] = acc[8 * a + 4 * c + b]


@compiled(pass_configs=CONFIG)
def forward_kernel(N, Tk):
    """Build the kernel that computes In[:, Tk - 1 - t] = dA^t dB and Es[k] = dA^(2^k) - I.

    N is at most NP = 32.
    """
    S = 2 * N
    H, K = T.dynamic("H"), T.dynamic("K")

    @T.prim_func
    def powers_forward(
        dt: T.Tensor((H,), f32),
        Lam: T.Tensor((H, N, 2), f32),
        P: T.Tensor((H, N, 2), f32),
        B: T.Tensor((H, N, 2), f32),
        In: T.Tensor((H, Tk, S), f32),
        Es: T.Tensor((K, H, S, S), f32),
    ):
        with T.Kernel(H, threads=CHAIN + 32) as h:
            E = T.alloc_shared((S2, S2), f32)
            Et = T.alloc_shared((S2, S2), f32)
            md = T.alloc_shared((NP, 2), f32)
            uv = T.alloc_shared((2, S2), f32)
            terms = T.alloc_shared((2, NP), f32)
            JB = T.alloc_shared((NP, 2), f32)
            scalars = T.alloc_shared((3,), f32)
            cu = T.alloc_shared((S2,), f32)
            acc = T.alloc_local((64,), f32)
            xr = T.alloc_var(f32)
            xi = T.alloc_var(f32)
            tx = T.get_thread_binding()
            ti, tj, kh = tile(tx)
            prologue(dt, Lam, P, B, h, N, md, uv, terms, scalars, JB, cu)
            if tx < CHAIN:
                # E = blockdiag(m) - c u v^T. The threads store the rank-one term, and then thread n
                # stores the 2 x 2 block of mode n.
                for a in T.unroll(8):
                    if a // 4 == kh:
                        for b in T.unroll(8):
                            E[rc(ti, a), rc(tj, b)] = -cu[rc(ti, a)] * uv[1, rc(tj, b)]
                T.sync_threads(1, CHAIN)
                if tx < NP:
                    for row in T.unroll(2):
                        for col in T.unroll(2):
                            m_entry = T.if_then_else(
                                row == col,
                                md[tx, 0],
                                T.if_then_else(row == 0, -md[tx, 1], md[tx, 1]),
                            )
                            E[2 * tx + row, 2 * tx + col] = (
                                m_entry - scalars[2] * uv[0, 2 * tx + row] * uv[1, 2 * tx + col]
                            )
                T.sync_threads(1, CHAIN)
                for a in T.unroll(8):
                    for b in T.unroll(8):
                        acc[8 * a + b] = E[rc(ti, a), rc(tj, b)]
                put_t(Et, acc, ti, tj, kh)
                store(Es, acc, 0, h, ti, tj, kh, S)
                T.sync_threads(1, CHAIN)
                for k in T.serial(K - 1):
                    for q in T.unroll(64):
                        acc[q] = 0.0
                    mm(acc, Et, E, ti, tj, kh)
                    fold(acc, kh == 0)
                    for a in T.unroll(8):
                        if a // 4 == kh:
                            for b in T.unroll(8):
                                acc[8 * a + b] += 2 * E[rc(ti, a), rc(tj, b)]
                    T.sync_threads(1, CHAIN)
                    put(E, Et, acc, ti, tj, kh)
                    store(Es, acc, k + 1, h, ti, tj, kh, S)
                    T.sync_threads(1, CHAIN)
            else:
                # The last 32 threads compute the Krylov block, one mode per thread. The product
                # dA x is x + m x - c u (v . x).
                n = tx - CHAIN
                mr, mi = var(md[n, 0]), var(md[n, 1])
                cur, cui = var(cu[2 * n]), var(cu[2 * n + 1])
                vr, vi = var(uv[1, 2 * n]), var(uv[1, 2 * n + 1])
                xr = 2 * (JB[n, 0] - uv[0, 2 * n] * scalars[1] / scalars[0])
                xi = 2 * (JB[n, 1] - uv[0, 2 * n + 1] * scalars[1] / scalars[0])
                for t in T.serial(Tk):
                    if tx - CHAIN < N:
                        In[h, Tk - 1 - t, 2 * n] = xr
                        In[h, Tk - 1 - t, 2 * n + 1] = xi
                    vx = T.warp_reduce_sum(vr * xr + vi * xi)
                    yr = xr + (mr * xr - mi * xi) - cur * vx
                    xi = xi + (mr * xi + mi * xr) - cui * vx
                    xr = yr

    return powers_forward
