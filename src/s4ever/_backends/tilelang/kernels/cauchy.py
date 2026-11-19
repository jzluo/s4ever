"""S4's kernel spectrum, as Cauchy sums over the modes with a low-rank correction.

For each channel of C, the spectrum at frequency k is

    k_f = (1 + z/2) (r_BC - r_BQ r_PC / (1 + r_PQ))
    z = i nu
    nu = 2 tan(pi k / L)

The resolvent sums r_XY are defined for X in (B, P) and Y in (C..., Q), where Q is conj(P):

    w = dt Lambda
    a = Re w
    v_XY = dt X Y
    r_XY(z) = sum_n v / (z - w) + conj(v) / (z - conj(w))

At z = i nu, the kernels compute r_XY from two Cauchy sums:

    r_XY = S0 + i nu S1
    S_p[row] = sum_n coef_p[row, n] / ((|w_n|^2 - nu^2) - 2 i nu a_n)
    coef_0 = -2 Re(v conj(w))
    coef_1 = 2 Re(v)

For even L, the Nyquist bin is the limit sum_n Re v_BC. The kernels scale the spectrum by 1 / L, so
that an inverse transform without normalization gives the convolution kernel. The backward pass
takes g, the forward transform of the kernel's gradient, and weights its bins by 2 / L, except the
bins 0 and L/2, which have no conjugate bin and get 1 / L. These weights give the adjoint of the
scaled inverse transform.

At k = 0, the term of mode n in r_XY(0) is t_XY,n = -2 Re(X_n Y_n / Lambda_n). These terms are large
for modes with a small |Lambda|, and their rounding errors would dominate
r_BC - r_BQ r_PC / (1 + r_PQ).
The kernels therefore compute the spectrum at k = 0 from the solution x of M x = B over the modes
and their conjugates:

    M = Lambda - P P^*
    k_f(0) = -C^T x = -2 Re sum_n C_n x_n
    x_n = (B_n (1 + r_PQ) - P_n r_BQ) / (Lambda_n (1 + r_PQ))

In the numerator, r_PQ and r_BQ are summed over the other modes, and the terms of mode n are added
in closed form:

    B_n t_PQ,n - P_n t_BQ,n = -2 i P_n Im(B_n conj(P_n)) / conj(Lambda_n)

Both passes compute |w|^2 - nu^2 with an fma, so that they round it the same way. Near resonance,
|w|^2 - nu^2 is small, and a rounding difference between the passes would make the gradients of P, B
and C inaccurate.
"""

import math

import tilelang.language as T
import torch

from s4ever._backends.shared import ceil_div

from ..runtime import compiled
from .dsl import add, cmul, conj, const, cos, f32, fma, keep, scale, sin, var

BK = 128  # frequencies per tile
MB = 16  # modes per block in the sums of `make_others`


def rcp(x):
    """Return the hardware's approximate reciprocal of x.

    On AMD this calls __builtin_amdgcn_rcpf, because TileLang's HIP code generation has no lowering
    for T.fast_rcp. Lowering a kernel that uses T.fast_rcp for the target "hip" fails with
    "Unresolved call".
    """
    return T.call_extern(f32, "__builtin_amdgcn_rcpf", x) if torch.version.hip else T.fast_rcp(x)


def frequency(k, L):
    """Return nu = 2 tan(pi k / L) for 0 <= k <= L / 2.

    For k > L / 4, the function computes 2 cot(pi (L - 2 k) / (2 L)), so that the arguments of sin
    and cos are at most pi / 4.
    """
    t = T.cast(k, f32) * (const(math.pi) / T.cast(L, f32))
    u = T.cast(L - 2 * k, f32) * (const(math.pi / 2) / T.cast(L, f32))
    return T.if_then_else(4 * k <= L, 2 * sin(t) / cos(t), 2 * cos(u) / sin(u))


def mode(dt, Lam, P, B, Cm, h, n, C):
    """Return w, X, Y and the products u of mode n as (re, im) pairs.

    The product u[x (C + 1) + y] is X_x Y_y = v_xy / dt.
    """
    X = [(B[h, n, 0], B[h, n, 1]), (P[h, n, 0], P[h, n, 1])]
    Y = [(Cm[c, h, n, 0], Cm[c, h, n, 1]) for c in range(C)] + [(P[h, n, 0], -P[h, n, 1])]
    u = [(xr * yr - xi * yi, xr * yi + xi * yr) for xr, xi in X for yr, yi in Y]
    return (dt[h] * Lam[h, n, 0], dt[h] * Lam[h, n, 1]), X, Y, u


def mode_chunks(N):
    """Return the chunks (first mode, count) of at most BK modes that cover N modes.

    A block of BK threads then loads at most one mode of a chunk per thread.
    """
    return [(m0, min(BK, N - m0)) for m0 in range(0, N, BK)]


def tables(params, h, n, C, cs, aw, m0=0, N=None):
    """Store coef_p[row, m] in cs[n, 2 p (C + 1) + row] and (a_m, |w_m|^2) in aw[n].

    The mode m is m0 + n. With N, the modes m >= N get zero coefficients.
    """
    dt = params[0][h]
    m = m0 + n if N is None else T.min(m0 + n, N - 1)
    (wr, wi), _, _, u = mode(*params, h, m, C)
    T.buffer_store(aw, wr, [n, 0])
    T.buffer_store(aw, wr * wr + wi * wi, [n, 1])
    for xy, (ur, ui) in enumerate(u):
        vr, vi = dt * ur, dt * ui
        coefs = -2 * (vr * wr + vi * wi), 2 * vr
        if N is not None:
            coefs = [T.if_then_else(m0 + n < N, c, 0.0) for c in coefs]
        T.buffer_store(cs, coefs[0], [n, xy])
        T.buffer_store(cs, coefs[1], [n, 2 * (C + 1) + xy])


def terms(params, h, n, C, rows):
    """Return mode n's terms t = -2 Re(u / Lambda) of the sums r_XY(0) in `rows`."""
    Lam = params[1]
    lam_r, lam_i = Lam[h, n, 0], Lam[h, n, 1]
    u = mode(*params, h, n, C)[3]
    return [
        -2 * (u[xy][0] * lam_r + u[xy][1] * lam_i) / (lam_r * lam_r + lam_i * lam_i) for xy in rows
    ]


def make_others(N, J):
    """Build the macros that add up the terms of r_XY(0) in the J columns of terms0.

    `others` stores r0[j] = sum_m terms0[m, j] and rest[j] = sum_{m != n} terms0[m, j]. Above 2 MB
    modes, `block_sums` stores the sum of each block of MB modes in block_terms. `others` then
    adds these sums, and for rest it replaces the sum of n's block with the other terms of that
    block.
    """
    NB = ceil_div(N, MB)

    @T.macro
    def block_sums(terms0, block_terms, tx):
        if tx < NB:
            # Each sum goes to block_terms in one store. TileLang 0.1.15's barrier inference counts
            # two stores of one thread to the same word as a conflict, so a sum that accumulates in
            # block_terms gets a barrier inside this branch.
            for j in T.unroll(J):
                tot = T.alloc_var(f32, init=0.0)
                for m in T.serial(MB):
                    if tx * MB + m < N:
                        tot = tot + terms0[tx * MB + m, j]
                block_terms[tx, j] = tot

    @T.macro
    def others(terms0, block_terms, n, rest, r0):
        for j in T.unroll(J):
            rest[j] = 0.0
            r0[j] = 0.0
        if N <= 2 * MB:
            for m in T.serial(N):
                for j in T.unroll(J):
                    rest[j] = rest[j] + T.if_then_else(m == n, 0.0, terms0[m, j])
                    r0[j] = r0[j] + terms0[m, j]
        else:
            for blk in T.serial(NB):
                for j in T.unroll(J):
                    rest[j] = rest[j] + T.if_then_else(blk == n // MB, 0.0, block_terms[blk, j])
                    r0[j] = r0[j] + block_terms[blk, j]
            for m in T.serial(MB):
                i = n // MB * MB + m
                for j in T.unroll(J):
                    t = terms0[T.min(i, N - 1), j]
                    rest[j] = rest[j] + T.if_then_else(T.Or(i == n, i >= N), 0.0, t)

    return block_sums, others


def solve(V, W, Lam, h, n, rest_pq, rest_v, den):
    """Return entry n of (Lambda - W W^*)^-1 V.

    `rest_pq` and `rest_v` are sums of terms over the modes other than n. `rest_pq` sums the terms
    of r_PQ(0), and `rest_v` sums those of r_BQ(0) for (V, W) = (B, P) or those of r_PC(0) for
    (V, W) = (C, Q). `den` is 1 + r_PQ(0).
    """
    lam_r, lam_i = Lam[h, n, 0], Lam[h, n, 1]
    factor = 1 / ((lam_r * lam_r + lam_i * lam_i) * den)
    num_r = V[0] + (V[0] * rest_pq - W[0] * rest_v)
    num_i = V[1] + (V[1] * rest_pq - W[1] * rest_v)
    im = V[1] * W[0] - V[0] * W[1]  # Im(V conj(W))
    return (
        (num_r * lam_r + num_i * lam_i + 2 * im * W[1]) * factor,
        (num_i * lam_r - num_r * lam_i - 2 * im * W[0]) * factor,
    )


def stores(buf, values, *idx):
    for j, v in enumerate(values):
        T.buffer_store(buf, v, [*idx, j])


def make_sums(N, R, init=True):
    @T.macro
    def sums(nu, cs, aw, S):
        """Add sum_n cs[n, j] / ((|w_n|^2 - nu^2) - 2 i nu a_n) to S[2 j] + i S[2 j + 1].

        The sum runs over the N modes in the tables. With `init`, the macro first zeroes S.
        """
        mr = T.alloc_var(f32)
        mi = T.alloc_var(f32)
        inv = T.alloc_var(f32)
        if init:
            for j in T.unroll(2 * R):
                S[j] = 0.0
        for n in T.serial(N):
            dr = fma(-nu, nu, aw[n, 1])
            di = 2 * nu * aw[n, 0]
            inv = rcp(dr * dr + di * di)
            mr = dr * inv
            mi = di * inv
            for j in T.unroll(R):
                S[2 * j] = S[2 * j] + cs[n, j] * mr
                S[2 * j + 1] = S[2 * j + 1] + cs[n, j] * mi

    return sums


def resolvents(S, r, nu, R):
    """Store r_XY = S0 + i nu S1 of the R / 2 pairs XY in r."""
    for j in range(R // 2):
        T.buffer_store(r, S[2 * j] - nu * S[R + 2 * j + 1], [2 * j])
        T.buffer_store(r, S[2 * j + 1] + nu * S[R + 2 * j], [2 * j + 1])


@compiled
def forward_kernel(N, C):
    """Build the kernel that computes the spectrum k_f (C, H, L // 2 + 1, 2), scaled by 1 / L.

    Thread tx of block (bk, h) computes bin bk BK + tx of feature h. The blocks with bk = 0 also
    compute the bins 0 and L / 2.
    """
    R, pairs = 4 * (C + 1), 2 * (C + 1)  # rows of coef_0 and coef_1 together, pairs XY
    H, F = T.dynamic("H"), T.dynamic("F")  # F = L // 2 + 1 bins
    chunks = mode_chunks(N)
    KN, last = chunks[0][1], len(chunks) - 1  # modes per chunk, index of the last chunk
    sums = [make_sums(nc, R, m0 == 0) for m0, nc in chunks]
    rows = (C, 2 * C + 1)  # r_BQ, r_PQ
    block_sums, others = make_others(N, len(rows))

    @T.prim_func
    def cauchy_forward(
        dt: T.Tensor((H,), f32),
        Lam: T.Tensor((H, N, 2), f32),
        P: T.Tensor((H, N, 2), f32),
        B: T.Tensor((H, N, 2), f32),
        Cm: T.Tensor((C, H, N, 2), f32),
        out: T.Tensor((C, H, F, 2), f32),
        L: T.int32,
    ):
        with T.Kernel(T.ceildiv(F, BK), H, threads=BK) as (bk, h):
            params = (dt, Lam, P, B, Cm)
            cs = T.alloc_shared((KN, R), f32)
            aw = T.alloc_shared((KN, 2), f32)
            terms0 = T.alloc_shared((N, len(rows)), f32)  # the terms of r_XY(0) by mode
            # block_terms holds the sums of terms0 over blocks of MB modes.
            block_terms = T.alloc_shared((ceil_div(N, MB), len(rows)), f32) if N > 2 * MB else None
            kx = T.alloc_shared((C, KN), f32)  # Re(C_n x_n), summed over the chunks
            rest = T.alloc_local((len(rows),), f32)
            r0 = T.alloc_local((len(rows),), f32)
            S = T.alloc_local((2 * R,), f32)
            r = T.alloc_local((R,), f32)
            nu = T.alloc_var(f32)
            acc = T.alloc_var(f32)
            nyq = T.alloc_var(f32, init=0.0)
            inv_L = T.alloc_var(f32, init=1 / T.cast(L, f32))
            tx = T.get_thread_binding()
            k = bk * BK + tx
            kmax = (L + 1) // 2 - 1  # the last bin below L / 2
            for ci, (m0, nc) in enumerate(chunks):
                if ci > 0:
                    T.sync_threads()
                for n in T.Parallel(nc):
                    tables(params, h, n, C, cs, aw, m0)
                T.sync_threads()
                if bk == 0:
                    if ci == 0:
                        for m1, n1 in chunks:
                            if tx < n1:
                                stores(terms0, terms(params, h, m1 + tx, C, rows), m1 + tx)
                        T.sync_threads()
                        if N > 2 * MB:
                            block_sums(terms0, block_terms, tx)
                            T.sync_threads()
                    if tx < nc:
                        n = m0 + tx
                        others(terms0, block_terms, n, rest, r0)
                        Bn, Pn = (B[h, n, 0], B[h, n, 1]), (P[h, n, 0], P[h, n, 1])
                        x = keep(solve(Bn, Pn, Lam, h, n, rest[1], rest[0], 1 + r0[1]))
                        for c in T.unroll(C):
                            re = Cm[c, h, n, 0] * x[0] - Cm[c, h, n, 1] * x[1]
                            kx[c, tx] = re if ci == 0 else kx[c, tx] + re
                    # Threads below C sum the Nyquist limit of channel tx and, at the last chunk,
                    # store the bins 0 and L / 2.
                    if ci < last:
                        if tx < C:
                            for n in T.serial(nc):
                                nyq = nyq + cs[n, pairs + tx]
                    else:
                        T.sync_threads()
                        if tx < C:
                            acc = 0.0
                            for n in T.serial(KN):
                                acc = acc + kx[tx, n]
                                nyq = nyq + T.if_then_else(n < nc, cs[n, pairs + tx], 0.0)
                            out[tx, h, 0, 0] = -2 * acc * inv_L
                            out[tx, h, 0, 1] = 0.0
                            if L % 2 == 0:
                                out[tx, h, L // 2, 0] = nyq / 2 * inv_L
                                out[tx, h, L // 2, 1] = 0.0
                if ci == 0:
                    nu = frequency(T.max(T.min(k, kmax), 1), L)
                sums[ci](nu, cs, aw, S)
            resolvents(S, r, nu, R)
            bq = C
            pq = 2 * C + 1
            den = (1 + r[2 * pq]) * (1 + r[2 * pq]) + r[2 * pq + 1] * r[2 * pq + 1]
            sr = (1 + r[2 * pq]) / den  # 1 / (1 + r_PQ)
            si = -r[2 * pq + 1] / den
            br = r[2 * bq] * sr - r[2 * bq + 1] * si  # b = r_BQ / (1 + r_PQ)
            bi = r[2 * bq] * si + r[2 * bq + 1] * sr
            for c in T.unroll(C):
                pc = C + 1 + c
                kr = r[2 * c] - (br * r[2 * pc] - bi * r[2 * pc + 1])
                ki = r[2 * c + 1] - (br * r[2 * pc + 1] + bi * r[2 * pc])
                if T.And(k >= 1, k <= kmax):
                    out[c, h, k, 0] = (kr - ki * nu / 2) * inv_L
                    out[c, h, k, 1] = (ki + kr * nu / 2) * inv_L

    return cauchy_forward


@compiled
def backward_kernel(N, C):
    """Build the kernel that computes partial gradients of the coefficient rows, a and |w|^2.

    The gradients are (H, SP, R + 2, N). Thread block (sp, h) handles `tiles_per_block` tiles of
    BK frequencies of feature h, and `finalize_kernel` adds the partial gradients of the SP blocks.
    For each frequency, a thread recomputes the forward pass. Each thread then loops over the
    frequencies for one mode, so that the gradient sums of that mode stay in registers.

    The kernel has one of two forms. With one chunk of modes (N <= BK, so last == 0), the
    tables stay loaded for all tiles, the sums of each thread accumulate over all tiles, and the W
    threads of each mode add their sums through `red` at the end. With more chunks, KN is BK, so W
    is 1 and thread tx handles mode tx of each chunk. Each tile then loads the tables of the chunks
    in turn, once for the forward sums and once in reverse order for the gradient sums, and adds
    the sums of each chunk to `part`.
    """
    R = 4 * (C + 1)  # rows of coef_0 and coef_1 together
    chunks = mode_chunks(N)
    KN = chunks[0][1]  # modes per chunk
    W = BK // KN  # threads per mode in the frequency loop
    H, F, SP = T.dynamic("H"), T.dynamic("F"), T.dynamic("SP")  # SP blocks per feature
    sums, add_sums = make_sums(KN, R), make_sums(KN, R, init=False)
    last = len(chunks) - 1
    pad = None if N % KN == 0 else N  # `tables` zeroes the modes past N in the last chunk

    @T.prim_func
    def cauchy_backward(
        dt: T.Tensor((H,), f32),
        Lam: T.Tensor((H, N, 2), f32),
        P: T.Tensor((H, N, 2), f32),
        B: T.Tensor((H, N, 2), f32),
        Cm: T.Tensor((C, H, N, 2), f32),
        g: T.Tensor((C, H, F, 2), f32),
        part: T.Tensor((H, SP, R + 2, N), f32),
        L: T.int32,
        tiles_per_block: T.int32,
    ):
        with T.Kernel(SP, H, threads=BK) as (sp, h):
            params = (dt, Lam, P, B, Cm)
            cs = T.alloc_shared((KN, R), f32)
            aw = T.alloc_shared((KN, 2), f32)
            GS = T.alloc_shared((BK, 2 * R), f32)  # the gradient of S at each thread's frequency
            nus = T.alloc_shared((BK,), f32)  # the nu of each thread's frequency
            red = T.alloc_shared((W, KN, R + 2), f32)  # the sums of the W threads of each mode
            S = T.alloc_local((2 * R,), f32)
            r = T.alloc_local((R,), f32)
            gr = T.alloc_local((R,), f32)
            acc = T.alloc_local((R + 2,), f32)
            cn = T.alloc_local((R,), f32)  # the coefficient rows of mode n
            nu = T.alloc_var(f32)
            an = T.alloc_var(f32)  # a_n
            w2n = T.alloc_var(f32)  # |w_n|^2
            inv = T.alloc_var(f32)
            mr = T.alloc_var(f32)
            mi = T.alloc_var(f32)
            gbr = T.alloc_var(f32)  # gradient of b = r_BQ / (1 + r_PQ)
            gbi = T.alloc_var(f32)
            gmr = T.alloc_var(f32)
            gmi = T.alloc_var(f32)
            for m in T.Parallel(KN):
                tables(params, h, m, C, cs, aw)
            tx = T.get_thread_binding()
            n = tx % KN
            grp = tx // KN  # the index of the thread among the threads of mode n
            kmax = (L + 1) // 2 - 1  # the last bin below L / 2
            T.sync_threads()
            if last == 0:
                for j in T.unroll(R):
                    cn[j] = cs[n, j]
                an = aw[n, 0]
                w2n = aw[n, 1]
                for j in T.unroll(R + 2):
                    acc[j] = 0.0
            for t in T.serial(tiles_per_block):
                k = (sp * tiles_per_block + t) * BK + tx + 1
                nu = frequency(T.min(k, kmax), L)
                sums(nu, cs, aw, S)
                # With one chunk, the loops over the chunks are Python ranges. This one then emits
                # nothing, and the one in the gradient loop below emits its body once without a
                # loop.
                for ci in T.serial(1, last + 1) if last > 0 else range(0):
                    T.sync_threads()
                    for m in T.Parallel(KN):
                        tables(params, h, m, C, cs, aw, ci * KN, pad)
                    T.sync_threads()
                    add_sums(nu, cs, aw, S)
                resolvents(S, r, nu, R)
                bq = C
                pq = 2 * C + 1
                den = (1 + r[2 * pq]) * (1 + r[2 * pq]) + r[2 * pq + 1] * r[2 * pq + 1]
                sr = (1 + r[2 * pq]) / den
                si = -r[2 * pq + 1] / den
                br = r[2 * bq] * sr - r[2 * bq + 1] * si
                bi = r[2 * bq] * si + r[2 * bq + 1] * sr
                weight = T.cast(k <= kmax, f32) * 2 / T.cast(L, f32)
                gbr = 0.0
                gbi = 0.0
                for c in T.unroll(C):
                    pc = C + 1 + c
                    g0 = g[c, h, T.min(k, kmax), 0] * weight
                    g1 = g[c, h, T.min(k, kmax), 1] * weight
                    # p = conj(1 + i nu / 2) g
                    pr = g0 + nu / 2 * g1
                    pi = g1 - nu / 2 * g0
                    gr[2 * c] = pr
                    gr[2 * c + 1] = pi
                    # k_f = (1 + z/2) (r_BC - b r_PC)
                    gr[2 * pc] = -(br * pr + bi * pi)
                    gr[2 * pc + 1] = -(br * pi - bi * pr)
                    gbr = gbr - (r[2 * pc] * pr + r[2 * pc + 1] * pi)
                    gbi = gbi - (r[2 * pc] * pi - r[2 * pc + 1] * pr)
                # b = r_BQ s with s = 1 / (1 + r_PQ): G_BQ = conj(s) G_b, G_PQ = -conj(s^2 r_BQ) G_b
                gr[2 * bq] = sr * gbr + si * gbi
                gr[2 * bq + 1] = sr * gbi - si * gbr
                tr = br * sr - bi * si  # b s = s^2 r_BQ
                ti = br * si + bi * sr
                gr[2 * pq] = -(tr * gbr + ti * gbi)
                gr[2 * pq + 1] = -(tr * gbi - ti * gbr)
                # r = S0 + i nu S1
                for j in T.unroll(R // 2):
                    GS[tx, 2 * j] = gr[2 * j]
                    GS[tx, 2 * j + 1] = gr[2 * j + 1]
                    GS[tx, R + 2 * j] = nu * gr[2 * j + 1]
                    GS[tx, R + 2 * j + 1] = -nu * gr[2 * j]
                nus[tx] = nu
                T.sync_threads()
                for cr in T.serial(last + 1) if last > 0 else range(1):
                    ci = last - cr  # The tables of the last chunk are already loaded.
                    if last > 0:
                        if cr > 0:
                            T.sync_threads()
                            for m in T.Parallel(KN):
                                tables(params, h, m, C, cs, aw, ci * KN, pad)
                            T.sync_threads()
                        for j in T.unroll(R):
                            cn[j] = cs[n, j]
                        an = aw[n, 0]
                        w2n = aw[n, 1]
                        for j in T.unroll(R + 2):
                            acc[j] = 0.0
                    if grp < W:
                        for i in T.serial(ceil_div(BK, W)):
                            kl = i * W + grp
                            if kl < BK:
                                nuk = nus[kl]
                                dr = fma(-nuk, nuk, w2n)
                                di = 2 * nuk * an
                                inv = rcp(dr * dr + di * di)
                                mr = dr * inv
                                mi = di * inv
                                gmr = 0.0
                                gmi = 0.0
                                for j in T.unroll(R):
                                    gmr = gmr + cn[j] * GS[kl, 2 * j]
                                    gmi = gmi + cn[j] * GS[kl, 2 * j + 1]
                                    acc[j] = acc[j] + mr * GS[kl, 2 * j] + mi * GS[kl, 2 * j + 1]
                                # q = (|w|^2 - nu^2) - 2 i nu a and m = 1 / q: G_q = -conj(m^2) G_m
                                c2 = mr * mr - mi * mi
                                s2 = 2 * mr * mi
                                acc[R] = acc[R] + 2 * nuk * (c2 * gmi - s2 * gmr)
                                acc[R + 1] = acc[R + 1] - (c2 * gmr + s2 * gmi)
                    if last > 0 and ci * KN + tx < N:
                        for j in T.unroll(R + 2):
                            if t == 0:
                                part[h, sp, j, ci * KN + tx] = acc[j]
                            else:
                                part[h, sp, j, ci * KN + tx] += acc[j]
                T.sync_threads()
            if last == 0:
                if grp < W:
                    for j in T.unroll(R + 2):
                        red[grp, n, j] = acc[j]
                T.sync_threads()
                if tx < N:
                    # The loop over j is inside the loop over i, because with the loops swapped,
                    # nvcc spills registers when W is BK (`tests/lowering.py` reports the spills).
                    for j in T.unroll(R + 2):
                        acc[j] = 0.0
                    for i in T.serial(W):
                        for j in T.unroll(R + 2):
                            acc[j] = acc[j] + red[i, tx, j]
                    for j in T.unroll(R + 2):
                        part[h, sp, j, tx] = acc[j]

    return cauchy_backward


def grad_zero(params, g, rest, r0, h, n, C, L):
    """Return mode n's gradients through k_f(0) / L with respect to Lambda, B, P and C.

    The gradient of a parameter is sum_c g_c 2 conj(G_c), where g_c is bin 0 of channel c of g and
    G_c is the derivative of channel c's k_f(0) / L with respect to the parameter. The derivatives
    use the vectors mu_c = M^-T C_c.
    """
    _, Lam, P, B, Cm = params
    Pn = (P[h, n, 0], P[h, n, 1])
    den = 1 + r0[C + 1]
    x = keep(solve((B[h, n, 0], B[h, n, 1]), Pn, Lam, h, n, rest[C + 1], rest[0], den))
    # gmu and gs collect sum_c g_c mu_c and sum_c g_c P^T mu_c, and gc collects the gradients of
    # the channels of C.
    gmu, gs, gc = (0.0, 0.0), 0.0, []
    for c in range(C):
        gk = var(g[c, h, 0, 0] / T.cast(L, f32))
        Vc = (Cm[c, h, n, 0], Cm[c, h, n, 1])
        gmu = add(gmu, scale(gk, solve(Vc, conj(Pn), Lam, h, n, rest[C + 1], rest[1 + c], den)))
        gs = gs - gk * r0[1 + c] / den  # P^T mu_c = -r_PC / den
        gc.append(scale(-2 * gk, conj(x)))
    gmu, gs = keep(gmu), var(gs)
    sx = var(-r0[0] / den)  # Q^T x
    gl = scale(2, conj(cmul(gmu, x)))
    gp = scale(-2, add(scale(sx, conj(gmu)), scale(gs, x)))
    return [gl, scale(-2, conj(gmu)), gp, *gc]


def grad_v(params, h, n, C, G, g, L, gv, gw):
    """Store the gradients of mode n's v in gv and of its w in gw.

    G holds the gradients of the coefficient rows, a and |w|^2 from `backward_kernel`. The
    gradients of v also take the Nyquist bin of g.
    """
    pairs, R = 2 * (C + 1), 4 * (C + 1)
    dt = params[0][h]
    (wr, wi), _, _, u = mode(*params, h, n, C)
    nyquist = T.cast(L % 2 == 0, f32) / T.cast(L, f32)
    gwr, gwi = G[R] + 2 * wr * G[R + 1], 2 * wi * G[R + 1]
    for xy in range(pairs):
        vr, vi = dt * u[xy][0], dt * u[xy][1]
        # the gradients of coef_0 = -2 (vr wr + vi wi) and of coef_1 = 2 vr
        g_0, g_1 = G[xy], G[pairs + xy]
        gvr, gvi = -2 * wr * g_0 + 2 * g_1, -2 * wi * g_0
        gwr, gwi = gwr - 2 * vr * g_0, gwi - 2 * vi * g_0
        if xy < C:
            gvr = gvr + nyquist * g[xy, h, L // 2, 0]
        T.buffer_store(gv, gvr, [2 * xy])
        T.buffer_store(gv, gvi, [2 * xy + 1])
    T.buffer_store(gw, gwr, [0])
    T.buffer_store(gw, gwi, [1])


def grad_params(params, grads, h, n, C, gv, gw, zero):
    """Store mode n's gradients of Lambda, P, B and C, and return its term of dt's gradient.

    The inputs are the gradients of v (gv), of w (gw) and through k_f(0) (zero).
    """
    dt, Lam = params[0][h], params[1]
    gLam, gP, gB, gC = grads
    _, X, Y, u = mode(*params, h, n, C)
    gdt = Lam[h, n, 0] * gw[0] + Lam[h, n, 1] * gw[1]
    gX = zero[1:3]
    gY = [*zero[3:], (0.0, 0.0)]
    for x, (xr, xi) in enumerate(X):
        for y, (yr, yi) in enumerate(Y):
            xy = x * (C + 1) + y
            gvr, gvi = gv[2 * xy], gv[2 * xy + 1]
            gdt = gdt + gvr * u[xy][0] + gvi * u[xy][1]
            gur, gui = dt * gvr, dt * gvi
            gX[x] = (gX[x][0] + gur * yr + gui * yi, gX[x][1] + gui * yr - gur * yi)
            gY[y] = (gY[y][0] + gur * xr + gui * xi, gY[y][1] + gui * xr - gur * xi)
    for j in range(2):
        T.buffer_store(gB, gX[0][j], [h, n, j])
        T.buffer_store(gLam, dt * gw[j] + zero[0][j], [h, n, j])
        for c in range(C):
            T.buffer_store(gC, gY[c][j], [c, h, n, j])
    T.buffer_store(gP, gX[1][0] + gY[C][0], [h, n, 0])  # Y_C = conj(P)
    T.buffer_store(gP, gX[1][1] - gY[C][1], [h, n, 1])
    return gdt


@compiled
def finalize_kernel(N, C):
    """Build the kernel that computes the gradients of dt, Lambda, P, B and C.

    The kernel adds the partial gradients of `backward_kernel` over the SP blocks and takes the
    gradients through the bins 0 and L / 2 of g. Thread tx handles mode m0 + tx of each chunk.
    """
    R, pairs = 4 * (C + 1), 2 * (C + 1)
    chunks = mode_chunks(N)
    threads = ceil_div(chunks[0][1], 32) * 32
    H, F, SP = T.dynamic("H"), T.dynamic("F"), T.dynamic("SP")
    rows = range(C, pairs)  # r_BQ, r_PC..., r_PQ
    block_sums, others = make_others(N, len(rows))

    @T.prim_func
    def cauchy_finalize(
        dt: T.Tensor((H,), f32),
        Lam: T.Tensor((H, N, 2), f32),
        P: T.Tensor((H, N, 2), f32),
        B: T.Tensor((H, N, 2), f32),
        Cm: T.Tensor((C, H, N, 2), f32),
        g: T.Tensor((C, H, F, 2), f32),
        part: T.Tensor((H, SP, R + 2, N), f32),
        gdt: T.Tensor((H,), f32),
        gLam: T.Tensor((H, N, 2), f32),
        gP: T.Tensor((H, N, 2), f32),
        gB: T.Tensor((H, N, 2), f32),
        gC: T.Tensor((C, H, N, 2), f32),
        L: T.int32,
    ):
        with T.Kernel(H, threads=threads) as h:
            params = (dt, Lam, P, B, Cm)
            terms0 = T.alloc_shared((N, len(rows)), f32)
            block_terms = T.alloc_shared((ceil_div(N, MB), len(rows)), f32) if N > 2 * MB else None
            gd = T.alloc_shared((threads,), f32)  # each thread's terms of dt's gradient
            rest = T.alloc_local((len(rows),), f32)
            r0 = T.alloc_local((len(rows),), f32)
            G = T.alloc_local((R + 2,), f32)
            gv = T.alloc_local((2 * pairs,), f32)
            gw = T.alloc_local((2,), f32)
            tot = T.alloc_var(f32)
            tx = T.get_thread_binding()
            for m0, nc in chunks:
                if tx < nc:
                    stores(terms0, terms(params, h, m0 + tx, C, rows), m0 + tx)
            T.sync_threads()
            if N > 2 * MB:
                block_sums(terms0, block_terms, tx)
                T.sync_threads()
            for m0, nc in chunks:
                n = T.min(m0 + tx, N - 1)
                others(terms0, block_terms, n, rest, r0)
                zero = grad_zero(params, g, rest, r0, h, n, C, L)
                for j in T.unroll(R + 2):
                    G[j] = 0.0
                for s in T.serial(SP):
                    for j in T.unroll(R + 2):
                        G[j] = G[j] + part[h, s, j, n]
                grad_v(params, h, n, C, G, g, L, gv, gw)
                if tx < nc:
                    grads = (gLam, gP, gB, gC)
                    if m0 == 0:
                        gd[tx] = grad_params(params, grads, h, n, C, gv, gw, zero)
                    else:
                        gd[tx] = gd[tx] + grad_params(params, grads, h, n, C, gv, gw, zero)
            T.sync_threads()
            if tx == 0:
                tot = 0.0
                for m in T.serial(chunks[0][1]):
                    tot = tot + gd[m]
                gdt[h] = tot

    return cauchy_finalize
