"""Recurrent scans for inference (mode='recurrent' and stepping).

`s4_kernel` computes S4's bilinear step in O(N) with the factors of `ops._s4.factors`:

    x' = (2/dt - M)^-1 ((2/dt + M) x + 2 B u)
       = lam x - g (w . x) + dB u
    M = Lambda - P P^*

The kernel computes the factors once from Lambda, P, B and dt. Each step then computes w . x' for
the next step and 2 Re(c x') for the output, and neither sum over the modes depends on the other.

A thread block stages the inputs and outputs of STEPS steps in shared memory, so that its global
loads and stores cover consecutive features. The launchers round the mode count Nt up to N, a power
of two of at least 32.
"""

import tilelang.language as T
import torch

from ..runtime import compiled
from .discretization import discretize
from .dsl import cos, exp, f32, sin, var

REG_MODES = 64  # largest mode count for which the parameters are in registers
STEPS = 32

# HIP code that specializes TileLang's AllReduce for the sums over the 32 lanes of a feature.
# TileLang's template makes five shuffles, which go through the LDS unit. This code moves values
# between the 16 lanes of each row with DPP instructions in the VALU, and then one shuffle adds the
# value from the lane 16 away. TileLang places the source before its own includes, so the source
# includes `tl_templates/hip/reduce.h` itself.
LANE_SUMS_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <tl_templates/hip/reduce.h>

template <int ctrl> __device__ __forceinline__ float s4ever_dpp(float x) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), ctrl, 0xf, 0xf, false));
}

template <> template <>
__device__ float tl::AllReduce<tl::SumOp, 32, 1, 0>::run<float>(float x, float *) {
  x += s4ever_dpp<0xb1>(x);   // quad_perm [1, 0, 3, 2]: the neighboring lane
  x += s4ever_dpp<0x4e>(x);   // quad_perm [2, 3, 0, 1]: the other pair of the quad
  x += s4ever_dpp<0x141>(x);  // row_half_mirror: the other quad of the 8 lanes
  x += s4ever_dpp<0x140>(x);  // row_mirror: the other 8 lanes of the row
  return x + __shfl_xor(x, 16);
}
"""


def features(N):
    """Return the features per thread block for N modes, with one warp per feature."""
    return max(1, min(4, 256 // N))


def lanes(buf):
    """Return a layout for a fragment (..., F, N) that puts feature f in warp f.

    Each lane holds runs of up to 4 consecutive modes, and consecutive lanes hold consecutive runs,
    so that a warp reads the modes in vectors without bank conflicts. Without this layout, TileLang
    spreads 128 or more modes over all of a block's threads.
    """
    *lead, F, N = buf.shape
    V = min(N // 32, 4)  # modes per run

    def index(*i):
        flat = 0
        for j, extent in zip(i[:-2], lead, strict=True):
            flat = flat * extent + j
        return (flat * N + i[-1]) // (32 * V) * V + i[-1] % V

    return T.Fragment(
        buf.shape, forward_thread_fn=lambda *i: i[-2] * 32 + i[-1] // V % 32, forward_index_fn=index
    )


@T.macro
def load_readout(Cm, D, use_D, h0, H, Nt, cr, ci, Ds):
    C, F, N = cr.shape
    for c in T.unroll(C):
        for f, n in T.Parallel(F, N):
            h, m = T.min(h0 + f, H - 1), T.min(n, Nt - 1)
            cr[c, f, n] = T.if_then_else(n < Nt, Cm[c, h, m, 0], 0.0)
            ci[c, f, n] = T.if_then_else(n < Nt, Cm[c, h, m, 1], 0.0)
        for f in T.Parallel(F):
            Ds[c, f] = T.if_then_else(use_D != 0, D[c, T.min(h0 + f, H - 1)], 0.0)


@T.macro
def load_state(x0, has_x0, Bsz, b0, h0, H, Nt, xr, xi):
    R, F, N = xr.shape
    for r in T.unroll(R):
        for f, n in T.Parallel(F, N):
            ok = has_x0 != 0 and b0 + r < Bsz and h0 + f < H and n < Nt
            xr[r, f, n] = T.if_then_else(ok, x0[b0 + r, h0 + f, n, 0], 0.0)
            xi[r, f, n] = T.if_then_else(ok, x0[b0 + r, h0 + f, n, 1], 0.0)


@T.macro
def store_state(xN, Bsz, b0, h0, H, Nt, xr, xi):
    R, F, N = xr.shape
    for r in T.unroll(R):
        for f, n in T.Parallel(F, N):
            if b0 + r < Bsz and h0 + f < H and n < Nt:
                xN[b0 + r, h0 + f, n, 0] = xr[r, f, n]
                xN[b0 + r, h0 + f, n, 1] = xi[r, f, n]


def step(t, L, reverse):
    """Return the sequence position of scan step t."""
    return (L - 1 - t) if reverse else t


@T.macro
def load_inputs(u, Us, Bsz, b0, h0, H, t0, L, reverse):
    for r, j, f in T.Parallel(*Us.shape):
        t = t0 + j
        ok = b0 + r < Bsz and h0 + f < H and t < L
        Us[r, j, f] = T.if_then_else(ok, T.cast(u[b0 + r, step(t, L, reverse), h0 + f], f32), 0.0)


@T.macro
def store_outputs(y, Ys, Bsz, b0, h0, H, t0, L, reverse):
    for c, r, j, f in T.Parallel(*Ys.shape):
        t = t0 + j
        if b0 + r < Bsz and h0 + f < H and t < L:
            y[b0 + r, step(t, L, reverse), c, h0 + f] = Ys[c, r, j, f]


@T.macro
def s4_project(r, w, x, terms, sums, wx):
    """Store w . x[r] in wx[r]."""
    F, N = terms.shape
    for f, n in T.Parallel(F, N):
        terms[f, n] = w[0][f, n] * x[0][r, f, n] + w[1][f, n] * x[1][r, f, n]
    T.reduce_sum(terms, sums, dim=1)
    for f in T.Parallel(F):
        wx[r, f] = sums[f]


@T.macro
def s4_step(j, advance, params, x, terms, sums, wx, Us, Ys):
    """Advance the states of all rows by step j and stage the outputs in Ys.

    With `advance`, the macro also computes w . x' for the next step.
    """
    (zr, zi), (gr, gi), w, (qr, qi), (cr, ci), Ds = params
    xr, xi = x
    C, rows, _, F = Ys.shape
    N = terms.shape[1]
    for r in T.unroll(rows):
        for f, n in T.Parallel(F, N):
            a, b = var(xr[r, f, n]), var(xi[r, f, n])
            ur = Us[r, j, f]
            xr[r, f, n] = zr[f, n] * a - zi[f, n] * b - gr[f, n] * wx[r, f] + qr[f, n] * ur
            xi[r, f, n] = zr[f, n] * b + zi[f, n] * a - gi[f, n] * wx[r, f] + qi[f, n] * ur
        if advance:
            s4_project(r, w, x, terms, sums, wx)
        for c in T.unroll(C):
            for f, n in T.Parallel(F, N):
                terms[f, n] = cr[c, f, n] * xr[r, f, n] + ci[c, f, n] * xi[r, f, n]
            T.reduce_sum(terms, sums, dim=1)
            for f in T.Parallel(F):
                Ys[c, r, j, f] = T.cast(sums[f] + Ds[c, f] * Us[r, j, f], Ys.dtype)


@compiled
def s4d_kernel(N, C, dtype, disc, reverse, rows):
    """Build the scan kernel that writes the outputs y and the final state xN for Nt <= N modes.

    A thread block scans `rows` batch rows of F features. Tensors that are not used can have any
    size.
    """
    zoh = disc == "zoh"
    F = features(N)
    param = T.alloc_fragment if N <= REG_MODES else T.alloc_shared
    Bsz, H, L, Nt = (T.dynamic(n) for n in ("B", "H", "L", "N"))
    D0, D1, X0, X1, X2 = (T.dynamic(f"p{i}") for i in range(5))

    @T.prim_func
    def recurrent_s4d(
        u: T.Tensor((Bsz, L, H), dtype),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, Nt, 2), f32),
        Bm: T.Tensor((H, Nt, 2), f32),
        Cm: T.Tensor((C, H, Nt, 2), f32),
        D: T.Tensor((D0, D1), f32),
        x0: T.Tensor((X0, X1, X2, 2), f32),
        y: T.Tensor((Bsz, L, C, H), dtype),
        xN: T.Tensor((Bsz, H, Nt, 2), f32),
        use_D: T.int32,
        has_x0: T.int32,
    ):
        with T.Kernel(T.ceildiv(H, F), T.ceildiv(Bsz, rows), threads=32 * F) as (
            feature_block,
            row_block,
        ):
            if torch.version.hip:
                T.import_source(LANE_SUMS_SOURCE)
            # TileLang 0.1.15 assumes only that sizes are non-negative, so it guards a load at an
            # index clamped to H - 1 or Nt - 1 with a check that the size is at least 1. The sizes
            # are positive, and these assumes remove the guards.
            T.assume(H > 0)
            T.assume(Nt > 0)
            h0, b0 = feature_block * F, row_block * rows
            zr = param((F, N), f32)  # dA
            zi = param((F, N), f32)
            qr = param((F, N), f32)  # dB
            qi = param((F, N), f32)
            cr = param((C, F, N), f32)
            ci = param((C, F, N), f32)
            Ds = T.alloc_fragment((C, F), f32)
            xr = T.alloc_fragment((rows, F, N), f32)
            xi = T.alloc_fragment((rows, F, N), f32)
            T.annotate_layout({xr: lanes(xr), xi: lanes(xi)})
            terms = T.alloc_fragment((F, N), f32)
            sums = T.alloc_fragment((F,), f32)
            for f, n in T.Parallel(F, N):
                h, ok = T.min(h0 + f, H - 1), n < Nt
                lr, li, br, bi = discretize(dt[h], A, Bm, h, T.min(n, Nt - 1), zoh)
                zr[f, n] = T.if_then_else(ok, exp(lr) * cos(li), 0.0)
                zi[f, n] = T.if_then_else(ok, exp(lr) * sin(li), 0.0)
                qr[f, n] = T.if_then_else(ok, br, 0.0)
                qi[f, n] = T.if_then_else(ok, bi, 0.0)
            load_readout(Cm, D, use_D, h0, H, Nt, cr, ci, Ds)
            load_state(x0, has_x0, Bsz, b0, h0, H, Nt, xr, xi)
            Us = T.alloc_shared((rows, STEPS, F), f32)
            Ys = T.alloc_shared((C, rows, STEPS, F), dtype)
            for k in T.serial(T.ceildiv(L, STEPS)):
                t0 = k * STEPS
                load_inputs(u, Us, Bsz, b0, h0, H, t0, L, reverse)
                for j in T.serial(T.min(STEPS, L - t0)):
                    for r in T.unroll(rows):
                        for f, n in T.Parallel(F, N):
                            a, b = var(xr[r, f, n]), var(xi[r, f, n])
                            xr[r, f, n] = zr[f, n] * a - zi[f, n] * b + qr[f, n] * Us[r, j, f]
                            xi[r, f, n] = zr[f, n] * b + zi[f, n] * a + qi[f, n] * Us[r, j, f]
                    for c in T.unroll(C):
                        for r in T.unroll(rows):
                            for f, n in T.Parallel(F, N):
                                terms[f, n] = cr[c, f, n] * xr[r, f, n] - ci[c, f, n] * xi[r, f, n]
                            T.reduce_sum(terms, sums, dim=1)
                            for f in T.Parallel(F):
                                Ys[c, r, j, f] = T.cast(2 * sums[f] + Ds[c, f] * Us[r, j, f], dtype)
                store_outputs(y, Ys, Bsz, b0, h0, H, t0, L, reverse)
            store_state(xN, Bsz, b0, h0, H, Nt, xr, xi)

    return recurrent_s4d


@compiled
def s4_kernel(N, C, dtype, reverse, rows):
    """Build the scan of `s4d_kernel` for S4's Lambda, P and B.

    The output is y = 2 Re(c x) for the readout c, which Cm holds as 2 conj(c).
    """
    F = features(N)
    param = T.alloc_fragment if N <= REG_MODES else T.alloc_shared
    Bsz, H, L, Nt = (T.dynamic(n) for n in ("B", "H", "L", "N"))
    D0, D1, X0, X1, X2 = (T.dynamic(f"p{i}") for i in range(5))

    @T.prim_func
    def recurrent_s4(
        u: T.Tensor((Bsz, L, H), dtype),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, Nt, 2), f32),
        P: T.Tensor((H, Nt, 2), f32),
        Bm: T.Tensor((H, Nt, 2), f32),
        Cm: T.Tensor((C, H, Nt, 2), f32),
        D: T.Tensor((D0, D1), f32),
        x0: T.Tensor((X0, X1, X2, 2), f32),
        y: T.Tensor((Bsz, L, C, H), dtype),
        xN: T.Tensor((Bsz, H, Nt, 2), f32),
        use_D: T.int32,
        has_x0: T.int32,
    ):
        with T.Kernel(T.ceildiv(H, F), T.ceildiv(Bsz, rows), threads=32 * F) as (
            feature_block,
            row_block,
        ):
            if torch.version.hip:
                T.import_source(LANE_SUMS_SOURCE)
            T.assume(H > 0)
            T.assume(Nt > 0)
            h0, b0 = feature_block * F, row_block * rows
            zr = param((F, N), f32)  # Lambda, then lam
            zi = param((F, N), f32)
            gr = param((F, N), f32)  # P, then g
            gi = param((F, N), f32)
            wr = param((F, N), f32)
            wi = param((F, N), f32)
            qr = param((F, N), f32)  # B, then dB
            qi = param((F, N), f32)
            cr = param((C, F, N), f32)
            ci = param((C, F, N), f32)
            Ds = T.alloc_fragment((C, F), f32)
            dts = T.alloc_fragment((F,), f32)
            xr = T.alloc_fragment((rows, F, N), f32)
            xi = T.alloc_fragment((rows, F, N), f32)
            T.annotate_layout({xr: lanes(xr), xi: lanes(xi)})
            terms = T.alloc_fragment((F, N), f32)
            sums = T.alloc_fragment((F,), f32)
            terms_b = T.alloc_fragment((F, N), f32)
            sums_b = T.alloc_fragment((F,), f32)
            wx = T.alloc_fragment((rows, F), f32)  # w . x
            # The kernel loads Lambda, P and B before any arithmetic, so that every load is issued
            # before the first instruction that uses a loaded value.
            for f in T.Parallel(F):
                dts[f] = dt[T.min(h0 + f, H - 1)]
            for f, n in T.Parallel(F, N):
                h, m, ok = T.min(h0 + f, H - 1), T.min(n, Nt - 1), n < Nt
                zr[f, n] = T.Select(ok, A[h, m, 0], 0.0)
                zi[f, n] = T.Select(ok, A[h, m, 1], 0.0)
                gr[f, n] = T.Select(ok, P[h, m, 0], 0.0)
                gi[f, n] = T.Select(ok, P[h, m, 1], 0.0)
                qr[f, n] = T.Select(ok, Bm[h, m, 0], 0.0)
                qi[f, n] = T.Select(ok, Bm[h, m, 1], 0.0)
            load_readout(Cm, D, use_D, h0, H, Nt, cr, ci, Ds)
            load_state(x0, has_x0, Bsz, b0, h0, H, Nt, xr, xi)
            for f, n in T.Parallel(F, N):
                pr, pi, br, bi = var(gr[f, n]), var(gi[f, n]), var(qr[f, n]), var(qi[f, n])
                # a = dt Lambda / 2 and E = (1 - a)^-1, so that (2/dt - Lambda)^-1 = E dt / 2
                ar, ai = var(dts[f] / 2 * zr[f, n]), var(dts[f] / 2 * zi[f, n])
                den = var((1 - ar) * (1 - ar) + ai * ai)
                er, ei = var((1 - ar) / den), var(ai / den)
                # lam = 1 + 2 a E = -1 + 2 E. The form that adds the smaller term to +-1, which is
                # the first form for |a| < 1, has the smaller rounding error, and that error
                # compounds over the steps.
                small = ar * ar + ai * ai < 1
                zr[f, n] = T.Select(small, 1 + 2 * (ar * er - ai * ei), 2 * er - 1)
                zi[f, n] = T.Select(small, 2 * (ar * ei + ai * er), 2 * ei)
                # w = dt conj(E) P
                vr, vi = var(dts[f] * (pr * er + pi * ei)), var(dts[f] * (pi * er - pr * ei))
                wr[f, n] = vr
                wi[f, n] = vi
                gr[f, n] = er * pr - ei * pi  # E P
                gi[f, n] = er * pi + ei * pr
                qr[f, n] = er * br - ei * bi  # E B
                qi[f, n] = er * bi + ei * br
                terms[f, n] = vr * pr + vi * pi
                terms_b[f, n] = vr * br + vi * bi
            T.reduce_sum(terms, sums, dim=1)  # w . P
            T.reduce_sum(terms_b, sums_b, dim=1)  # w . B
            for f in T.Parallel(F):
                sums[f] = 1 / (1 + sums[f])
            for f, n in T.Parallel(F, N):
                # g = 2 E P / (1 + w . P) and dB = dt (E B - E P (w . B) / (1 + w . P))
                qr[f, n] = dts[f] * (qr[f, n] - gr[f, n] * sums_b[f] * sums[f])
                qi[f, n] = dts[f] * (qi[f, n] - gi[f, n] * sums_b[f] * sums[f])
                gr[f, n] = 2 * sums[f] * gr[f, n]
                gi[f, n] = 2 * sums[f] * gi[f, n]
            for r in T.unroll(rows):
                s4_project(r, (wr, wi), (xr, xi), terms, sums, wx)
            Us = T.alloc_shared((rows, STEPS, F), f32)
            Ys = T.alloc_shared((C, rows, STEPS, F), dtype)
            params = ((zr, zi), (gr, gi), (wr, wi), (qr, qi), (cr, ci), Ds)
            for k in T.serial(T.ceildiv(L, STEPS)):
                t0 = k * STEPS
                load_inputs(u, Us, Bsz, b0, h0, H, t0, L, reverse)
                # The last step does not compute w . x', because no step follows.
                for j in T.serial(T.min(STEPS, L - 1 - t0)):
                    s4_step(j, True, params, (xr, xi), terms, sums, wx, Us, Ys)
                if L - t0 <= STEPS:
                    s4_step(L - 1 - t0, False, params, (xr, xi), terms, sums, wx, Us, Ys)
                store_outputs(y, Ys, Bsz, b0, h0, H, t0, L, reverse)
            store_state(xN, Bsz, b0, h0, H, Nt, xr, xi)

    return recurrent_s4
