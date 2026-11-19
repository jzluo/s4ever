"""Chunked scans of diagonal SSMs on tensor cores (S4D's mode='chunk').

The kernels cut the sequence into chunks of Tc steps. Within a chunk, the outputs and the state
update are products with per-feature matrices, which the kernels build in their prologue:

    y = Kt u + Out x
    x <- z^Tc x + In u
    Out[i] = 2 C z^(i+1)
    In[:, j] = z^(Tc-1-j) dB

Kt is the causal Toeplitz matrix of the first Tc kernel taps, with D added to its diagonal.

The states are stored in blocks of W real parts followed by the W matching imaginary parts, where W
is `layout.CHUNK_COLUMN_STRIDE`. Columns that are W apart are in the same thread of a tensor-core
accumulator, so both parts of a mode are in the same thread. The factory argument `precision` is
the number of TF32 passes of a float32 product (`runtime.precision`).
"""

import tilelang.language as T
import torch

from ..layout import CHUNK_COLUMN_STRIDE as W
from ..layout import CHUNK_ROWS as ROWS
from ..runtime import compiled
from .discretization import discretize, discretize_grad
from .dsl import cos, exp, f32, int_var, sin, store_tf32, var


def mode(s):
    """Return the mode whose real or imaginary part is in state column s."""
    return s // (2 * W) * W + s % W


def part(s):
    return s % (2 * W) // W


def real(s):
    return s % (2 * W) < W


def slot(n):
    """Return the state column of the real part of mode n."""
    return n // W * 2 * W + n % W


def partner(s):
    """Return the state column that holds the other part of column s's mode."""
    return s // (2 * W) * 2 * W + (s + W) % (2 * W)


def threads(S):
    """Return the thread count of a block with S state columns.

    A block has one warp per 16 columns, up to 4 warps.
    """
    return 32 * min(4, S // 16)


def gemm3(lhs, rhs, out, lhs_lo, rhs_lo, clear=False, **kw):
    """Add the product of lhs and rhs to out, where each operand is a pair of TF32 halves (hi, lo).

    The product of the high halves is always added. With `lhs_lo`, the product of lhs's low half
    and rhs's high half is added as well, and with `rhs_lo`, the product of lhs's high half and
    rhs's low half is added. With `clear`, the first product overwrites out. Other keyword
    arguments go to T.gemm.
    """
    terms = [(lhs[1], rhs[0])] * lhs_lo + [(lhs[0], rhs[1])] * rhs_lo + [(lhs[0], rhs[0])]
    for i, (a, b) in enumerate(terms):
        T.evaluate(T.gemm(a, b, out, clear_accum=clear and i == 0, **kw))


@T.macro
def discretized(h, n0, Nt, dt, A, Bm, zoh, ws, dBs):
    """Store log dA in ws and dB in dBs for the modes [n0, n0 + N), and zeros from mode Nt on."""
    for n in T.Parallel(ws.shape[0]):
        m = T.min(n0 + n, Nt - 1)
        ok = n0 + n < Nt
        lr, li, br, bi = discretize(dt[h], A, Bm, h, m, zoh)
        ws[n, 0] = T.if_then_else(ok, lr, 0.0)
        ws[n, 1] = T.if_then_else(ok, li, 0.0)
        dBs[n, 0] = T.if_then_else(ok, br, 0.0)
        dBs[n, 1] = T.if_then_else(ok, bi, 0.0)


@T.macro
def powers(ws, P):
    """Store z^e in P[e] for the z = exp(ws) of each mode."""
    for e, n in T.Parallel(P.shape[0], ws.shape[0]):
        mag = exp(T.cast(e, f32) * ws[n, 0])
        P[e, n, 0] = mag * cos(T.cast(e, f32) * ws[n, 1])
        P[e, n, 1] = mag * sin(T.cast(e, f32) * ws[n, 1])


@T.macro
def add_param_grads(sum_row, h, n0, Nt, n, dt, A, Bm, zoh, gw, gb, gq, dA, dB):
    """Add the gradients of mode n0 + n's A and B to row `sum_row` of dA and dB.

    The gradients come from gw and gb, the gradients of log dA and dB. The macro stores the mode's
    term of dt's gradient in gq[n].
    """
    m = T.min(n0 + n, Nt - 1)
    Gar, Gai, Gbr, Gbi, Gdt = discretize_grad(dt[h], A, Bm, h, m, gw, gb, zoh)
    gq[n] = T.if_then_else(n0 + n < Nt, Gdt, 0.0)
    if n0 + n < Nt:
        dA[sum_row, n0 + n, 0] = dA[sum_row, n0 + n, 0] + Gar
        dA[sum_row, n0 + n, 1] = dA[sum_row, n0 + n, 1] + Gai
        dB[sum_row, n0 + n, 0] = dB[sum_row, n0 + n, 0] + Gbr
        dB[sum_row, n0 + n, 1] = dB[sum_row, n0 + n, 1] + Gbi


@T.macro
def add_dt_grad(sum_row, gq, ddt):
    """Add the modes' terms gq of dt's gradient to ddt[sum_row].

    One warp adds the at most 32 terms, in the same order in every run.
    """
    tx = T.get_thread_binding()
    if tx < 32:
        N = gq.shape[0]
        tot = T.warp_reduce_sum(T.Select(tx < N, gq[T.min(tx, N - 1)], 0.0))
        if tx == 0:
            ddt[sum_row] = ddt[sum_row] + tot


def macros(Tc, N, C, reverse, precision, disc):
    """Build the prologue and the parameter gradients of `forward_kernel` and `backward_kernel`."""
    S, CT = 2 * N, C * Tc  # state columns, rows of Kt and Out
    split = precision == 3  # whether float32 operands are stored as two TF32 halves
    zoh = disc == "zoh"

    @T.macro
    def setup(h, n0, Nt, c0, use_D, dt, A, Bm, Cm, D, dBs, Cs, Ws, Ds, P):
        """Store dB, the block's channels of Cm and D, the powers P of z, and Ws = C dB."""
        ws = T.alloc_shared((N, 2), f32)
        discretized(h, n0, Nt, dt, A, Bm, zoh, ws, dBs)
        for cc, n, r in T.Parallel(C, N, 2):
            m = T.min(n0 + n, Nt - 1)
            Cs[cc, n, r] = T.if_then_else(n0 + n < Nt, Cm[c0 + cc, h, m, r], 0.0)
        for cc in T.Parallel(C):
            Ds[cc] = T.if_then_else(use_D != 0, D[cc, h], 0.0)
        powers(ws, P)
        for cc, n in T.Parallel(C, N):
            Ws[cc, n, 0] = Cs[cc, n, 0] * dBs[n, 0] - Cs[cc, n, 1] * dBs[n, 1]
            Ws[cc, n, 1] = Cs[cc, n, 0] * dBs[n, 1] + Cs[cc, n, 1] * dBs[n, 0]

    @T.macro
    def matrices(P, Ws, Cs, dBs, Ds, Kh, Kl, Oh, Ol, Ih, Il):
        """Store the TF32 halves of Kt, Out and In."""
        Kp = T.alloc_fragment((CT, N), f32)
        Kr = T.alloc_fragment((CT,), f32)
        Ks = T.alloc_shared((CT,), f32)  # the kernel tap r % Tc of channel r // Tc
        for r, n in T.Parallel(CT, N):
            e = r % Tc
            Kp[r, n] = 2 * (Ws[r // Tc, n, 0] * P[e, n, 0] - Ws[r // Tc, n, 1] * P[e, n, 1])
        T.reduce_sum(Kp, Kr, dim=1)
        T.copy(Kr, Ks)
        for r, j in T.Parallel(CT, Tc):
            d = (j - r % Tc) if reverse else (r % Tc - j)
            v = T.if_then_else(d >= 0, Ks[r // Tc * Tc + T.max(d, 0)], 0.0) + T.if_then_else(
                d == 0, Ds[r // Tc], 0.0
            )
            store_tf32(Kh, Kl, v, split, r, j)
        for r, s in T.Parallel(CT, S):
            e = (Tc - r % Tc) if reverse else (r % Tc + 1)
            n = mode(s)
            cr = Cs[r // Tc, n, 0]
            ci = Cs[r // Tc, n, 1]
            v = T.if_then_else(
                real(s),
                2 * (cr * P[e, n, 0] - ci * P[e, n, 1]),
                -2 * (cr * P[e, n, 1] + ci * P[e, n, 0]),
            )
            store_tf32(Oh, Ol, v, split, r, s)
        for s, j in T.Parallel(S, Tc):
            e = j if reverse else Tc - 1 - j
            n = mode(s)
            v = T.if_then_else(
                real(s),
                P[e, n, 0] * dBs[n, 0] - P[e, n, 1] * dBs[n, 1],
                P[e, n, 0] * dBs[n, 1] + P[e, n, 1] * dBs[n, 0],
            )
            store_tf32(Ih, Il, v, split, s, j)

    @T.macro
    def transition(P, zs, conj):
        """Store zs such that zs[0] x + zs[1] partner(x) is z^Tc x, or conj(z)^Tc x with `conj`.

        Here x is a state, and partner(x) is x with each column swapped with its partner column.
        """
        for s in T.Parallel(S):
            zs[0, s] = P[Tc, mode(s), 0]
            if conj:
                zs[1, s] = T.if_then_else(real(s), P[Tc, mode(s), 1], -P[Tc, mode(s), 1])
            else:
                zs[1, s] = T.if_then_else(real(s), -P[Tc, mode(s), 1], P[Tc, mode(s), 1])

    def prologue(h, n0, Nt, c0, use_D, dt, A, Bm, Cm, D, conj):
        """Return the TF32 halves of Kt, Out and In, the tables of `setup`, and zs."""
        Kh = T.alloc_fragment((CT, Tc), f32)
        Kl = T.alloc_fragment((CT, Tc), f32)
        Oh = T.alloc_fragment((CT, S), f32)
        Ol = T.alloc_fragment((CT, S), f32)
        Ih = T.alloc_fragment((S, Tc), f32)
        Il = T.alloc_fragment((S, Tc), f32)
        dBs = T.alloc_shared((N, 2), f32)
        Cs = T.alloc_shared((C, N, 2), f32)
        Ws = T.alloc_shared((C, N, 2), f32)  # C dB
        Ds = T.alloc_shared((C,), f32)
        P = T.alloc_shared((Tc + 1, N, 2), f32)  # z^e
        zs = T.alloc_shared((2, S), f32)
        setup(h, n0, Nt, c0, use_D, dt, A, Bm, Cm, D, dBs, Cs, Ws, Ds, P)
        matrices(P, Ws, Cs, dBs, Ds, Kh, Kl, Oh, Ol, Ih, Il)
        transition(P, zs, conj)
        return Kh, Kl, Oh, Ol, Ih, Il, dBs, Cs, Ws, P, zs

    @T.macro
    def param_grads(sum_row, c_offset, h, n0, Nt, c0, use_D, params, grads, tables, sums):
        """Add the gradients of dt, A, B, C and D to the batch block's sums.

        The block adds to row `sum_row` of ddt, dA and dB, to the same column of dD and to the rows
        from `c_offset` on of dC. `grads` holds the gradients aK, aO and aI of Kt, Out and In and
        the gradients Z1 and Z2 of zs.
        """
        dt, A, Bm = params
        aK, aO, aI, Z1, Z2 = grads
        P, Cs, Ws, dBs = tables
        ddt, dA, dB, dC, dD = sums
        gKt = T.alloc_shared((CT, Tc), f32)
        gO = T.alloc_shared((CT, S), f32)
        gI = T.alloc_shared((S, Tc), f32)
        gK = T.alloc_shared((C, Tc), f32)
        gz = T.alloc_shared((2, S), f32)
        gp = T.alloc_shared((N, 4), f32)  # gradients of log dA and dB
        gq = T.alloc_shared((N,), f32)
        rz = T.alloc_fragment((S,), f32)
        T.copy(aK, gKt)
        T.copy(aO, gO)
        T.copy(aI, gI)
        T.reduce_sum(Z1, rz, dim=0)
        T.copy(rz, gz[0, :])
        T.reduce_sum(Z2, rz, dim=0)
        T.copy(rz, gz[1, :])
        for cc, d in T.Parallel(C, Tc):
            acc = T.alloc_var(f32, init=0.0)
            for i in T.serial(Tc - d):
                if reverse:
                    acc = acc + gKt[cc * Tc + i, i + d]
                else:
                    acc = acc + gKt[cc * Tc + i + d, i]
            gK[cc, d] = acc
        if use_D != 0:
            for cc in T.Parallel(C):
                dD[cc, sum_row] = dD[cc, sum_row] + gK[cc, 0]
        for n in T.Parallel(N):
            gwr = T.alloc_var(f32, init=0.0)
            gwi = T.alloc_var(f32, init=0.0)
            gbr = T.alloc_var(f32, init=0.0)
            gbi = T.alloc_var(f32, init=0.0)
            gWr = T.alloc_var(f32, init=0.0)
            gWi = T.alloc_var(f32, init=0.0)
            gcr = T.alloc_var(f32, init=0.0)
            gci = T.alloc_var(f32, init=0.0)
            sr = slot(n)
            br = dBs[n, 0]
            bi = dBs[n, 1]
            for cc in T.serial(C):
                gWr = 0.0
                gWi = 0.0
                gcr = 0.0
                gci = 0.0
                # The taps K[e] = 2 Re(W z^e) give G_W += g conj(z^e) and
                # G_log z += e g conj(W z^e) with g = 2 G_K[e].
                for e in T.serial(Tc):
                    g = 2 * gK[cc, e]
                    ef = T.cast(e, f32)
                    gWr = gWr + g * P[e, n, 0]
                    gWi = gWi - g * P[e, n, 1]
                    gwr = gwr + ef * g * (P[e, n, 0] * Ws[cc, n, 0] - P[e, n, 1] * Ws[cc, n, 1])
                    gwi = gwi - ef * g * (P[e, n, 0] * Ws[cc, n, 1] + P[e, n, 1] * Ws[cc, n, 0])
                # The columns of Out hold 2 Re(a) and -2 Im(a) for a = C z^e. With v = G_a, they
                # give G_C += v conj(z^e) and G_log z += e v conj(C z^e).
                for i in T.serial(Tc):
                    e = (Tc - i) if reverse else (i + 1)
                    ef = T.cast(e, f32)
                    vr = 2 * gO[cc * Tc + i, sr]
                    vi = -2 * gO[cc * Tc + i, sr + W]
                    gcr = gcr + vr * P[e, n, 0] + vi * P[e, n, 1]
                    gci = gci + vi * P[e, n, 0] - vr * P[e, n, 1]
                    qr = vr * Cs[cc, n, 0] + vi * Cs[cc, n, 1]
                    qi = vi * Cs[cc, n, 0] - vr * Cs[cc, n, 1]
                    gwr = gwr + ef * (P[e, n, 0] * qr + P[e, n, 1] * qi)
                    gwi = gwi + ef * (P[e, n, 0] * qi - P[e, n, 1] * qr)
                if n0 + n < Nt:
                    gr = gcr + gWr * br + gWi * bi
                    gi = gci + gWi * br - gWr * bi
                    dC[c_offset + c0 + cc, h, n0 + n, 0] = dC[c_offset + c0 + cc, h, n0 + n, 0] + gr
                    dC[c_offset + c0 + cc, h, n0 + n, 1] = dC[c_offset + c0 + cc, h, n0 + n, 1] + gi
                gbr = gbr + gWr * Cs[cc, n, 0] + gWi * Cs[cc, n, 1]
                gbi = gbi + gWi * Cs[cc, n, 0] - gWr * Cs[cc, n, 1]
            # The columns of In hold a = z^e dB. With v = G_a, they give G_dB += v conj(z^e) and
            # G_log z += e v conj(z^e dB).
            for j in T.serial(Tc):
                e = j if reverse else Tc - 1 - j
                ef = T.cast(e, f32)
                vr = gI[sr, j]
                vi = gI[sr + W, j]
                gbr = gbr + vr * P[e, n, 0] + vi * P[e, n, 1]
                gbi = gbi + vi * P[e, n, 0] - vr * P[e, n, 1]
                qr = vr * br + vi * bi
                qi = vi * br - vr * bi
                gwr = gwr + ef * (P[e, n, 0] * qr + P[e, n, 1] * qi)
                gwi = gwi + ef * (P[e, n, 0] * qi - P[e, n, 1] * qr)
            # The zs of `forward_kernel` hold Re z^Tc in both columns of the mode, and -Im z^Tc and
            # Im z^Tc in its real and imaginary columns. zr and zi are the gradients of Re z^Tc and
            # Im z^Tc.
            zr = gz[0, sr] + gz[0, sr + W]
            zi = gz[1, sr + W] - gz[1, sr]
            gp[n, 0] = gwr + Tc * (P[Tc, n, 0] * zr + P[Tc, n, 1] * zi)
            gp[n, 1] = gwi + Tc * (P[Tc, n, 0] * zi - P[Tc, n, 1] * zr)
            gp[n, 2] = gbr
            gp[n, 3] = gbi
        for n in T.Parallel(N):
            gw, gb = (gp[n, 0], gp[n, 1]), (gp[n, 2], gp[n, 3])
            add_param_grads(sum_row, h, n0, Nt, n, dt, A, Bm, zoh, gw, gb, gq, dA, dB)
        add_dt_grad(sum_row, gq, ddt)

    return prologue, param_grads


@compiled
def forward_kernel(Tc, N, C, dtype, precision, reverse, disc, grouped, save, ydtype):
    """Build the kernel that scans u (B, H, L) for the modes [n0, n0 + N) of A and B.

    The scan reads the channels [c0, c0 + C) of Cm, and a thread block scans ROWS batch rows of one
    feature. Tensors that are not used can have any size.

    Args:
        grouped: whether the kernel writes its outputs to the channels [c0, c0 + C) of y.
        save: whether the kernel stores the states at the starts of the chunks in Xs.

    Kernel arguments:
        acc: whether the kernel adds its outputs to the values in y.
        final: whether the kernel stores the final state in xN.
    """
    S, CT = 2 * N, C * Tc
    split = precision == 3
    # Values in bf16 and fp16 are exact in TF32, so u is split into two halves only in float32, as
    # are u and gy in `backward_kernel`.
    split_u = split and dtype == f32
    prologue, _ = macros(Tc, N, C, reverse, precision, disc)
    B, H, L, Nt, Ct, Sx = (T.dynamic(n) for n in ("B", "H", "L", "N", "Ct", "Sx"))
    Cy = T.dynamic("Cy") if grouped else C
    D0, D1, X0, X1, X2, F0, F1, F2 = (T.dynamic(f"p{i}") for i in range(8))
    nC = T.ceildiv(L, Tc)

    @T.macro
    def load(u, Un, b0, h, start):
        """Load the inputs at the steps [start, start + Tc) into Un from clamped indices."""
        for b, j in T.Parallel(ROWS, Tc):
            Un[b, j] = u[T.min(b0 + b, B - 1), h, T.clamp(start + j, 0, L - 1)]

    @T.macro
    def stage(Un, X, Uh, Ul, Xh, Xl, p, b0, start):
        """Store the TF32 halves of the inputs Un and of the state X in buffer p.

        The inputs of rows and steps out of range are zero.
        """
        for b, j in T.Parallel(ROWS, Tc):
            t = start + j
            v = T.Select(b0 + b < B and t >= 0 and t < L, T.cast(Un[b, j], f32), 0.0)
            store_tf32(Uh, Ul, v, split_u, p, b, j)
        for b, s in T.Parallel(ROWS, S):
            store_tf32(Xh, Xl, X[b, s], split, p, b, s)

    @T.prim_func
    def chunk_forward(
        u: T.Tensor((B, H, L), dtype),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, Nt, 2), f32),
        Bm: T.Tensor((H, Nt, 2), f32),
        Cm: T.Tensor((Ct, H, Nt, 2), f32),
        D: T.Tensor((D0, D1), f32),
        x0: T.Tensor((X0, X1, X2, 2), f32),
        y: T.Tensor((B, Cy, H, L), ydtype),
        xN: T.Tensor((F0, F1, F2, 2), f32),
        Xs: T.Tensor((B, H, nC, Sx) if save else (1, 1, 1, 1), f32),
        n0: T.int32,
        c0: T.int32,
        acc: T.int32,
        use_D: T.int32,
        has_x0: T.int32,
        final: T.int32,
    ):
        with T.Kernel(T.ceildiv(B, ROWS), H, threads=threads(S)) as (row_block, h):
            # TileLang 0.1.15 assumes only that sizes are non-negative, so it guards a load at an
            # index clamped to B - 1 or Nt - 1 with a check that the size is at least 1. The sizes
            # are positive, and these assumes remove the guards.
            T.assume(B > 0)
            T.assume(Nt > 0)
            if save and torch.version.hip:
                # These assumes remove TileLang's bounds checks on the accesses of Xs. On AMD the
                # checks are branches, and after each branch the loads wait for all pending memory
                # operations. On NVIDIA the checks are predicates, which do not delay the loads, so
                # the kernel adds the assumes only on AMD.
                T.assume(n0 >= 0)
                T.assume(2 * n0 + S <= Sx)
            # Without `grouped`, the channel offset of y is the constant 0, because nvcc spills
            # registers in these kernels on sm_100 and sm_120 when the offset is c0, a run-time
            # value (`tests/lowering.py` reports the spills).
            yc0 = c0 if grouped else 0
            Kh, Kl, Oh, Ol, Ih, Il, dBs, Cs, Ws, P, zs = prologue(
                h, n0, Nt, c0, use_D, dt, A, Bm, Cm, D, False
            )

            Un = T.alloc_fragment((ROWS, Tc), dtype)
            Uh = T.alloc_shared((2, ROWS, Tc), f32)
            Ul = T.alloc_shared((2, ROWS, Tc), f32)
            Xh = T.alloc_shared((2, ROWS, S), f32)
            Xl = T.alloc_shared((2, ROWS, S), f32)
            X = T.alloc_fragment((ROWS, S), f32)
            Xn = T.alloc_fragment((ROWS, S), f32)
            Xp = T.alloc_fragment((ROWS, S), f32)
            Yu = T.alloc_fragment((ROWS, CT), f32)  # Kt u
            Yx = T.alloc_fragment((ROWS, CT), f32)  # Out x
            b0 = row_block * ROWS

            for b, s in T.Parallel(ROWS, S):
                ok = has_x0 != 0 and b0 + b < B and n0 + mode(s) < Nt
                X[b, s] = T.if_then_else(
                    ok, x0[T.min(b0 + b, B - 1), h, T.min(n0 + mode(s), Nt - 1), part(s)], 0.0
                )
            load(u, Un, b0, h, (L - Tc) if reverse else 0)
            stage(Un, X, Uh, Ul, Xh, Xl, 0, b0, (L - Tc) if reverse else 0)
            for k in T.serial(nC):
                p = k % 2
                t0 = (L - (k + 1) * Tc) if reverse else k * Tc
                # The kernel loads and stages the next chunk's inputs even at the last chunk,
                # because on AMD a load under `k + 1 < nC` would wait at every chunk for all
                # pending stores.
                t1 = (t0 - Tc) if reverse else (t0 + Tc)
                load(u, Un, b0, h, t1)
                if save:
                    for b, s in T.Parallel(ROWS, S):
                        if b0 + b < B:
                            Xs[b0 + b, h, k, 2 * n0 + s] = X[b, s]
                for b, s in T.Parallel(ROWS, S):
                    Xp[b, s] = X[b, partner(s)]
                for b, s in T.Parallel(ROWS, S):
                    Xn[b, s] = zs[0, s] * X[b, s] + zs[1, s] * Xp[b, s]
                u_halves, x_halves = (Uh[p, :, :], Ul[p, :, :]), (Xh[p, :, :], Xl[p, :, :])
                gemm3(u_halves, (Ih, Il), Xn, split_u, split, transpose_B=True)
                gemm3(u_halves, (Kh, Kl), Yu, split_u, split, clear=True, transpose_B=True)
                gemm3(x_halves, (Oh, Ol), Yx, split, split, clear=True, transpose_B=True)
                if acc != 0:
                    for b, r in T.Parallel(ROWS, CT):
                        t = T.clamp(t0 + r % Tc, 0, L - 1)
                        Yu[b, r] = Yu[b, r] + T.cast(
                            y[T.min(b0 + b, B - 1), yc0 + r // Tc, h, t], f32
                        )
                for b, r in T.Parallel(ROWS, CT):
                    t = t0 + r % Tc
                    if b0 + b < B and t >= 0 and t < L:
                        y[b0 + b, yc0 + r // Tc, h, t] = T.cast(Yu[b, r] + Yx[b, r], ydtype)
                T.copy(Xn, X)
                stage(Un, X, Uh, Ul, Xh, Xl, 1 - p, b0, t1)
            if final != 0:
                # X has been advanced over all Tc steps of the last chunk, so xN advances the state
                # staged at the start of that chunk over only the chunk's `last` valid steps.
                last = L - (nC - 1) * Tc
                p_last = (nC - 1) % 2  # the staging buffer of the last chunk
                for b, n in T.Parallel(ROWS, N):
                    sr = slot(n)
                    ur = T.alloc_var(f32, init=0.0)
                    ui = T.alloc_var(f32, init=0.0)
                    for i in T.serial(last):
                        j = (Tc - last + i) if reverse else i
                        e = i if reverse else last - 1 - i
                        uj = (Uh[p_last, b, j] + Ul[p_last, b, j]) if split_u else Uh[p_last, b, j]
                        ur = ur + P[e, n, 0] * uj
                        ui = ui + P[e, n, 1] * uj
                    xr = (Xh[p_last, b, sr] + Xl[p_last, b, sr]) if split else Xh[p_last, b, sr]
                    xi = (
                        (Xh[p_last, b, sr + W] + Xl[p_last, b, sr + W])
                        if split
                        else Xh[p_last, b, sr + W]
                    )
                    if b0 + b < B and n0 + n < Nt:
                        xN[b0 + b, h, n0 + n, 0] = (
                            P[last, n, 0] * xr
                            - P[last, n, 1] * xi
                            + dBs[n, 0] * ur
                            - dBs[n, 1] * ui
                        )
                        xN[b0 + b, h, n0 + n, 1] = (
                            P[last, n, 0] * xi
                            + P[last, n, 1] * xr
                            + dBs[n, 0] * ui
                            + dBs[n, 1] * ur
                        )

    return chunk_forward


@compiled
def backward_kernel(Tc, N, C, dtype, precision, reverse, disc, grouped, gudtype):
    """Build the kernel that computes the gradients of `forward_kernel`.

    Args:
        grouped: whether the kernel reads the channels [c0, c0 + C) of gy.

    Kernel arguments:
        acc: whether the kernel adds the gradient of u to the values in gu.
        ddt, dA, dB, dC, dD: the partial sums for `sums_kernel`. Block b of batch rows adds its
            gradients of dt, A and B to the rows [b H, (b + 1) H) of ddt, dA and dB. It adds its
            gradient of D to the same columns of dD and its gradient of C to the rows
            [b Ct, (b + 1) Ct) of dC.
    """
    S, CT = 2 * N, C * Tc
    split = precision == 3
    split_u = split and dtype == f32
    prologue, param_grads = macros(Tc, N, C, reverse, precision, disc)
    B, H, L, Nt, Ct, Sx = (T.dynamic(n) for n in ("B", "H", "L", "N", "Ct", "Sx"))
    Cy = T.dynamic("Cy") if grouped else C
    D0, D1, G0, G1, G2, X0, X1, X2 = (T.dynamic(f"p{i}") for i in range(8))
    nC, NB = T.ceildiv(L, Tc), T.ceildiv(B, ROWS)

    @T.prim_func
    def chunk_backward(
        u: T.Tensor((B, H, L), dtype),
        gy: T.Tensor((B, Cy, H, L), dtype),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, Nt, 2), f32),
        Bm: T.Tensor((H, Nt, 2), f32),
        Cm: T.Tensor((Ct, H, Nt, 2), f32),
        D: T.Tensor((D0, D1), f32),
        Xs: T.Tensor((B, H, nC, Sx), f32),
        g_last: T.Tensor((G0, G1, G2, 2), f32),
        gu: T.Tensor((B, H, L), gudtype),
        dx0: T.Tensor((X0, X1, X2, 2), f32),
        ddt: T.Tensor((NB * H,), f32),
        dA: T.Tensor((NB * H, Nt, 2), f32),
        dB: T.Tensor((NB * H, Nt, 2), f32),
        dC: T.Tensor((NB * Ct, H, Nt, 2), f32),
        dD: T.Tensor((D0, NB * D1), f32),
        n0: T.int32,
        c0: T.int32,
        acc: T.int32,
        use_D: T.int32,
        has_g_last: T.int32,
        want_dx0: T.int32,
    ):
        with T.Kernel(T.ceildiv(B, ROWS), H, threads=threads(S)) as (row_block, h):
            T.assume(B > 0)
            T.assume(Nt > 0)
            if torch.version.hip:
                T.assume(n0 >= 0)
                T.assume(2 * n0 + S <= Sx)
            yc0 = c0 if grouped else 0
            Kh, Kl, Oh, Ol, Ih, Il, dBs, Cs, Ws, P, zs = prologue(
                h, n0, Nt, c0, use_D, dt, A, Bm, Cm, D, True
            )

            Gh = T.alloc_shared((ROWS, CT), f32)
            Gl = T.alloc_shared((ROWS, CT), f32)
            Uh = T.alloc_shared((ROWS, Tc), f32)
            Ul = T.alloc_shared((ROWS, Tc), f32)
            Xh = T.alloc_shared((ROWS, S), f32)
            Xl = T.alloc_shared((ROWS, S), f32)
            Mh = T.alloc_shared((ROWS, S), f32)
            Ml = T.alloc_shared((ROWS, S), f32)
            Mu = T.alloc_fragment((ROWS, S), f32)  # adjoint of the state after the chunk
            Mn = T.alloc_fragment((ROWS, S), f32)  # adjoint of the state before it
            Mp = T.alloc_fragment((ROWS, S), f32)
            X = T.alloc_fragment((ROWS, S), f32)
            Xp = T.alloc_fragment((ROWS, S), f32)
            Z1 = T.alloc_fragment((ROWS, S), f32)
            Z2 = T.alloc_fragment((ROWS, S), f32)
            Gu = T.alloc_fragment((ROWS, Tc), f32)
            Gm = T.alloc_fragment((ROWS, Tc), f32)
            aK = T.alloc_fragment((CT, Tc), f32)
            aO = T.alloc_fragment((CT, S), f32)
            aI = T.alloc_fragment((S, Tc), f32)
            b0 = row_block * ROWS

            T.clear(Mu)
            T.clear(aK)
            T.clear(aO)
            T.clear(aI)
            T.clear(Z1)
            T.clear(Z2)
            for kk in T.serial(nC):
                k = nC - 1 - kk
                t0 = (L - (k + 1) * Tc) if reverse else k * Tc
                for b, r in T.Parallel(ROWS, CT):
                    t = t0 + r % Tc
                    v = var(
                        T.cast(
                            gy[T.min(b0 + b, B - 1), yc0 + r // Tc, h, T.clamp(t, 0, L - 1)], f32
                        )
                    )
                    g = T.Select(b0 + b < B and t >= 0 and t < L, v, 0.0)
                    store_tf32(Gh, Gl, g, split_u, b, r)
                for b, j in T.Parallel(ROWS, Tc):
                    t = t0 + j
                    v = var(T.cast(u[T.min(b0 + b, B - 1), h, T.clamp(t, 0, L - 1)], f32))
                    uj = T.Select(b0 + b < B and t >= 0 and t < L, v, 0.0)
                    store_tf32(Uh, Ul, uj, split_u, b, j)
                for b, s in T.Parallel(ROWS, S):
                    v = var(Xs[T.min(b0 + b, B - 1), h, k, 2 * n0 + s])
                    X[b, s] = T.Select(b0 + b < B, v, 0.0)
                for b, s in T.Parallel(ROWS, S):
                    store_tf32(Xh, Xl, X[b, s], split, b, s)
                    store_tf32(Mh, Ml, Mu[b, s], split, b, s)
                # Z1 and Z2 are the gradients of `forward_kernel`'s zs[0] and zs[1] for each row.
                for b, s in T.Parallel(ROWS, S):
                    Xp[b, s] = X[b, partner(s)]
                for b, s in T.Parallel(ROWS, S):
                    Z1[b, s] = Z1[b, s] + Mu[b, s] * X[b, s]
                    Z2[b, s] = Z2[b, s] + Mu[b, s] * Xp[b, s]
                for b, s in T.Parallel(ROWS, S):
                    Mp[b, s] = Mu[b, partner(s)]
                for b, s in T.Parallel(ROWS, S):
                    Mn[b, s] = zs[0, s] * Mu[b, s] + zs[1, s] * Mp[b, s]
                if kk == 0:
                    for b, s in T.Parallel(ROWS, S):
                        ok = has_g_last != 0 and b0 + b < B and n0 + mode(s) < Nt
                        Mn[b, s] = Mn[b, s] + T.if_then_else(
                            ok,
                            g_last[T.min(b0 + b, B - 1), h, T.min(n0 + mode(s), Nt - 1), part(s)],
                            0.0,
                        )
                gemm3((Gh, Gl), (Oh, Ol), Mn, split_u, split)
                gemm3((Gh, Gl), (Kh, Kl), Gu, split_u, split, clear=True)
                gemm3((Mh, Ml), (Ih, Il), Gm, split, split, clear=True)
                if acc != 0:
                    for b, j in T.Parallel(ROWS, Tc):
                        t = T.clamp(t0 + j, 0, L - 1)
                        Gu[b, j] = Gu[b, j] + T.cast(gu[T.min(b0 + b, B - 1), h, t], f32)
                for b, j in T.Parallel(ROWS, Tc):
                    t = t0 + j
                    if b0 + b < B and t >= 0 and t < L:
                        gu[b0 + b, h, t] = T.cast(Gu[b, j] + Gm[b, j], gudtype)
                gemm3((Gh, Gl), (Uh, Ul), aK, split_u, split_u, transpose_A=True)
                gemm3((Gh, Gl), (Xh, Xl), aO, split_u, split, transpose_A=True)
                gemm3((Mh, Ml), (Uh, Ul), aI, split, split_u, transpose_A=True)
                T.copy(Mn, Mu)
            if want_dx0 != 0:
                for b, s in T.Parallel(ROWS, S):
                    if b0 + b < B and n0 + mode(s) < Nt:
                        dx0[b0 + b, h, n0 + mode(s), part(s)] = Mu[b, s]
            param_grads(
                row_block * H + h,
                row_block * Ct,
                h,
                n0,
                Nt,
                c0,
                use_D,
                (dt, A, Bm),
                (aK, aO, aI, Z1, Z2),
                (P, Cs, Ws, dBs),
                (ddt, dA, dB, dC, dD),
            )

    return chunk_backward


@compiled
def final_kernel(Tc, N, dtype, reverse, disc):
    """Build the kernel that takes the final state's gradient g_final back through the last chunk.

    The kernel handles the modes [n0, n0 + N). With x the state at the start of the last chunk,
    which has `last` steps, the final state is

        xN = z^last x + sum_e z^e dB u_e

    where u_e is the input e steps before the end of the scan. The kernel stores the gradient g_last
    of x for `backward_kernel` and the gradient of u at the chunk's step i in gt[..., i], or adds it
    to gt with `acc`. It adds the gradients of dt, A and B to the batch block's sums.
    """
    zoh = disc == "zoh"
    B, H, L, Nt, Sx = (T.dynamic(n) for n in ("B", "H", "L", "N", "Sx"))
    nC, NB = T.ceildiv(L, Tc), T.ceildiv(B, ROWS)

    @T.prim_func
    def chunk_final(
        u: T.Tensor((B, H, L), dtype),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, Nt, 2), f32),
        Bm: T.Tensor((H, Nt, 2), f32),
        Xs: T.Tensor((B, H, nC, Sx), f32),
        g_final: T.Tensor((B, H, Nt, 2), f32),
        g_last: T.Tensor((B, H, Nt, 2), f32),
        gt: T.Tensor((B, H, Tc), f32),
        ddt: T.Tensor((NB * H,), f32),
        dA: T.Tensor((NB * H, Nt, 2), f32),
        dB: T.Tensor((NB * H, Nt, 2), f32),
        n0: T.int32,
        acc: T.int32,
    ):
        with T.Kernel(T.ceildiv(B, ROWS), H, threads=128) as (row_block, h):
            T.assume(B > 0)
            T.assume(Nt > 0)
            ws = T.alloc_shared((N, 2), f32)
            dBs = T.alloc_shared((N, 2), f32)
            P = T.alloc_shared((Tc + 1, N, 2), f32)
            G = T.alloc_shared((ROWS, N, 2), f32)
            U = T.alloc_shared((ROWS, Tc), f32)
            gq = T.alloc_shared((N,), f32)
            b0 = row_block * ROWS
            last = L - (nC - 1) * Tc  # steps in the last chunk
            t0 = 0 if reverse else L - last  # first step of the last chunk
            discretized(h, n0, Nt, dt, A, Bm, zoh, ws, dBs)
            powers(ws, P)
            for b, n, r in T.Parallel(ROWS, N, 2):
                ok = b0 + b < B and n0 + n < Nt
                bc, m = T.min(b0 + b, B - 1), T.min(n0 + n, Nt - 1)
                G[b, n, r] = T.if_then_else(ok, g_final[bc, h, m, r], 0.0)
            for b, i in T.Parallel(ROWS, Tc):
                ok = b0 + b < B and i < last
                v = u[T.min(b0 + b, B - 1), h, T.min(t0 + i, L - 1)]
                U[b, i] = T.if_then_else(ok, T.cast(v, f32), 0.0)
            for b, n, r in T.Parallel(ROWS, N, 2):  # conj(z^last) g_final
                if b0 + b < B and n0 + n < Nt:
                    sign = T.if_then_else(r == 0, 1.0, -1.0)
                    g = P[last, n, 0] * G[b, n, r] + sign * P[last, n, 1] * G[b, n, 1 - r]
                    g_last[b0 + b, h, n0 + n, r] = g
            for b, i in T.Parallel(ROWS, Tc):  # gt: Re(conj(z^e dB) g_final) summed over the modes
                e = T.clamp(i if reverse else last - 1 - i, 0, Tc)
                tot = T.alloc_var(f32, init=0.0)
                for n in T.serial(N):
                    ar = P[e, n, 0] * dBs[n, 0] - P[e, n, 1] * dBs[n, 1]
                    ai = P[e, n, 0] * dBs[n, 1] + P[e, n, 1] * dBs[n, 0]
                    tot = tot + G[b, n, 0] * ar + G[b, n, 1] * ai
                if b0 + b < B:
                    gt[b0 + b, h, i] = tot + T.if_then_else(acc != 0, gt[b0 + b, h, i], 0.0)
            # gw and gb are the gradients of log dA and dB.
            for n in T.Parallel(N):
                br, bi = dBs[n, 0], dBs[n, 1]
                gwr = T.alloc_var(f32, init=0.0)
                gwi = T.alloc_var(f32, init=0.0)
                gbr = T.alloc_var(f32, init=0.0)
                gbi = T.alloc_var(f32, init=0.0)
                rf = T.cast(last, f32)
                # Through z^last x, G_log z += last conj(z^last x) g_final over the rows.
                for b in T.serial(ROWS):
                    bc = T.min(b0 + b, B - 1)
                    xr = T.if_then_else(b0 + b < B, Xs[bc, h, nC - 1, 2 * n0 + slot(n)], 0.0)
                    xi = T.if_then_else(b0 + b < B, Xs[bc, h, nC - 1, 2 * n0 + slot(n) + W], 0.0)
                    zr = xr * G[b, n, 0] + xi * G[b, n, 1]
                    zi = xr * G[b, n, 1] - xi * G[b, n, 0]
                    gwr = gwr + rf * (P[last, n, 0] * zr + P[last, n, 1] * zi)
                    gwi = gwi + rf * (P[last, n, 0] * zi - P[last, n, 1] * zr)
                # Through z^e dB u, with v the sum of u g_final over the rows, G_dB += v conj(z^e)
                # and G_log z += e v conj(z^e dB).
                for i in T.serial(last):
                    e = i if reverse else last - 1 - i
                    ef = T.cast(e, f32)
                    vr = T.alloc_var(f32, init=0.0)
                    vi = T.alloc_var(f32, init=0.0)
                    for b in T.serial(ROWS):
                        vr = vr + U[b, i] * G[b, n, 0]
                        vi = vi + U[b, i] * G[b, n, 1]
                    gbr = gbr + vr * P[e, n, 0] + vi * P[e, n, 1]
                    gbi = gbi + vi * P[e, n, 0] - vr * P[e, n, 1]
                    qr = vr * br + vi * bi
                    qi = vi * br - vr * bi
                    gwr = gwr + ef * (P[e, n, 0] * qr + P[e, n, 1] * qi)
                    gwi = gwi + ef * (P[e, n, 0] * qi - P[e, n, 1] * qr)
                gw, gb = (gwr, gwi), (gbr, gbi)
                add_param_grads(row_block * H + h, h, n0, Nt, n, dt, A, Bm, zoh, gw, gb, gq, dA, dB)
            add_dt_grad(row_block * H + h, gq, ddt)

    return chunk_final


@compiled
def sums_kernel(N):
    """Build the kernel that adds the batch blocks' sums of `backward_kernel` and `final_kernel`.

    The kernel adds the sums in block order, so that the gradients are the same in every run. When
    use_D is 0, sD and dD can have any size.
    """
    H, C, SH, SC = (T.dynamic(n) for n in ("H", "C", "SH", "SC"))  # SH = nb H, SC = nb C
    D0, D1, E0, E1 = (T.dynamic(f"p{i}") for i in range(4))

    @T.prim_func
    def chunk_sums(
        sdt: T.Tensor((SH,), f32),
        sA: T.Tensor((SH, N, 2), f32),
        sB: T.Tensor((SH, N, 2), f32),
        sC: T.Tensor((SC, H, N, 2), f32),
        sD: T.Tensor((D0, D1), f32),
        ddt: T.Tensor((H,), f32),
        dA: T.Tensor((H, N, 2), f32),
        dB: T.Tensor((H, N, 2), f32),
        dC: T.Tensor((C, H, N, 2), f32),
        dD: T.Tensor((E0, E1), f32),
        use_D: T.int32,
    ):
        with T.Kernel(H, threads=64) as h:
            nb = int_var(SH // H)  # batch blocks
            for n, r in T.Parallel(N, 2):
                tot_a = T.alloc_var(f32, init=0.0)
                tot_b = T.alloc_var(f32, init=0.0)
                for k in T.serial(nb):
                    tot_a = tot_a + sA[k * H + h, n, r]
                    tot_b = tot_b + sB[k * H + h, n, r]
                dA[h, n, r] = tot_a
                dB[h, n, r] = tot_b
            for c in T.serial(C):
                for n, r in T.Parallel(N, 2):
                    tot = T.alloc_var(f32, init=0.0)
                    for k in T.serial(nb):
                        tot = tot + sC[k * C + c, h, n, r]
                    dC[c, h, n, r] = tot
            if T.get_thread_binding() == 0:
                tot = T.alloc_var(f32, init=0.0)
                for k in T.serial(nb):
                    tot = tot + sdt[k * H + h]
                ddt[h] = tot
                if use_D != 0:
                    for c in T.serial(E0):
                        tot = 0.0
                        for k in T.serial(nb):
                            tot = tot + sD[c, k * H + h]
                        dD[c, h] = tot

    return chunk_sums
