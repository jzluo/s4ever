"""Kernels for the layers' parameter transforms (`layers.layer_params`).

`forward_kernel` computes dt = t(p) rate and A = -(t(A_real) + i t(A_imag)) for the transforms t
of `s4ever._layers.transform`. It also repeats the (S, N, 2) parameters of S SSMs over the H
features, so that feature h uses SSM h % S. The gradient of a repeat, viewed as an (H / S, 2 S N)
matrix, then has one column for each entry of the parameter, and `backward_kernel` sums each
column.
"""

import tilelang.language as T

from ..runtime import compiled
from .dsl import exp, f32, int_var, var

THREADS = 1024
UNROLL = 8


# TileLang's safe-memory pass cannot bound indices that are held in variables, so it would put a
# branch around each access. The kernels check the bounds of their accesses themselves.
CONFIG = {"tl.disable_safe_memory_legalize": True}


def transform(kind, p):
    """Return t(p) and t'(p) for the transform t of this kind."""
    if kind == "none":
        return p, 1.0
    if kind == "relu":
        return T.max(p, 0.0) + 1e-4, T.if_then_else(p > 0, 1.0, 0.0)
    if kind == "exp":
        e = var(exp(p))
        return e, e
    if kind == "sigmoid":
        s = var(1 / (1 + exp(-p)))
        return s, s * (1 - s)
    # As in torch, softplus returns p for p > 20.
    z = var(exp(p))
    return T.if_then_else(p > 20, p, T.log1p(z)), T.if_then_else(p > 20, 1.0, z / (z + 1))


@compiled(pass_configs=CONFIG)
def forward_kernel(kinds, n_repeated):
    """Build the kernel that computes dt and A and repeats n_repeated parameters over the features.

    `kinds` holds the transform kinds of dt, A_real and A_imag, and n_repeated is 1 or 2.

    Kernel arguments:
        X, X2: the parameters to repeat (S, N, 2). With one parameter, X2 is a stand-in.
        Y, Y2: the repeats (H, N, 2) of X and X2. With one parameter, Y2 is a stand-in.
    """
    dt_kind, real_kind, imag_kind = kinds
    H, S, N = (T.dynamic(n) for n in ("H", "S", "N"))
    # X2 and Y2 have separate dynamic sizes, so that the (1, 1, 2) stand-ins match them.
    X2_shape = T.dynamic("p0"), T.dynamic("p1"), 2
    Y2_shape = T.dynamic("p2"), T.dynamic("p3"), 2

    @T.prim_func
    def transforms_forward(
        p: T.Tensor((H,), f32),
        Ar: T.Tensor((S, N), f32),
        Ai: T.Tensor((S, N), f32),
        X: T.Tensor((S, N, 2), f32),
        X2: T.Tensor(X2_shape, f32),
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, N, 2), f32),
        Y: T.Tensor((H, N, 2), f32),
        Y2: T.Tensor(Y2_shape, f32),
        rate: f32,
    ):
        with T.Kernel(T.ceildiv(H * N * 2, THREADS), threads=THREADS) as bx:
            # TileLang 0.1.15 assumes only that sizes are non-negative, so without these assumes it
            # checks the signs of the divisors N and S.
            T.assume(N > 0)
            T.assume(S > 0)
            e = bx * THREADS + T.get_thread_binding()
            if e < H:
                dt[e] = transform(dt_kind, p[e])[0] * rate
            if e < H * N * 2:
                h, n, part = int_var(e // (2 * N)), int_var(e % (2 * N) // 2), int_var(e % 2)
                s = int_var(h % S)
                if part == 0:
                    A[h, n, 0] = -transform(real_kind, Ar[s, n])[0]
                else:
                    A[h, n, 1] = -transform(imag_kind, Ai[s, n])[0]
                Y[h, n, part] = X[s, n, part]
                if n_repeated == 2:
                    Y2[h, n, part] = X2[s, n, part]

    return transforms_forward


@compiled(pass_configs=CONFIG)
def backward_kernel(kinds, n_repeated, column_threads):
    """Build the kernel that computes the gradients of the transformed and repeated parameters.

    `column_threads` threads sum the H / S rows of each column. With one repeated parameter, gY2
    and gX2 are stand-ins.
    """
    dt_kind, real_kind, imag_kind = kinds
    COLS = THREADS // column_threads
    H, S, N = (T.dynamic(n) for n in ("H", "S", "N"))
    X2_shape = T.dynamic("p0"), T.dynamic("p1"), 2
    Y2_shape = T.dynamic("p2"), T.dynamic("p3"), 2

    @T.prim_func
    def transforms_backward(
        p: T.Tensor((H,), f32),
        Ar: T.Tensor((S, N), f32),
        Ai: T.Tensor((S, N), f32),
        gdt: T.Tensor((H,), f32),
        gA: T.Tensor((H, N, 2), f32),
        gY: T.Tensor((H, N, 2), f32),
        gY2: T.Tensor(Y2_shape, f32),
        gp: T.Tensor((H,), f32),
        gAr: T.Tensor((S, N), f32),
        gAi: T.Tensor((S, N), f32),
        gX: T.Tensor((S, N, 2), f32),
        gX2: T.Tensor(X2_shape, f32),
        rate: f32,
    ):
        blocks = T.max(T.ceildiv(S * N * 2, COLS), T.ceildiv(H, THREADS))
        with T.Kernel(blocks, threads=THREADS) as bx:
            T.assume(N > 0)
            T.assume(S > 0)
            partial = T.alloc_shared((column_threads, n_repeated + 1, COLS), f32)
            acc = T.alloc_local((n_repeated + 1,), f32)
            tx = T.get_thread_binding()
            if bx * THREADS + tx < H:
                h = bx * THREADS + tx
                gp[h] = gdt[h] * rate * transform(dt_kind, p[h])[1]
            # The thread sums the rows first_row, first_row + column_threads, ... of its column.
            col_in_block, first_row = tx % COLS, tx // COLS
            col = int_var(T.min(bx * COLS + col_in_block, S * N * 2 - 1))
            s, n, part = int_var(col // (2 * N)), int_var(col % (2 * N) // 2), int_var(col % 2)
            rows = int_var(H // S)
            for t in T.unroll(n_repeated + 1):
                acc[t] = 0.0
            for i in T.serial(T.ceildiv(rows, column_threads * UNROLL)):
                for u in T.unroll(UNROLL):
                    row = (i * UNROLL + u) * column_threads + first_row
                    if row < rows:
                        acc[0] += gA[row * S + s, n, part]
                        acc[1] += gY[row * S + s, n, part]
                        if n_repeated == 2:
                            acc[2] += gY2[row * S + s, n, part]
            for t in T.unroll(n_repeated + 1):
                partial[first_row, t, col_in_block] = acc[t]
            if first_row == 0 and bx * COLS + col_in_block < S * N * 2:
                for t in T.unroll(n_repeated + 1):
                    acc[t] = 0.0
                    for q in T.unroll(column_threads):
                        acc[t] += partial[q, t, col_in_block]
                if part == 0:
                    gAr[s, n] = -transform(real_kind, Ar[s, n])[1] * acc[0]
                else:
                    gAi[s, n] = -transform(imag_kind, Ai[s, n])[1] * acc[0]
                gX[s, n, part] = acc[1]
                if n_repeated == 2:
                    gX2[s, n, part] = acc[2]

    return transforms_backward
