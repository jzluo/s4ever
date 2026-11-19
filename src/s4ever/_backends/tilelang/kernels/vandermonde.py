"""S4D's convolution kernel and its gradients.

With z_n = dA for mode n, the kernel is

    k[c, l] = 2 Re sum_n C[c, n] dB[n] z_n^l

A tile has NA rows a and TB columns b, and its entry (a, b) is at position l = l0 + a TB + b. Over a
tile, the sum over the modes is the product X Y:

    X[a, n] = 2 C dB z^(l0 + a TB)
    Y[n, b] = z^b

`backward_kernel` computes two sums from the tile products P = G conj(Y)^T and
Q = (b G) conj(Y)^T:

    S0 = sum_l g_l conj(z)^l
    S1 = sum_l l g_l conj(z)^l

The kernels form each power as the product of two tabulated powers, z^(i + 8 j) = z^i z^(8 j).
`backward_kernel` computes the row powers conj(z)^(l0 + TB a) with `power` instead, because products
would compound the error of dt's gradient. The kernels handle the M modes in chunks of K modes.
"""

import math

import tilelang.language as T

from s4ever._backends.shared import PI2, PI2_LO, ceil_div

from ..runtime import compiled
from .discretization import discretize, discretize_grad
from .dsl import cmul, const, cos, exp, f32, fma, sin, var

TB = 64


def config(M: int, C: int, L: int, smem: int, backward: bool) -> tuple[int, int]:
    """Return the channels per launch and the tile rows NA.

    The function picks the fewest launches, and for those the most tile rows, whose kernel fits in
    `smem` bytes of shared memory. It raises ValueError when no configuration fits.
    """
    rows = max(16, min(64, 1 << (ceil_div(L, TB) - 1).bit_length()))
    for launches in range(1, C + 1):
        Cg = ceil_div(C, launches)
        for NA in (64, 32, 16):
            if rows >= NA and smem_bytes(M, Cg, NA, backward) <= smem:
                return Cg, NA
    raise ValueError(
        f"S4D's fused convolution kernel does not support {M} modes with {smem} bytes of shared "
        "memory; use fused_kernels=False"
    )


def chunk(M: int, backward: bool) -> int:
    """Return the modes per chunk, K."""
    return min(32 if backward else 16, ceil_div(M, 8) * 8)


def thread_rows(NA: int, K: int) -> int:
    """Return the tile rows per thread of `backward_kernel`.

    The count is 4 only if the block then has at least 256 threads, because a block needs that many
    threads to hide latencies.
    """
    return 4 if NA // 4 * (K // 2) >= 256 else 2


def shared_shapes(M: int, C: int, NA: int, backward: bool) -> dict[str, tuple[int, ...]]:
    """Return the shapes of the float32 buffers that the kernel keeps in shared memory."""
    K = chunk(M, backward)
    Mp = ceil_div(M, K) * K
    if backward:
        return {
            "ws": (Mp, 2),
            "dBs": (Mp, 2),
            "pw": (K, 17, 2),  # the entry after the 16 powers avoids bank conflicts
            "Ys": (TB, K, 2),
            "Gs": (NA, TB + 1),  # the column after the TB positions avoids bank conflicts
            "red": (NA // thread_rows(NA, K), C, K, 4),
            "S": (C, Mp, 4),
        }
    return {
        "ws": (Mp, 2),
        "Ws": (C, Mp, 2),
        "pw": (K, 24 + NA // 8, 2),
        "Ys": (K, 2, TB),
        "Xs": (C, K, 2, NA),
    }


def smem_bytes(M: int, C: int, NA: int, backward: bool) -> int:
    """Return the shared memory in bytes that the kernel allocates for C channels."""
    return 4 * sum(math.prod(shape) for shape in shared_shapes(M, C, NA, backward).values())


def power(e, lr, li):
    """Return z^e = exp(e log z) for a float exponent e and log z = lr + i li."""
    phase = var(e * li)
    turns = var(T.floor(phase * const(1 / (2 * math.pi)) + 0.5))
    # r is the phase e li reduced to [-pi, pi]. The rounding error of e li is added back, so that r
    # is the same whether or not the compiler contracts the products into fmas.
    r = fma(-turns, const(PI2), phase) - turns * const(PI2_LO) + fma(e, li, -phase)
    mag = exp(e * lr)
    return mag * cos(r), mag * sin(r)


@compiled
def forward_kernel(M, C, disc, NA):
    """Build the kernel that computes k (C, H, L).

    Block (bl, h) computes the tile at l0 = bl NA TB of feature h, and each thread computes 4 x 4
    entries of the tile for each channel.
    """
    zoh = disc == "zoh"
    K = chunk(M, False)  # modes per chunk
    chunks = ceil_div(M, K)
    NO = NA // 8  # rows a = 8 j + i with i < 8, j < NO
    shapes = shared_shapes(M, C, NA, False)
    H, L = T.dynamic("H"), T.dynamic("L")

    @T.prim_func
    def vandermonde_forward(
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, M, 2), f32),
        B: T.Tensor((H, M, 2), f32),
        Cm: T.Tensor((C, H, M, 2), f32),
        k: T.Tensor((C, H, L), f32),
    ):
        with T.Kernel(T.ceildiv(L, NA * TB), H, threads=4 * NA) as (bl, h):
            ws = T.alloc_shared(shapes["ws"], f32)  # log z
            Ws = T.alloc_shared(shapes["Ws"], f32)  # 2 C dB
            pw = T.alloc_shared(shapes["pw"], f32)  # z^i, z^(8 i), z^(l0 + TB i), z^(8 TB j)
            Ys = T.alloc_shared(shapes["Ys"], f32)
            Xs = T.alloc_shared(shapes["Xs"], f32)
            acc = T.alloc_local((C * 16,), f32)
            xr = T.alloc_local((4,), f32)
            xi = T.alloc_local((4,), f32)
            yr = T.alloc_local((4,), f32)
            yi = T.alloc_local((4,), f32)
            tx = T.get_thread_binding()
            # a0 and b0 are the first row and column of the thread's outputs. Each warp computes
            # 16 x 32 outputs, and each lane computes 4 x 4 of them.
            a0 = tx // 64 * 16 + tx % 32 // 8 * 4
            b0 = tx // 32 % 2 * 32 + tx % 8 * 4
            l0 = bl * NA * TB
            for n in T.Parallel(chunks * K):
                m = T.min(n, M - 1)
                lr, li, br, bi = discretize(dt[h], A, B, h, m, zoh)
                ws[n, 0] = lr
                ws[n, 1] = li
                for c in T.serial(C):
                    Wr, Wi = cmul((Cm[c, h, m, 0], Cm[c, h, m, 1]), (br, bi))
                    Ws[c, n, 0] = T.if_then_else(n < M, 2 * Wr, 0.0)
                    Ws[c, n, 1] = T.if_then_else(n < M, 2 * Wi, 0.0)
            for i in T.unroll(C * 16):
                acc[i] = 0.0
            for ch in T.serial(chunks):
                for n, p in T.Parallel(K, 24 + NO):
                    j = p % 8
                    e = T.if_then_else(
                        p < 16,
                        T.if_then_else(p < 8, j, 8 * j),
                        T.if_then_else(p < 24, l0 + TB * j, 8 * TB * (p - 24)),
                    )
                    pr, pi = power(T.cast(e, f32), ws[ch * K + n, 0], ws[ch * K + n, 1])
                    pw[n, p, 0] = pr
                    pw[n, p, 1] = pi
                for n, b in T.Parallel(K, TB):
                    i, j = b % 8, 8 + b // 8
                    vr, vi = cmul((pw[n, i, 0], pw[n, i, 1]), (pw[n, j, 0], pw[n, j, 1]))
                    Ys[n, 0, b] = vr
                    Ys[n, 1, b] = vi
                for c, n, a in T.Parallel(C, K, NA):
                    i, j = 16 + a % 8, 24 + a // 8
                    zr, zi = cmul((pw[n, i, 0], pw[n, i, 1]), (pw[n, j, 0], pw[n, j, 1]))
                    vr, vi = cmul((Ws[c, ch * K + n, 0], Ws[c, ch * K + n, 1]), (zr, zi))
                    Xs[c, n, 0, a] = vr
                    Xs[c, n, 1, a] = vi
                for n4 in T.serial(K // 4):
                    for u in T.unroll(4):
                        n = n4 * 4 + u
                        for j in T.vectorized(4):
                            yr[j] = Ys[n, 0, b0 + j]
                        for j in T.vectorized(4):
                            yi[j] = Ys[n, 1, b0 + j]
                        for c in T.unroll(C):
                            for i in T.vectorized(4):
                                xr[i] = Xs[c, n, 0, a0 + i]
                            for i in T.vectorized(4):
                                xi[i] = Xs[c, n, 1, a0 + i]
                            for i in T.unroll(4):
                                for j in T.unroll(4):
                                    q = c * 16 + i * 4 + j
                                    acc[q] = acc[q] + xr[i] * yr[j] - xi[i] * yi[j]
            for c in T.unroll(C):
                for i in T.unroll(4):
                    for j in T.unroll(4):
                        pos = l0 + (a0 + i) * TB + b0 + j
                        if pos < L:
                            k[c, h, pos] = acc[c * 16 + i * 4 + j]

    return vandermonde_forward


@compiled
def backward_kernel(M, C, disc, NA):
    """Build the kernel that computes the gradients of dt, A, B and C from the gradient g of k.

    Block h handles feature h. For each chunk of modes, it loops over the tiles of g, and each
    thread adds the sums S0 and S1 of RB tile rows and 2 modes.
    """
    zoh = disc == "zoh"
    K = chunk(M, True)  # modes per chunk
    chunks = ceil_div(M, K)
    Mp = chunks * K
    RB = thread_rows(NA, K)
    NG, MG = NA // RB, K // 2  # groups of RB rows, pairs of modes
    shapes = shared_shapes(M, C, NA, True)
    H, L = T.dynamic("H"), T.dynamic("L")

    @T.prim_func
    def vandermonde_backward(
        dt: T.Tensor((H,), f32),
        A: T.Tensor((H, M, 2), f32),
        B: T.Tensor((H, M, 2), f32),
        Cm: T.Tensor((C, H, M, 2), f32),
        g: T.Tensor((C, H, L), f32),
        gdt: T.Tensor((H,), f32),
        gA: T.Tensor((H, M, 2), f32),
        gB: T.Tensor((H, M, 2), f32),
        gC: T.Tensor((C, H, M, 2), f32),
    ):
        with T.Kernel(H, threads=NG * MG) as h:
            ws = T.alloc_shared(shapes["ws"], f32)  # log z
            dBs = T.alloc_shared(shapes["dBs"], f32)
            pw = T.alloc_shared(shapes["pw"], f32)  # z^i and z^(8 i)
            Ys = T.alloc_shared(shapes["Ys"], f32)  # conj(z)^b
            Gs = T.alloc_shared(shapes["Gs"], f32)  # a tile of g
            red = T.alloc_shared(shapes["red"], f32)  # the sums of each group of rows
            S = T.alloc_shared(shapes["S"], f32)  # S0 and S1 of each channel and mode
            P = T.alloc_local((RB * 8,), f32)  # P and Q of each row and mode
            row_powers = T.alloc_local((RB * 4,), f32)  # conj(z)^(l0 + TB a) of each row and mode
            acc = T.alloc_local((C * 8,), f32)
            gv = T.alloc_local((RB,), f32)  # g in the thread's rows at column b
            bg = T.alloc_local((RB,), f32)  # b g
            yv = T.alloc_local((4,), f32)
            tx = T.get_thread_binding()
            mg = tx % MG  # the thread's pair of modes
            rg = tx // MG  # the thread's group of rows
            for n in T.Parallel(Mp):
                lr, li, br, bi = discretize(dt[h], A, B, h, T.min(n, M - 1), zoh)
                ws[n, 0] = lr
                ws[n, 1] = li
                dBs[n, 0] = br
                dBs[n, 1] = bi
            for ch in T.serial(chunks):
                for n, p in T.Parallel(K, 16):
                    e = T.if_then_else(p < 8, p, 8 * (p - 8))
                    pr, pi = power(T.cast(e, f32), ws[ch * K + n, 0], ws[ch * K + n, 1])
                    pw[n, p, 0] = pr
                    pw[n, p, 1] = pi
                for b, n in T.Parallel(TB, K):
                    i, j = b % 8, 8 + b // 8
                    vr, vi = cmul((pw[n, i, 0], pw[n, i, 1]), (pw[n, j, 0], pw[n, j, 1]))
                    # The powers of a growing mode can overflow at the positions l >= L, where g is
                    # zero padding, and an infinite power times zero would make the sums NaN. The
                    # kernel sets those powers to 0.
                    Ys[b, n, 0] = T.if_then_else(b < L, vr, 0.0)
                    Ys[b, n, 1] = T.if_then_else(b < L, -vi, 0.0)
                for i in T.unroll(C * 8):
                    acc[i] = 0.0
                for t in T.serial(T.ceildiv(L, NA * TB)):
                    l0 = t * NA * TB
                    for i in T.unroll(RB):
                        for m in T.unroll(2):
                            n, e = ch * K + 2 * mg + m, l0 + (rg * RB + i) * TB
                            zr, zi = power(T.cast(e, f32), ws[n, 0], ws[n, 1])
                            row_powers[i * 4 + m * 2] = T.if_then_else(e < L, zr, 0.0)
                            row_powers[i * 4 + m * 2 + 1] = T.if_then_else(e < L, -zi, 0.0)
                    for c in T.serial(C):
                        for a, b in T.Parallel(NA, TB):
                            pos = l0 + a * TB + b
                            Gs[a, b] = T.if_then_else(pos < L, g[c, h, T.min(pos, L - 1)], 0.0)
                        for i in T.unroll(RB * 8):
                            P[i] = 0.0
                        for b in T.serial(TB):
                            for i in T.unroll(RB):
                                gv[i] = Gs[rg * RB + i, b]
                                bg[i] = T.cast(b, f32) * gv[i]
                            for j in T.vectorized(4):
                                yv[j] = Ys[b, 2 * mg + j // 2, j % 2]
                            for i in T.unroll(RB):
                                for m in T.unroll(2):
                                    q = i * 8 + m * 4
                                    P[q] = P[q] + gv[i] * yv[2 * m]
                                    P[q + 1] = P[q + 1] + gv[i] * yv[2 * m + 1]
                                    P[q + 2] = P[q + 2] + bg[i] * yv[2 * m]
                                    P[q + 3] = P[q + 3] + bg[i] * yv[2 * m + 1]
                        for i in T.unroll(RB):
                            e = T.cast(l0 + (rg * RB + i) * TB, f32)
                            for m in T.unroll(2):
                                q, rp = i * 8 + m * 4, i * 4 + m * 2
                                # For a row whose first position is e, S0 += conj(z)^e P and
                                # S1 += conj(z)^e (e P + Q).
                                s0r, s0i = cmul(
                                    (row_powers[rp], row_powers[rp + 1]), (P[q], P[q + 1])
                                )
                                s1 = (e * P[q] + P[q + 2], e * P[q + 1] + P[q + 3])
                                s1r, s1i = cmul((row_powers[rp], row_powers[rp + 1]), s1)
                                acc[c * 8 + m * 4] = acc[c * 8 + m * 4] + s0r
                                acc[c * 8 + m * 4 + 1] = acc[c * 8 + m * 4 + 1] + s0i
                                acc[c * 8 + m * 4 + 2] = acc[c * 8 + m * 4 + 2] + s1r
                                acc[c * 8 + m * 4 + 3] = acc[c * 8 + m * 4 + 3] + s1i
                for c in T.unroll(C):
                    for m in T.unroll(2):
                        for r in T.unroll(4):
                            red[rg, c, 2 * mg + m, r] = acc[c * 8 + m * 4 + r]
                for n, r in T.Parallel(K, 4):
                    for c in T.serial(C):
                        tot = T.alloc_var(f32, init=0.0)
                        for q in T.serial(NG):
                            tot = tot + red[q, c, n, r]
                        S[c, ch * K + n, r] = tot
            for n in T.Parallel(Mp):
                if n < M:
                    gwr = T.alloc_var(f32, init=0.0)
                    gwi = T.alloc_var(f32, init=0.0)
                    gbr = T.alloc_var(f32, init=0.0)
                    gbi = T.alloc_var(f32, init=0.0)
                    for c in T.serial(C):
                        cr = Cm[c, h, n, 0]
                        ci = Cm[c, h, n, 1]
                        # W = C dB, G_W = 2 S0 and G_log z = 2 conj(W) S1
                        Wr = cr * dBs[n, 0] - ci * dBs[n, 1]
                        Wi = cr * dBs[n, 1] + ci * dBs[n, 0]
                        sr = 2 * S[c, n, 0]
                        si = 2 * S[c, n, 1]
                        gC[c, h, n, 0] = dBs[n, 0] * sr + dBs[n, 1] * si
                        gC[c, h, n, 1] = dBs[n, 0] * si - dBs[n, 1] * sr
                        gbr = gbr + cr * sr + ci * si
                        gbi = gbi + cr * si - ci * sr
                        gwr = gwr + 2 * (Wr * S[c, n, 2] + Wi * S[c, n, 3])
                        gwi = gwi + 2 * (Wr * S[c, n, 3] - Wi * S[c, n, 2])
                    Gar, Gai, Gbr, Gbi, Gdt = discretize_grad(
                        dt[h], A, B, h, n, (gwr, gwi), (gbr, gbi), zoh
                    )
                    gA[h, n, 0] = Gar
                    gA[h, n, 1] = Gai
                    gB[h, n, 0] = Gbr
                    gB[h, n, 1] = Gbi
                    # After the chunk loop, nothing reads log z from ws, so ws[n, 0] can hold mode
                    # n's term of dt's gradient.
                    ws[n, 0] = Gdt
            tot = T.alloc_var(f32, init=0.0)
            for n in T.serial(M):
                tot = tot + ws[n, 0]
            gdt[h] = tot

    return vandermonde_backward
