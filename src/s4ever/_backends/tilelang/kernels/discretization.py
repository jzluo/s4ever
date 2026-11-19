"""S4D's discretization and its chain rule, for one mode in one thread."""

import tilelang.language as T

from .dsl import cos, exp, sin, var


def expm1(x, ex):
    """Return exp(x) - 1, given ex = exp(x).

    For x <= 1 the result is 2 t / (1 - t) with t = tanh(x / 2), which does not cancel near 0. For
    x > 1, ex - 1 is accurate and 1 - t is not.
    """
    t = T.tanh(x / 2)
    return T.Select(x > 1, ex - 1, 2 * t / (1 - t))


def discretize(dt, A, B, h, n, zoh):
    """Return log dA and dB for mode n of feature h as (re, im, re, im).

    With `zoh` the discretization is the zero-order hold, and otherwise it is the bilinear
    transform.
    """
    Ar, Ai, Br, Bi = A[h, n, 0], A[h, n, 1], B[h, n, 0], B[h, n, 1]
    dt = var(dt)
    ar, ai = var(dt * Ar), var(dt * Ai)
    if zoh:
        # log dA = a = dt A, dB = B expm1(a) / A
        s, ex = var(sin(ai / 2)), var(exp(ar))
        er = var(expm1(ar, ex) * cos(ai) - 2 * s * s)
        ei = var(ex * sin(ai))
        den = var(Ar * Ar + Ai * Ai)
        qr = var((er * Ar + ei * Ai) / den)
        qi = var((ei * Ar - er * Ai) / den)
        lr, li = ar, ai
    else:
        # p = a/2, log dA = 2 atanh(p) and dB = B dt / (1 - p). The atanh formula matches
        # `_complex.atanh`.
        pr, pi = var(ar / 2), var(ai / 2)
        den = var((1 - pr) * (1 - pr) + pi * pi)
        f = var(((1 + pr) * (1 + pr) + pi * pi) / den)
        lr = var(0.5 * T.Select(f < 0.5, T.log(f), T.log1p(4 * pr / den)))
        li = var(T.atan2(2 * pi, (1 - pr) * (1 + pr) - pi * pi))
        qr = var(dt * (1 - pr) / den)
        qi = var(dt * pi / den)
    return lr, li, Br * qr - Bi * qi, Br * qi + Bi * qr


def discretize_grad(dt, A, B, h, n, gw, gb, zoh):
    """Return the gradients of A[h, n], B[h, n] and dt as (A re, A im, B re, B im, dt).

    gw and gb are the gradients of log dA and dB. The chain rule for a holomorphic y(x) is
    G_x = conj(dy/dx) G_y, where G_x is the gradient of x.
    """
    (gwr, gwi), (gbr, gbi) = gw, gb
    Ar, Ai, Br, Bi = A[h, n, 0], A[h, n, 1], B[h, n, 0], B[h, n, 1]
    dt = var(dt)
    ar, ai = var(dt * Ar), var(dt * Ai)
    if zoh:
        # E = expm1(a) / A, dE/dA = (dt exp(a) - E) / A and dE/d dt = exp(a)
        den = var(Ar * Ar + Ai * Ai)
        ex, c, s = var(exp(ar)), var(cos(ai)), var(sin(ai / 2))
        xr, xi = var(ex * c), var(ex * sin(ai))
        mr = var(expm1(ar, ex) * c - 2 * s * s)
        Er = var((mr * Ar + xi * Ai) / den)
        Ei = var((xi * Ar - mr * Ai) / den)
        fr = var(dt * xr - Er)
        fi = var(dt * xi - Ei)
        ur, ui = var(fr * Ar + fi * Ai), var(fi * Ar - fr * Ai)
        jr = var((Br * ur - Bi * ui) / den)  # d(dB)/dA = B dE/dA
        ji = var((Br * ui + Bi * ur) / den)
        Gar = dt * gwr + jr * gbr + ji * gbi
        Gai = dt * gwi + jr * gbi - ji * gbr
        Gbr = Er * gbr + Ei * gbi
        Gbi = Er * gbi - Ei * gbr
        Gdt = Ar * gwr + Ai * gwi + (Br * xr - Bi * xi) * gbr + (Br * xi + Bi * xr) * gbi
    else:
        # q = 1 / (1 - a/2), dB = dt B q, d(dB)/da = dt B q^2 / 2 and
        # d log dA / da = 1 / (1 - a^2/4) = v
        den = var((1 - ar / 2) * (1 - ar / 2) + ai * ai / 4)
        qr, qi = var((1 - ar / 2) / den), var(ai / 2 / den)
        mr, mi = var(1 - (ar * ar - ai * ai) / 4), var(-(ar * ai) / 2)
        vr = var(mr / (mr * mr + mi * mi))
        vi = var(-mi / (mr * mr + mi * mi))
        jr = var(dt / 2 * (Br * (qr * qr - qi * qi) - Bi * 2 * qr * qi))  # d(dB)/da
        ji = var(dt / 2 * (Br * 2 * qr * qi + Bi * (qr * qr - qi * qi)))
        gar = var(vr * gwr + vi * gwi + jr * gbr + ji * gbi)
        gai = var(vr * gwi - vi * gwr + jr * gbi - ji * gbr)
        Gar, Gai = dt * gar, dt * gai
        Gbr = dt * (qr * gbr + qi * gbi)
        Gbi = dt * (qr * gbi - qi * gbr)
        Gdt = Ar * gar + Ai * gai + (Br * qr - Bi * qi) * gbr + (Br * qi + Bi * qr) * gbi
    return Gar, Gai, Gbr, Gbi, Gdt
