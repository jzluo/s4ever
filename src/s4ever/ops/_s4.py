"""S4: state space models with a diagonal plus low-rank state matrix Lambda - P P^*.

The models are discretized with the bilinear transform, and C is in the C-tilde parameterization of
the original S4.
"""

import math
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from s4ever._backends import metal, shared, tilelang
from s4ever._backends.shared import ceil_div, register_op

from . import _complex as cx
from ._fftconv import fft_mode, fftconv_with_addend
from ._shared import (
    auto_mode,
    check_channels,
    check_params,
    check_shapes,
    longest_sequence,
    no_autocast,
    packed,
    unsupported,
    varlen,
)

type S4Mode = Literal["auto", "fft", "recurrent"]
S4_MODES = get_args(S4Mode.__value__)
CHUNK = 256


def frequencies(L: int, like: Tensor) -> Tensor:
    """Return nu = 2 tan(pi k / L) for 0 < k < L/2.

    At the L-th roots of unity omega, z = 2 (1 - omega) / (1 + omega) = i nu. For k > L/4, nu is
    computed as 2 / tan(pi (L/2 - k) / L), so that no argument of tan exceeds pi/4. Near pi/2, the
    rounding error of the argument makes tan inaccurate.
    """
    k = torch.arange(1, (L + 1) // 2, device=like.device, dtype=like.dtype)
    return torch.where(
        4 * k <= L, 2 * torch.tan(math.pi / L * k), 2 / torch.tan(math.pi / L * (L / 2 - k))
    )


def chunks(H: int, N: int, n_freq: int, device: torch.device) -> list[slice]:
    """Return slices that split the n_freq frequencies into chunks.

    Each chunk has about 2^22 Cauchy denominators, H N per frequency. On MPS it has about 2^20,
    so that the denominators fit in the system-level cache.
    """
    step = max(1, (1 << (20 if device.type == "mps" else 22)) // (H * N))
    return [slice(s, min(s + step, n_freq)) for s in range(0, max(n_freq, 1), step)]


def inverse_q(a: Tensor, wabs2: Tensor, nu: Tensor):
    """Return a function inverse(k) that gives 1 / q at the frequencies nu[k].

    inverse(k) returns the real and imaginary parts as one (2, H, N, len(k)) tensor:

        q = (|w|^2 - nu^2) - 2i nu a
        d = conj(q) = (|w|^2 - nu^2, 2 nu a)
        1 / q = d / |d|^2

    With the two parts in one tensor, each slice of frequencies needs four ops, which matters on
    MPS, where each op runs in about the same time whatever its size.
    """
    p = torch.stack((wabs2, a))[..., None]
    r = torch.stack((-nu * nu, torch.zeros_like(nu)))[:, None, None]
    s = torch.stack((torch.ones_like(nu), 2 * nu))[:, None, None]

    def inverse(k):
        d = torch.addcmul(r[..., k], p, s[..., k])
        return d / torch.addcmul(d[0] * d[0], d[1], d[1])

    return inverse


def cauchy(coef: Tensor, wabs2: Tensor, a: Tensor, nu: Tensor) -> tuple[Tensor, Tensor]:
    """Return the real and imaginary parts (H, R, n_freq) of the sums r at z = i nu.

    The coefficients coef = (c0, c1) are (H, 2R, N), and q is the denominator of `inverse_q`:

        r = sum_n (c0_n + i nu c1_n) / q_n
          = S0 + i nu S1
        S = coef m
        m = 1 / q
    """
    H, N = a.shape
    inverse = inverse_q(a, wabs2, nu)
    S = [coef @ inverse(k) for k in chunks(H, N, len(nu), a.device)]
    # Concatenating a single chunk would copy it.
    Sr, Si = S[0] if len(S) == 1 else torch.cat(S, -1)
    R = Sr.shape[1] // 2
    return Sr[:, :R] - nu * Si[:, R:], Si[:, :R] + nu * Sr[:, R:]


def cauchy_grads(
    coef: Tensor, wabs2: Tensor, a: Tensor, nu: Tensor, g_rr: Tensor, g_ri: Tensor
) -> tuple[Tensor, Tensor]:
    """Return the gradient G_coef of coef, and Z (H, N, 2), from the gradient of `cauchy`'s r.

    That gradient is G_r = g_rr + i g_ri (H, R, n_freq). G_S and G_m are the gradients of S and m:

        G_S = (G_r, -i nu G_r)
        G_coef = Re(conj(G_S) m^T)
        G_m = coef^T G_S
        m^2 = c2 + 2i p
        Z = sum_k (c2 G_mr + 2 p G_mi, nu (c2 G_mi - 2 p G_mr))
        U = (G_Sr, nu G_Si)
        V = 2 (G_Si, -nu G_Sr)

    Z is the product of coef^T with the sum of c2 U + p V over the frequencies, so the
    (H, N, n_freq) tensor G_m is never formed.
    """
    g_Sr, g_Si = torch.cat((g_rr, nu * g_ri), 1), torch.cat((g_ri, -nu * g_rr), 1)
    (H, N), n_freq = a.shape, len(nu)
    U = torch.cat((g_Sr, nu * g_Si), 1).mT
    V = torch.cat((g_Si, -nu * g_Sr), 1).mT.mul_(2)
    # g_coef is allocated contiguous, because with the transposed strides of coef, MKL builds
    # would run baddbmm_ into it as one addmm_ per batch entry.
    g_coef, Z = coef.new_zeros(coef.shape), coef.new_zeros(H, N, U.shape[-1])
    inverse = inverse_q(a, wabs2, nu)
    for k in chunks(H, N, n_freq, a.device):
        mr, mi = inverse(k)
        g_coef.baddbmm_(g_Sr[..., k], mr.mT).baddbmm_(g_Si[..., k], mi.mT)
        c2 = torch.addcmul(mr * mr, mi, mi, value=-1)
        Z.baddbmm_(c2, U[:, k]).baddbmm_(mr.mul_(mi), V[:, k])
    return g_coef, (Z.unflatten(-1, (2, -1)) * coef.mT[:, :, None]).sum(-1)


def others(t: Tensor) -> Tensor:
    """Return sum_{m != n} t_m over the last axis.

    The prefix and suffix sums never add t_n, so a large t_n does not leave a rounding error in
    the result.
    """
    z = torch.zeros_like(t[..., :1])
    left = torch.cat((z, t[..., :-1]), -1).cumsum(-1)
    return left + torch.cat((t[..., 1:], z), -1).flip(-1).cumsum(-1).flip(-1)


def zero_bin(A: Tensor, P: Tensor, B: Tensor, C: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Return the spectrum (C, H) at z = 0, and the solutions that the backward uses.

    Each r_XY(0) is the sum of the terms t = -2 Re(X Y / Lambda) over the modes. These terms are
    large for modes with a small |Lambda|, and their rounding errors would dominate
    r_BC - r_BQ r_PC / (1 + r_PQ). The spectrum k(0) is therefore computed from the solution x of
    (Lambda - P P^*) x = B:

        k(0) = -2 Re sum_n C_n x_n
        x_n = (B_n (1 + r_PQ) - P_n r_BQ) / (Lambda_n (1 + r_PQ))

    In the numerator of x_n, r_PQ and r_BQ are summed over the modes other than n, and the terms
    of mode n are added in closed form:

        B t_PQ - P t_BQ = -2i P Im(B conj(P)) / conj(Lambda)

    The kernels in `tilelang.kernels.cauchy` and `cauchy_spectrum.metal` compute k(0) with the same
    closed form.

    The returned solutions x and mu, where mu solves (Lambda - Q Q^*) mu = C, are stacked as
    (C + 1, H, N), and the values Q^T x and P^T mu as (C + 1, H).
    """
    lr, li = A.unbind(-1)
    pr, pi = P.unbind(-1)
    Vr, Vi = torch.cat((B[None], C)).unbind(-1)  # V = (B, C...)
    # W = (P, Q...). Only the imaginary parts of W are stored, because every entry of W has the real
    # part of P.
    Wi = torch.cat((pi[None], -pi.expand_as(Vi[1:])))
    inv = 1 / (lr * lr + li * li)
    # The terms t = -2 Re(u / Lambda) of r_BQ(0), r_PC(0)... have u = V conj(W). The terms of
    # r_PQ(0) have u = |P|^2.
    ur = torch.cat((Vr * pr + Vi * Wi, (pr * pr + pi * pi)[None]))
    ui = torch.cat((Vi * pr - Vr * Wi, torch.zeros_like(pr)[None]))
    t = -2 * (ur * lr + ui * li) * inv
    o = others(t)
    den = 1 + t[-1].sum(-1)
    s = inv / den[:, None]
    mr = Vr + (Vr * o[-1] - pr * o[:-1])
    mi = Vi + (Vi * o[-1] - Wi * o[:-1])
    zr = (mr * lr + mi * li + 2 * ui[:-1] * Wi) * s
    zi = (mi * lr - mr * li - 2 * ui[:-1] * pr) * s
    k0 = -2 * (Vr[1:] * zr[0] - Vi[1:] * zi[0]).sum(-1)
    return k0, (zr, zi, -t[:-1].sum(-1) / den)


def zero_bin_grads(g0: Tensor, zr: Tensor, zi: Tensor, sigma: Tensor) -> list[Tensor]:
    """Return the gradients of Lambda, P, B and C from the gradient g0 (C, H) of k(0).

    M is Lambda - P P^* over the modes and their conjugates. The gradient of each parameter is
    2 conj(G), where G is the derivative of k(0) with respect to the parameter:

        k(0) = -C^T M^-1 B
        x = M^-1 B
        mu = M^-T C
    """
    xr, xi = zr[0], zi[0]
    # sum_c g_c mu_c
    gmur, gmui = (g0[..., None] * zr[1:]).sum(0), (g0[..., None] * zi[1:]).sum(0)
    sx, gs = sigma[0, :, None], (g0 * sigma[1:]).sum(0)[:, None]  # Q^T x, sum_c g_c P^T mu_c
    return [
        cx.cplx(2 * (gmur * xr - gmui * xi), -2 * (gmur * xi + gmui * xr)),
        cx.cplx(-2 * (sx * gmur + gs * xr), 2 * (sx * gmui - gs * xi)),
        cx.cplx(-2 * gmur, 2 * gmui),
        cx.cplx(-2 * g0[..., None] * xr, 2 * g0[..., None] * xi),
    ]


class Spectrum(torch.autograd.Function):
    """The kernel's spectrum (C, H, L // 2 + 1, 2) at the L-th roots of unity.

    `cauchy` evaluates the resolvent sums r_XY for X in (B, P) and Y in (C, Q) at z = i nu, and the
    Woodbury identity for P P^* combines them into k:

        w = dt Lambda
        Q = conj(P)
        v = dt X Y
        r_XY(z) = sum_n v / (z - w) + conj(v) / (z - conj(w))
                = sum_n (2 z Re(v) - 2 Re(v conj(w))) / ((z - w) (z - conj(w)))
        s = 1 / (1 + r_PQ)
        beta = r_BQ s
        k = r_BC - beta r_PC
        k_f = (1 + z/2) k
    """

    @staticmethod
    def forward(ctx, dt, A, P, B, C, L):
        Cc, H, N = C.shape[:3]
        a, b = (dt[:, None, None] * A).unbind(-1)
        u = cx.mul(torch.stack((B, P))[:, None], torch.cat((C, cx.conj(P)[None]))[None])
        vr, vi = (u * dt[:, None, None]).unbind(-1)  # (2, Cc + 1, H, N)
        coef = torch.stack((-2 * (vr * a + vi * b), 2 * vr)).flatten(0, 2).transpose(0, 1)
        nu, wabs2 = frequencies(L, a), a * a + b * b
        R = Cc + 1
        rr, ri = cauchy(coef, wabs2, a, nu)  # rows r_BC..., r_BQ, r_PC..., r_PQ (H, 2R, n_freq)
        qr, qi = 1 + rr[:, -1:], ri[:, -1:]
        q2 = qr * qr + qi * qi
        sr, si = qr / q2, -qi / q2
        bqr, bqi = rr[:, Cc : Cc + 1], ri[:, Cc : Cc + 1]
        br, bi = bqr * sr - bqi * si, bqr * si + bqi * sr
        pr, pi = rr[:, R:-1], ri[:, R:-1]
        kr, ki = rr[:, :Cc] - (br * pr - bi * pi), ri[:, :Cc] - (br * pi + bi * pr)
        k_f = torch.stack((kr - ki * nu / 2, ki + kr * nu / 2), -1).transpose(0, 1)
        k0, zero = zero_bin(A, P, B, C)
        bins = [F.pad(k0[..., None, None], (0, 1)), k_f]
        if L % 2 == 0:
            # At omega = -1, z is infinite. The limit of (1 + z/2) r_BC there is Re sum_n v_BC.
            bins.append(F.pad(vr[0, :Cc].sum(-1)[..., None, None], (0, 1)))
        ctx.save_for_backward(dt, A, P, B, C, u, coef, pr, pi, br, bi, sr, si, *zero)
        ctx.L = L
        return torch.cat(bins, dim=-2)

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        dt, A, P, B, C, u, coef, pr, pi, br, bi, sr, si, *zero = ctx.saved_tensors
        g = grads[0]
        L, Cc = ctx.L, C.shape[0]
        a, b = (dt[:, None, None] * A).unbind(-1)
        nu, wabs2 = frequencies(L, a), a * a + b * b
        n_freq = nu.shape[0]
        # The gradient gr + i gi (H, C, n_freq) of k is conj(1 + z/2) times the gradient of k_f.
        gr, gi = g[:, :, 1 : n_freq + 1].transpose(0, 1).unbind(-1)
        gr, gi = gr + nu / 2 * gi, gi - nu / 2 * gr
        gbr = -(pr * gr + pi * gi).sum(1, keepdim=True)
        gbi = -(pr * gi - pi * gr).sum(1, keepdim=True)
        tr, ti = br * sr - bi * si, br * si + bi * sr  # beta s
        # the gradients of r_BC, r_BQ, r_PC and r_PQ
        g_rr = torch.cat((gr, sr * gbr + si * gbi, -(br * gr + bi * gi), -(tr * gbr + ti * gbi)), 1)
        g_ri = torch.cat((gi, sr * gbi - si * gbr, -(br * gi - bi * gr), -(tr * gbi - ti * gbr)), 1)
        # The gradients of |w|^2 and a are computed from Z. The gradient of q is
        # G_q = -conj(m^2) G_m. The derivatives of q are dq/d|w|^2 = 1 and dq/da = -2i nu.
        g_coef, Z = cauchy_grads(coef, wabs2, a, nu, g_rr, g_ri)
        g_abs, ga = -Z[..., 0], 2 * Z[..., 1]
        # coef_0 = -2 (vr a + vi b), coef_1 = 2 vr, v = dt u, w = (a, b) = dt Lambda
        g0, g1 = g_coef.transpose(0, 1).unflatten(0, (2, 2, Cc + 1))
        vr, vi = (u * dt[:, None, None]).unbind(-1)
        gvr, gvi = -2 * a * g0 + 2 * g1, -2 * b * g0
        if L % 2 == 0:
            gvr[0, :Cc] += g[:, :, L // 2, 0, None]
        ga = ga + 2 * a * g_abs - 2 * (vr * g0).sum((0, 1))
        gb = 2 * b * g_abs - 2 * (vi * g0).sum((0, 1))
        gv, gw = cx.cplx(gvr, gvi), cx.cplx(ga, gb)
        gdt = (gv * u).sum((0, 1, 3, 4)) + (gw * A).sum((1, 2))
        gu = gv * dt[:, None, None]
        X, Y = torch.stack((B, P)), torch.cat((C, cx.conj(P)[None]))
        gX, gY = cx.mul(cx.conj(Y)[None], gu).sum(1), cx.mul(cx.conj(X)[:, None], gu).sum(0)
        gA, gP, gB, gC = zero_bin_grads(g[:, :, 0, 0], *zero)
        gA = gA + gw * dt[:, None, None]
        return gdt, gA, gX[1] + cx.conj(gY[Cc]) + gP, gX[0] + gB, gY[:Cc] + gC, None


def s4_conv_kernel(
    dt: Tensor,
    A: Tensor,
    P: Tensor,
    B: Tensor,
    C: Tensor,
    L: int,
    *,
    fused_kernels: bool | None = None,
) -> Tensor:
    """Return the convolution kernel (C, H, L) of the discretized SSM.

    C is the C-tilde for length L. The shapes and `fused_kernels` are as in `s4`.
    """
    check_params(dt, A, P, B, C)
    check_channels(C)
    with no_autocast(dt):
        N = A.shape[1]
        fused = dt.dtype == torch.float32 and N <= shared.MAX_CONV_KERNEL_MODES
        if fused and tilelang.enabled(dt, fused_kernels):
            return tilelang.s4_conv_kernel(dt, A, P, B, C, L)
        if metal.enabled(dt, fused_kernels) and metal.cauchy.fits(N):
            return metal.cauchy.s4_conv_kernel(dt, A, P, B, C, L)
        if fused_kernels:
            most = shared.MAX_CONV_KERNEL_MODES
            raise unsupported(fused_kernels, "S4's convolution kernels", dt, modes=N, most=most)
        return torch.fft.irfft(torch.view_as_complex(Spectrum.apply(dt, A, P, B, C, L)), n=L)


def re_dot(a: Tensor, x: Tensor) -> Tensor:
    """Return 2 Re sum_n a_n x_n over the modes."""
    return 2 * (a[..., 0] * x[..., 0] - a[..., 1] * x[..., 1]).sum(-1)


def explicit(x: Tensor, dt: Tensor, A: Tensor, P: Tensor) -> Tensor:
    """Return (2/dt + Lambda - P P^*) x for Lambda = A."""
    return (2 / dt)[:, None, None] * x + cx.mul(A, x) - P * re_dot(cx.conj(P), x)[..., None, None]


def factors(dt: Tensor, A: Tensor, P: Tensor, B: Tensor) -> tuple[Tensor, ...]:
    """Return the factors lam, g, w and dB (H, N, 2) of the bilinear step.

    The step is x' = lam x - g (w . x) + dB u for states x (..., N, 2), where w . x is the real dot
    product over the last two axes. The factors follow from Sherman-Morrison:

        J = (2/dt - Lambda)^-1
        Q = conj(P)
        s = 1 + 2 Re(Q J P)
        lam = J (2/dt + Lambda)
        g = 4 / (dt s) J P
        w = 2 conj(Q J)
        w . x = 2 Re(Q J x)
    """
    two_dt = (2 / dt)[:, None]
    Ar, Ai = A.unbind(-1)
    z = cx.cplx(two_dt - Ar, -Ai)
    J = cx.conj(z) / z.square().sum(-1, keepdim=True)
    JP, QJ = cx.mul(J, P), cx.mul(cx.conj(P), J)
    s = 1 + re_dot(QJ, P)
    dB = 2 * (cx.mul(J, B) - JP * (re_dot(QJ, B) / s)[:, None, None])
    Jr, Ji = J.unbind(-1)
    four_dt = 2 * two_dt
    g = (four_dt / s[:, None])[..., None] * JP
    # lam is 1 + 2 Lambda J = 4/dt J - 1, and both forms have the imaginary part 4/dt Im J. The
    # real part is computed as 1 + 2 Re(Lambda J) when Re lam > 0, which holds exactly when
    # |Lambda| < 2/dt, and as 4/dt Re J - 1 otherwise. Either way the term added to +-1 is the
    # smaller of the two, which keeps the rounding error of lam small. That error compounds over
    # the steps.
    lr = four_dt * Jr - 1
    lr = torch.where(lr > 0, 1 + 2 * (Ar * Jr - Ai * Ji), lr)
    return cx.cplx(lr, four_dt * Ji), g, 2 * cx.conj(QJ), dB


def matrix(lam: Tensor, g: Tensor, w: Tensor) -> Tensor:
    """Return dA (H, 2N, 2N) from the lam, g and w of `factors`.

    dA acts on states flattened to (..., 2N).
    """
    H, N = lam.shape[:2]
    lr, li = lam.unbind(-1)
    blocks = torch.stack((lr, -li, li, lr), -1).unflatten(-1, (2, 2))  # (H, N, 2, 2)
    eye = torch.eye(N, device=lam.device, dtype=lam.dtype)[:, None, :, None]
    low = g.flatten(-2)[..., None] * w.flatten(-2)[:, None]
    return (eye * blocks[..., None, :]).reshape(H, 2 * N, 2 * N) - low


def dense(dt: Tensor, A: Tensor, P: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Return the dense dA (H, 2N, 2N) and dB (H, 2N) of the bilinear discretization."""
    lam, g, w, dB = factors(dt, A, P, B)
    return matrix(lam, g, w), dB.flatten(-2)


@register_op("matrix_power", lambda a, n: torch.empty_like(a))
def cuda_matrix_power(a: Tensor, n: int) -> Tensor:
    # torch.linalg.matrix_power holds a NoTF32Guard, but cuBLASLt and hipBLASLt choose TF32 from
    # the global setting without checking the guard. With TF32 on and
    # torch.backends.cuda.preferred_blas_library("cublaslt"), the power of a float32 batch
    # therefore has TF32 accuracy. PyTorch also calls these libraries under TunableOp when cuBLAS
    # is preferred, and hipBLASLt is the default on ROCm. Complex64 products run in float32 in
    # cuBLAS under the guard and in both BLASLt libraries.
    if a.dtype == torch.float32 and torch.backends.cuda.matmul.fp32_precision == "tf32":
        return torch.linalg.matrix_power(a.to(torch.complex64), n).real.contiguous()
    return torch.linalg.matrix_power(a, n)


def matrix_power(a: Tensor, n: int) -> Tensor:
    """Return a^n with float32 products, even when CUDA's float32 matmuls run in TF32.

    TF32 products make the powers of S4's dense dA inaccurate. On CUDA the power runs as a custom
    op, so that torch.compile does not decompose it into matmuls that follow the TF32 setting.
    """
    return cuda_matrix_power(a, n) if a.is_cuda else torch.linalg.matrix_power(a, n)


def s4_recurrence(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, C: Tensor, l_kernel: int
) -> tuple[Tensor, ...]:
    """Return 2 conj(c) (C, H, 2N), followed by the lam, g, dB and W of `s4_steps`.

    The readout c is recovered from the C-tilde C by a solve with I - dA^l_kernel.
    """
    with no_autocast(dt):
        lam, g, w, dB = factors(dt, A, P, B)
        dA = matrix(lam, g, w)
        c_tilde = (2 * cx.conj(C)).flatten(-2)
        eye = torch.eye(dA.shape[-1], device=dA.device, dtype=dA.dtype)
        M = eye - matrix_power(dA, l_kernel)
        cvec = torch.linalg.solve_ex(M.mT, c_tilde.permute(1, 2, 0))[0].permute(2, 0, 1)
        W = torch.cat((w[None], cvec.unflatten(-1, (-1, 2))))
    return cvec, lam, g, dB, W


def state_response(
    x0: Tensor, dt: Tensor, A: Tensor, P: Tensor, C: Tensor, L: int, fused_kernels: bool | None
) -> Tensor:
    """Return the response (C, B, H, L) of the outputs to the initial states x0.

    At step t < L the response is 2 Re(c dA^(t+1) x0), where C is the C-tilde c (I - dA^L) of the
    readout c. The responses are the convolution kernels of the transposed systems, which have the
    input matrix C and the output matrices X:

        M = Lambda - P P^*
        M^T = Lambda - conj(P) P^T
        X = (2/dt + M) x0 / 2
        C (zI - M)^-1 X = X^T (zI - M^T)^-1 C^T

    For B = X, the bilinear discretization gives dB = dA x0.
    """
    with no_autocast(dt):
        N = A.shape[1]
        fused = dt.dtype == torch.float32 and N <= shared.MAX_CONV_KERNEL_MODES
        if fused and tilelang.enabled(dt, fused_kernels):
            return tilelang.s4_state_response(dt, A, P, C, x0, L)
        X = explicit(x0, dt, A, P) / 2
        if metal.enabled(dt, fused_kernels) and metal.cauchy.fits(N):
            ks = [metal.cauchy.s4_conv_kernel(dt, A, cx.conj(P), Ci, X, L) for Ci in C]
            return torch.stack(ks) if len(ks) > 1 else ks[0][None]
        if fused_kernels:
            most = shared.MAX_CONV_KERNEL_MODES
            raise unsupported(fused_kernels, "S4's convolution kernels", dt, modes=N, most=most)
        ks = [Spectrum.apply(dt, A, cx.conj(P), Ci / L, X, L) for Ci in C]
        # Stacking a single channel would copy it.
        k_f = torch.stack(ks) if len(ks) > 1 else ks[0][None]
        return torch.fft.irfft(torch.view_as_complex(k_f), n=L, norm="forward")


def dense_state_powers(
    dA: Tensor, dB: Tensor, T: int, n_powers: int
) -> tuple[Tensor, list[Tensor]]:
    """Return `state_powers` computed from the dense dA and dB.

    Each update of X doubles the width w of the Krylov block, and each update of E squares dA^w:

        X = [dA^(w-1) dB, ..., dB]
        E = dA^w - I
        X <- [X + E X, X]
        E <- 2 E + E^2
    """
    Es = [dA - torch.eye(dA.shape[-1], device=dA.device, dtype=dA.dtype)]
    X = dB[..., None]
    while X.shape[-1] < T:
        X = torch.cat((torch.baddbmm(X, Es[-1], X), X), dim=-1)
        Es.append(matrix_power(Es[-1], 2).add_(Es[-1], alpha=2))
    while len(Es) < n_powers:
        Es.append(matrix_power(Es[-1], 2).add_(Es[-1], alpha=2))
    return X.mT, Es


def refuse_grad(*tensors: Tensor | None) -> None:
    if shared.tracks_grad(*tensors):
        raise ValueError(
            "S4 does not support states while gradients are tracked; use torch.no_grad(), or S4D "
            "to train with states"
        )


def powers_on_kernels(x: Tensor, N: int, fused_kernels: bool | None) -> bool:
    """Return whether TileLang's state kernels run for x and N modes."""
    fits = x.dtype == torch.float32 and N <= tilelang.MAX_POWER_MODES
    return fits and tilelang.enabled(x, fused_kernels)


def state_powers(
    dt: Tensor, A: Tensor, P: Tensor, B: Tensor, L: int, fused_kernels: bool | None
) -> tuple[Tensor, list[Tensor]]:
    """Return the factors that give the final state after at most L steps.

    These factors depend only on the parameters. The Krylov block [dA^(T-1) dB, ..., dA dB, dB]
    (H, T, 2N) maps a chunk of T inputs to its state, and the powers dA^(2^k) - I combine the
    states of the chunks. The powers are stored minus I so that their rounding error is relative
    to their distance from I, which is small for slow modes.
    """
    refuse_grad(dt, A, P, B)
    N = A.shape[1]
    fused = powers_on_kernels(dt, N, fused_kernels)
    T = tilelang.S4_CHUNK if fused else min(CHUNK, 1 << (L - 1).bit_length())
    n_powers = max(L - 1, T - 1).bit_length() + 1
    with no_autocast(dt):
        if fused:
            return tilelang.s4_state_powers(dt, A, P, B, T, n_powers)
        if metal.enabled(dt, fused_kernels) and metal.powers.fits(N):
            return metal.powers.state_powers(dt, A, P, B, T, n_powers)
        if fused_kernels:
            most = tilelang.MAX_POWER_MODES if dt.is_cuda else metal.powers.MAX_MODES
            raise unsupported(fused_kernels, "S4's state kernels", dt, modes=N, most=most)
        In, Es = dense_state_powers(*dense(dt, A, P, B), T, n_powers)
    return In, Es


def final_state(
    u: Tensor, powers: tuple[Tensor, list[Tensor]], x0: Tensor | None, fused_kernels: bool | None
) -> Tensor:
    """Return the state after the SSM runs over u (B, L, H) from x0 (B, H, 2N), or from zero."""
    In, Es = powers
    Bsz, L, H = u.shape
    T = In.shape[1]
    nC = 1 << (ceil_div(L, T) - 1).bit_length()
    N = In.shape[-1] // 2
    if tilelang.enabled(u, fused_kernels) and powers_on_kernels(In, N, fused_kernels):
        X = tilelang.s4_group_states(u, In, Es[T.bit_length() - 1])
    else:
        # Zero inputs before u do not change the state, which starts at zero.
        u = F.pad(u, (0, 0, nC * T - L, 0)) if nC * T > L else u
        U = u.new_empty(H, nC, Bsz, T, dtype=In.dtype)
        U.copy_(u.unflatten(1, (nC, T)).permute(3, 1, 0, 2))
        X = torch.bmm(U.view(H, nC * Bsz, T), In)
    k = (nC * T).bit_length() - 1
    # The states of the chunks are combined in halves as second + dA^(2^k) first. Because the rows
    # of X are ordered by chunk, then by batch row, each half is one block of rows.
    while X.shape[1] > Bsz:
        k -= 1
        first, second = X.chunk(2, dim=1)
        X = (first + second).baddbmm_(first, Es[k].mT)
    if x0 is not None:
        x0 = x0.transpose(0, 1)
        for k in range(L.bit_length()):  # dA^L x0
            if L >> k & 1:
                x0 = torch.baddbmm(x0, x0, Es[k].mT)
        X = X + x0
    return X.transpose(0, 1)


def run_fft(u, params, K, powers, D, l_kernel, *, initial_state=None, reverse=False, fused_kernels):
    """Run S4's fft mode with states, given the convolution kernel K of params.

    `powers` is the output of `state_powers`, or None when the final state is not needed.
    """
    refuse_grad(u, *params, D, initial_state)
    dt, A, P, B, C = params
    with no_autocast(u):
        u = u.flip(1) if reverse else u
        x0 = None if initial_state is None else initial_state.to(dt.dtype)
        free = None if x0 is None else state_response(x0, dt, A, P, C, l_kernel, fused_kernels)
        y = fftconv_with_addend(u, K, D, None, free, fused_kernels)
        y = y.flip(1) if reverse else y
        if powers is None:
            return y, None
        x = final_state(u, powers, None if x0 is None else x0.flatten(-2), fused_kernels)
    return y, x.unflatten(-1, (-1, 2))


def s4_steps(u, lam, g, dB, W, D, x0):
    """Run the steps x <- lam x - g (w . x) + dB u, and return the outputs and the final states.

    The outputs are y = 2 Re(c x), the states are (B, H, N, 2), and W is (w, 2 conj(c)...). The
    steps avoid matmuls, whose TF32 errors would accumulate from step to step.
    """
    Bsz, L, H = u.shape
    if x0 is None:
        x, r0 = u.new_zeros(Bsz, H, *g.shape[1:]), u.new_zeros(Bsz, H)
    else:
        x, r0 = x0.contiguous(), (W[0] * x0).sum((-2, -1))
    compiled = torch.compiler.is_compiling()
    z = lam if compiled else torch.view_as_complex(lam)
    ys = []
    for t in range(L):
        x = cx.mul(z, x) if compiled else torch.view_as_real(torch.view_as_complex(x) * z)
        x = x.addcmul_(g, r0[..., None, None], value=-1).addcmul_(dB, u[:, t, :, None, None])
        r = (W * x[:, None]).sum((-2, -1))
        r0 = r[:, 0]
        ys.append(r[:, 1:])
    # Stacking a single step would copy it.
    y = ys[0][:, None] if L == 1 else torch.stack(ys, dim=1)
    if D is not None:
        y = torch.addcmul(y, u[:, :, None], D)
    return y, x


def s4(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    P: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    *,
    l_kernel: int | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    mode: S4Mode = "auto",
    cu_seqlens: Tensor | None = None,
    fused_kernels: bool | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Run an S4 SSM over u (B, L, H), and return y (B, L, C, H) and the final state.

    The output is y = 2 Re(c x) + D u for the readout c, in u's dtype. The shapes are dt (H,),
    A, P and B (H, N, 2), C (C, H, N, 2) and D (C, H), where N counts one mode of each conjugate
    pair and the trailing axis is (re, im). A is the diagonal Lambda of the state matrix
    Lambda - P P^*, and the states (B, H, N, 2) are in the basis of that matrix.

    C is the C-tilde c (I - dA^l_kernel). The kernel length `l_kernel` defaults to the length of
    u, and in fft mode it must be at least the length of the longest sequence. With a layer's
    `ssm_params`, pass the layer's `l_kernel`.

    `mode` is 'auto', 'fft' or 'recurrent'. The default, 'auto', picks recurrent mode for a single
    step without gradients and fft mode otherwise. `reverse`, `cu_seqlens` and `fused_kernels`
    behave as in `s4d`. S4 does not support states while gradients are tracked, and recurrent mode
    does not support gradients.
    """
    if mode not in S4_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {S4_MODES}")
    check_params(dt, A, P, B, C, D)
    check_channels(C)
    bounds = packed(u, cu_seqlens)
    check_shapes(u, A.shape[0], A.shape[1], initial_state, bounds)
    L, l_kernel = longest_sequence(u, bounds), l_kernel or u.shape[1]
    with no_autocast(u):
        if mode == "auto":
            grad = shared.tracks_grad(u, dt, A, P, B, C, D, initial_state)
            mode = auto_mode(u.shape[1], grad)
        kw = dict(reverse=reverse, fused_kernels=fused_kernels)
        if mode == "fft":
            if l_kernel < L:
                raise ValueError(f"fft mode needs l_kernel >= L, got l_kernel={l_kernel} < L={L}")
            params = (dt, A, P, B, C)
            K = s4_conv_kernel(*params, l_kernel, fused_kernels=fused_kernels)
            if initial_state is None and not output_final_state:

                def run(u, x0):
                    return fft_mode(u, K, D, reverse, fused_kernels), None
            else:
                powers = state_powers(dt, A, P, B, L, fused_kernels) if output_final_state else None

                def run(u, x0):
                    return run_fft(u, params, K, powers, D, l_kernel, initial_state=x0, **kw)
        else:
            rec = s4_recurrence(dt, A, P, B, C, l_kernel)
            kw["output_final_state"] = output_final_state

            def run(u, x0):
                return run_s4_recurrence(u, (dt, A, P, B), rec, D, initial_state=x0, **kw)

        if bounds is None:
            return run(u, initial_state)
        return varlen(run, u, bounds, initial_state, output_final_state)


def run_s4_recurrence(
    u, params, rec, D, *, initial_state=None, output_final_state=False, reverse=False, fused_kernels
):
    """Run S4's recurrent mode from rec, the `s4_recurrence` of params (dt, A, P, B)."""
    cvec, *step = rec
    dtype = u.dtype
    with no_autocast(u):
        if initial_state is not None or output_final_state:
            refuse_grad(u, *params, cvec, D, initial_state)
        if shared.tracks_grad(u, *params, cvec, D):
            raise ValueError("S4's recurrent mode does not support gradients; use mode='fft'")
        dt, A, P, B = params
        args = (initial_state, output_final_state, reverse)
        if cvec.dtype == torch.float32 and tilelang.enabled(u, fused_kernels):
            return tilelang.s4_recurrent(u, dt, A, P, B, cvec, D, *args)
        if metal.enabled(u, fused_kernels) and metal.recurrent.fits(A.shape[1], u.dtype):
            return metal.recurrent.s4_recurrent(u, dt, A, P, B, cvec, D, *args)
        if fused_kernels:
            most = None if u.is_cuda else metal.recurrent.MAX_MODES
            raise unsupported(
                fused_kernels, "S4's recurrent kernels", cvec, u, modes=A.shape[1], most=most
            )
        u = u.to(cvec.dtype)
        u = u.flip(1) if reverse else u
        x0 = None if initial_state is None else initial_state.to(u.dtype)
        y, x = s4_steps(u, *step, D=D, x0=x0)
        y = y.flip(1) if reverse else y
    return y.to(dtype), x if output_final_state else None
