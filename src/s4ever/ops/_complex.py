"""Complex arithmetic on real tensors with a trailing (re, im) axis.

The ops use this form because Inductor does not generate code for complex dtypes, and because some
backends support complex dtypes only in part.
"""

import torch
from torch import Tensor


def cplx(re: Tensor, im: Tensor) -> Tensor:
    return torch.stack((re, im), dim=-1)


def conj(a: Tensor) -> Tensor:
    ar, ai = a.unbind(-1)
    return cplx(ar, -ai)


def mul(a: Tensor, b: Tensor) -> Tensor:
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    return cplx(ar * br - ai * bi, ar * bi + ai * br)


def div(a: Tensor, b: Tensor) -> Tensor:
    ar, ai = a.unbind(-1)
    br, bi = b.unbind(-1)
    d = br * br + bi * bi
    return cplx((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)


def exp(a: Tensor) -> Tensor:
    ar, ai = a.unbind(-1)
    e = ar.exp()
    return cplx(e * ai.cos(), e * ai.sin())


def expm1(a: Tensor) -> Tensor:
    """Return exp(a) - 1 without cancellation for small |a|."""
    ar, ai = a.unbind(-1)
    return cplx(ar.expm1() * ai.cos() - 2 * (ai / 2).sin().square(), ar.exp() * ai.sin())


def atanh(a: Tensor) -> Tensor:
    """Return atanh(a) = log((1 + a) / (1 - a)) / 2.

    The real part has two forms:

        f = |1 + a|^2 / |1 - a|^2
        Re atanh(a) = log(f) / 4 = log1p(4 Re(a) / |1 - a|^2) / 4

    The log form is accurate for small f, and the log1p form for f near 1, where the rounding
    error of f is large relative to f - 1.
    """
    ar, ai = a.unbind(-1)
    den = (1 - ar).square() + ai * ai
    f = ((1 + ar).square() + ai * ai) / den
    small = f < 0.5
    # The zero makes the derivative of log1p finite in the unused branch.
    v = torch.where(small, 0, 4 * ar / den)
    re = torch.where(small, f.log(), v.log1p())
    return cplx(0.25 * re, 0.5 * torch.atan2(2 * ai, (1 - ar) * (1 + ar) - ai * ai))


def powers(w: Tensor, n: Tensor) -> Tensor:
    """Return exp(w * n) (..., k, 2) for w (..., 2) and real exponents n (k,).

    The real and imaginary parts are scaled separately. For the complex product
    w[..., None, :] * n[:, None], Inductor compiles the backward on CPUs into a serial loop over the
    entries of w, in which each iteration opens a parallel region for the sum over n.
    """
    wr, wi = w.unbind(-1)
    e, a = (wr[..., None] * n).exp(), wi[..., None] * n
    return cplx(e * a.cos(), e * a.sin())
