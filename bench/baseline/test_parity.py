"""Tests of s4ever against the original S4.

Run them in the environment that bench/baseline/setup.sh builds:

    baseline/.venv/bin/python -m pytest -o addopts="" bench/baseline

A layer check measures the relative error in norm of s4ever and of the original against a float64
copy of the s4ever layer. It passes when s4ever's error is at most twice the original's, or at most
1e-5. Checks of outputs also bound the original's error, because a large error of the original
would mean that the two layers compute different functions.
"""

import copy

import pytest
import torch
from original import dplr, hippo, original

from s4ever import S4, S4D
from s4ever._initialization import diag, nplr

CONFIGS = [
    (S4D, {}),
    (S4D, {"init": "legs", "disc": "bilinear", "channels": 2}),
    (S4D, {"init": "diag", "n_ssm": 2, "bidirectional": True, "dt_transform": "exp"}),
    (S4D, {"init": "diag-real", "real_transform": "softplus", "imag_transform": "exp"}),
    (S4, {}),
    (S4, {"init": "hippo", "n_ssm": 2, "channels": 2}),
    (S4, {"init": "fout", "bidirectional": True, "real_transform": "relu"}),
]


def rel(x, y):
    return ((x.double() - y.double()).norm() / y.double().norm()).item()


def check(s4ever, orig, exact, name, agree=None):
    e_s4ever, e_orig = rel(s4ever, exact), rel(orig, exact)
    assert e_s4ever <= max(2 * e_orig, 1e-5), (
        f"{name}: error {e_s4ever:.2g} vs original's {e_orig:.2g}"
    )
    if agree is not None:
        assert e_orig <= agree, f"{name}: original differs from float64 by {e_orig:.2g}"


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def layers(cls, L, **kw):
    s4ever = (cls(64, 64, l_max=L, **kw) if cls is S4 else cls(64, 64, **kw)).cuda()
    return (
        s4ever,
        original(s4ever, l_max=L if cls is S4 else None).cuda(),
        copy.deepcopy(s4ever).double(),
    )


@pytest.mark.parametrize("cls,kw", CONFIGS)
@pytest.mark.parametrize("L", [256, 1000])
def test_forward_backward(cls, kw, L):
    s4ever, orig, exact = layers(cls, L, **kw)
    x = torch.randn(4, L, 64, device="cuda", requires_grad=True)
    x64 = x.detach().double().requires_grad_()
    ys = s4ever(x)[0], orig(x.mT)[0].mT, exact(x64)[0]  # the original's layout is (B, H, L)
    check(*ys, "y", agree=1e-3)
    g = torch.randn_like(ys[0])
    names, params = zip(*s4ever.named_parameters(), strict=True)
    params_orig = [orig.get_parameter(n if n == "D" else f"kernel.{n}") for n in names]
    grads = [
        torch.autograd.grad(y, [xx, *ps], g.to(y.dtype))
        for y, xx, ps in zip(
            ys, (x, x, x64), (params, params_orig, exact.parameters()), strict=True
        )
    ]
    # Gradients are checked without the `agree` bound, because in the nearly undamped modes of the
    # fout init, cancellation in float32 makes a few elements of each gradient inaccurate. With the
    # fout and hippo inits, the gradient of inv_dt is not checked, because its relative error is
    # 1e-3 to 1e-2 in both s4ever and the original.
    for name, *gs in zip(["x", *names], *grads, strict=True):
        if name != "inv_dt" or kw.get("init") not in ("hippo", "fout"):
            check(*(t.reshape(gs[2].shape) for t in gs), f"grad {name}")


@pytest.mark.parametrize("cls", [S4D, S4])
def test_state(cls):
    """Compare state passing and stepping with the original's state forwarding and stepping."""
    s4ever, orig, exact = layers(cls, 128)
    x = torch.randn(2, 128, 64, device="cuda")
    state = 0.1 * torch.randn(2, 64, 32, 2, device="cuda")
    with torch.no_grad():
        y, next_state = s4ever(x, state)
        y_orig, next_orig = orig(x.mT, state=torch.view_as_complex(state))
        y64, next64 = exact(x.double(), state.double())
        check(y, y_orig.mT, y64, "y", agree=1e-3)
        check(next_state, torch.view_as_real(next_orig), next64, "state", agree=1e-3)
        orig.setup_step()
        s, s_orig, s64 = (
            torch.zeros_like(state),
            orig.default_state(2),
            torch.zeros_like(state).double(),
        )
        for t in range(16):
            y_t, s = s4ever(x[:, t : t + 1], s)
            y_orig, s_orig = orig.step(x[:, t], s_orig)
            y64, s64 = exact(x[:, t : t + 1].double(), s64)
            check(y_t[:, 0], y_orig, y64[:, 0], f"step {t}", agree=1e-3)


@pytest.mark.parametrize("measure", ["legs", "fout"])
@pytest.mark.parametrize("N", [8, 64, 256])
def test_nplr(measure, N):
    """Compare `nplr` with the original's NPLR form.

    The phases of the eigenvectors are arbitrary, so P and B are compared through |P|, |B| and
    B conj(P).
    """

    def invariants(Lam, P, B):
        return torch.view_as_real(Lam), P.abs(), B.abs(), torch.view_as_real(B * P.conj())

    Lam, P, B, _ = hippo.nplr(measure, N)
    s4ever, orig = invariants(*nplr(measure, N)), invariants(Lam, P[0], B)
    for x, y in zip(s4ever, orig, strict=True):
        assert rel(x, y) <= 1e-4


@pytest.mark.parametrize("init", ["lin", "inv", "real"])
def test_diag(init):
    A, _, B, _ = dplr.dplr(init=init, N=64, H=1)
    A_s4ever, B_s4ever = diag(f"diag-{init}", 64)
    assert rel(torch.view_as_real(A_s4ever), torch.view_as_real(A[0])) <= 1e-6
    assert torch.equal(B_s4ever, B[0])
