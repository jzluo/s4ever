import concurrent.futures
import functools
import importlib
import math
from unittest import mock

import pytest
import reference as ref
import torch
from helpers import ACCELERATOR, DEVICES, assert_close, tf32

from s4ever._backends import metal, shared, tilelang
from s4ever._backends.tilelang import fftconv as fftconv_launchers
from s4ever._backends.tilelang import transpose
from s4ever._initialization import s4_init, s4d_init
from s4ever.ops import fftconv, s4, s4_conv_kernel, s4d, s4d_conv_kernel
from s4ever.ops._fftconv import fftconv_with_addend
from s4ever.ops._s4 import run_s4_recurrence, s4_recurrence, state_powers

S4_MODES = ["fft", "recurrent"]


def log_uniform(n, lo=1e-3, hi=1e-1):
    return torch.exp(torch.rand(n) * (math.log(hi) - math.log(lo)) + math.log(lo))


def s4d_params(H=4, N=8, Cc=2, init="diag"):
    A, B = (torch.view_as_real(t.repeat(H // 2, 1)) for t in s4d_init(init, 2 * N, 2))
    A = A * (1 + 0.1 * torch.randn_like(A))
    return (
        log_uniform(H),
        A,
        B + 0.1 * torch.randn_like(B),
        torch.randn(Cc, H, N, 2),
        torch.randn(Cc, H),
    )


def s4_params(H=4, N=8, Cc=2, init="hippo"):
    Lam, P, B = (torch.view_as_real(t.repeat(H // 2, 1)) for t in s4_init(init, 2 * N, 2))
    Lam = Lam.clone()
    # The layers clamp the initial Re(Lambda) to at most -1e-4.
    Lam[..., 0] = Lam[..., 0].clamp(max=-1e-4)
    return log_uniform(H), Lam, P, B, torch.randn(Cc, H, N, 2), torch.randn(Cc, H)


def to(device, *xs):
    return [None if x is None else x.to(device) for x in xs]


@pytest.mark.parametrize("device", DEVICES)
# The discretization and the direction do not interact.
@pytest.mark.parametrize("disc,reverse", [("zoh", False), ("bilinear", True)])
@pytest.mark.parametrize(
    "mode,stateful",
    [("fft", False), ("chunk", False), ("chunk", True), ("recurrent", False), ("recurrent", True)],
)
def test_s4d(device, disc, reverse, mode, stateful):
    dt, A, B, C, D = s4d_params()
    # PyTorch's chunk mode splits the 128 steps into two whole chunks and does not pad u.
    u = torch.randn(3, 128, 4)
    x0 = torch.randn(3, 4, 8, 2) if stateful else None
    y_ref, x_ref = ref.run(*ref.s4d_matrices(dt, A, B, C, disc), D, u, x0, reverse)
    args = to(device, u, dt, A, B, C, D)
    x0 = None if x0 is None else x0.to(device)
    kw = dict(disc=disc, initial_state=x0, output_final_state=stateful, reverse=reverse, mode=mode)
    y, x = s4d(*args, **kw)
    assert y.shape == (3, 128, 2, 4) and y.dtype == u.dtype
    assert_close(y, y_ref)
    if stateful:
        assert_close(x, x_ref)
    if mode == "recurrent":
        with pytest.raises(ValueError, match="does not support gradients"):
            s4d(args[0].requires_grad_(), *args[1:], **kw)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "op,mode,stateful",
    [("s4d", "chunk", True), ("s4d", "fft", False), ("s4", "fft", True), ("s4", "recurrent", True)],
)
def test_varlen(device, op, mode, stateful):
    """Compare packed sequences on the device with separate calls on the CPU."""
    params = s4d_params() if op == "s4d" else s4_params()
    fn = s4d if op == "s4d" else s4
    lengths = [37, 1, 70, 64]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    u = torch.randn(1, sum(lengths), 4)
    x0 = torch.randn(len(lengths), 4, 8, 2) if stateful else None
    kwargs = {"mode": mode, "output_final_state": stateful}
    if op == "s4":
        kwargs["l_kernel"] = 128
    ud, x0d, cu, *pd = to(device, u, x0, cu_seqlens, *params)
    y, x = fn(ud, *pd, initial_state=x0d, cu_seqlens=cu, **kwargs)
    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        x0_i = None if x0 is None else x0[i : i + 1]
        y_i, x_i = fn(u[:, s:e], *params, initial_state=x0_i, **kwargs)
        assert_close(y[:, s:e], y_i)
        if stateful:
            assert_close(x[i : i + 1], x_i)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize("view", ["slice", "transpose"])
@pytest.mark.parametrize(
    "op,mode",
    [("s4d", "fft"), ("s4d", "chunk"), ("s4d", "recurrent"), ("s4", "fft"), ("s4", "recurrent")],
)
def test_views(view, op, mode):
    """Compare the kernels on views of u and the initial state with PyTorch on contiguous copies.

    The slices have storage offsets and gaps between rows. In the transposed views, the steps of u
    and the modes of the state have stride 1.
    """
    H, N = 32, 8  # On CUDA, the transpose kernel runs for at least 32 steps and 32 features.
    if view == "slice":
        u, x0 = torch.randn(4, 80, H + 2)[1:, 8:72, 1:-1], torch.randn(3, H, N + 1, 2)[:, :, 1:]
    else:
        u, x0 = torch.randn(3, H, 64).mT, torch.randn(3, N, H, 2).transpose(1, 2)
    fn, params = (s4d, s4d_params(H, N)) if op == "s4d" else (s4, s4_params(H, N))
    stateful = not (op == "s4d" and mode == "fft")
    u, x0, *params = to(ACCELERATOR, u, x0 if stateful else None, *params)
    kw = dict(mode=mode, output_final_state=stateful)
    ours = fn(u, *params, initial_state=x0, fused_kernels=True, **kw)
    x0 = None if x0 is None else x0.contiguous()
    theirs = fn(u.contiguous(), *params, initial_state=x0, fused_kernels=False, **kw)
    for a, b in zip(ours, theirs, strict=True):
        if b is not None:
            assert_close(a, b)


@pytest.mark.parametrize("init", ["legs", "hippo"])
@pytest.mark.parametrize("L", [1, 2, 7, 300])
def test_s4_conv_kernel(L, init):
    params = [x.double().requires_grad_() for x in s4_params(init=init)[:5]]
    K, K_ref = s4_conv_kernel(*params, L), ref.conv_kernel(*ref.s4_matrices(*params, L), L)
    assert_close(K, K_ref, tol=1e-10)
    g = torch.randn_like(K)
    grads = [torch.autograd.grad(k, params, g) for k in (K, K_ref)]
    # The gradients are compared together, because the gradient of dt is zero at L = 1.
    assert_close(*(torch.cat([x.flatten() for x in gs]) for gs in grads), tol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_s4_auto(device):
    """Compare the outputs of mode='auto' at L = 1 with and without gradients.

    At L = 1, 'auto' picks recurrent mode without gradients and fft mode with gradients.
    """
    dt, Lam, P, B, C, D = s4_params()
    args = to(device, torch.randn(2, 1, 4), dt, Lam, P, B, C, D)
    y = s4(*args, l_kernel=16)[0]
    assert_close(s4(args[0].requires_grad_(), *args[1:], l_kernel=16)[0], y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", S4_MODES)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
def test_s4(device, mode, reverse, stateful):
    dt, Lam, P, B, C, D = s4_params()
    L, l_kernel = 100, 128
    u = torch.randn(2, L, 4)
    x0 = None
    if stateful:
        kw = dict(l_kernel=l_kernel, output_final_state=True, mode="recurrent")
        x0 = s4(torch.randn(2, 300, 4), dt, Lam, P, B, C, **kw)[1]
    dA, dB, Ctrue = ref.s4_matrices(dt, Lam, P, B, C, l_kernel)
    y_ref, x_ref = ref.run(dA, dB, Ctrue, D, u, x0, reverse)
    args = to(device, u, dt, Lam, P, B, C, D)
    x0 = None if x0 is None else x0.to(device)
    y, x = s4(
        *args,
        l_kernel=l_kernel,
        initial_state=x0,
        output_final_state=stateful,
        reverse=reverse,
        mode=mode,
    )
    assert_close(y, y_ref)
    if stateful:
        assert_close(x, x_ref)
        with pytest.raises(ValueError, match="does not support states while gradients"):
            s4(args[0].requires_grad_(), *args[1:], l_kernel=l_kernel, initial_state=x0, mode=mode)
    elif mode == "recurrent":
        with pytest.raises(ValueError, match="does not support gradients"):
            s4(args[0].requires_grad_(), *args[1:], l_kernel=l_kernel, mode=mode)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["fft", "chunk"])  # Recurrent mode does not compute powers of dA.
def test_bilinear_zero_transition(device, mode):
    """Check the bilinear discretization at and near dt A = -2, where dA is 0.

    With the layers' default dt range, dt A comes near -2 at the modes n >= 19 of diag-real.
    """
    A = torch.stack((-torch.arange(1.0, 9.0), torch.full((8,), -1e-4)), -1).repeat(4, 1, 1)
    dt = 2 / torch.tensor([5.0, 6.0, 7.0, 8.0])  # Mode h + 4 of feature h has Re(dt A) = -2.
    # The same modes of features 1 to 3 have dA near 0.
    dt[1:] *= torch.tensor([1 + 1e-6, 1 - 3e-5, 1 + 2e-4])
    _, _, B, C, D = s4d_params()
    u = torch.randn(3, 64, 4)
    params = [p.double().requires_grad_() for p in (dt, A, B, C, D)]
    y_ref = ref.run(*ref.s4d_matrices(*params[:4], "bilinear"), params[4], u)[0]
    args = [p.detach().float().to(device).requires_grad_() for p in params]
    y = s4d(u.to(device), *args, disc="bilinear", mode=mode)[0]
    assert_close(y, y_ref)
    gy = torch.randn_like(y_ref)
    grads = torch.autograd.grad(y, args, gy.to(device, torch.float32))
    for g, g_ref in zip(grads, torch.autograd.grad(y_ref, params, gy), strict=True):
        assert_close(g, g_ref, tol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("disc", ["zoh", "bilinear"])
def test_slow_mode_gradients(device, disc):
    """Check the float32 gradients of S4D's convolution kernel at small dt A.

    Because the powers dA^t multiply the error of log dA by t, log dA and dB need a small relative
    error there.
    """
    dt, A, B, C, _ = s4d_params(H=16, N=32, init="diag-real")
    A[..., 1] = -1e-4  # The layers clamp the initial Im(A) to at most -1e-4.
    params = [p.double().requires_grad_() for p in (dt, A, B, C)]
    K_ref = s4d_conv_kernel(*params, 1024, disc)
    g = torch.randn_like(K_ref)
    grads_ref = torch.autograd.grad(K_ref, params, g)
    for fused in [False] if device == "cpu" else [False, None]:
        args = [p.detach().float().to(device).requires_grad_() for p in params]
        K = s4d_conv_kernel(*args, 1024, disc, fused_kernels=fused)
        grads = torch.autograd.grad(K, args, g.float().to(device))
        for x, x_ref in zip(grads, grads_ref, strict=True):
            assert_close(x, x_ref, tol=1e-6)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
def test_growing_modes():
    """Check `s4d_conv_kernel` on the accelerator at growing modes, where Re(dt A) > 1.

    For x = Re(dt A) > 1, the kernels compute exp(x) - 1 without the difference 1 - tanh(x / 2),
    which is inaccurate when the tanh is near 1. Each feature is compared at its own scale, because
    the features grow at different rates. The backward kernels pad the gradient of K with zeros,
    and the powers of dA overflow at the padded positions.
    """
    growth = torch.tensor([1.0, 5.0, 9.0, 12.0, 16.0])
    dt, A = torch.ones(5), torch.stack((growth, torch.full((5,), 0.3)), -1)[:, None]
    B, C = torch.ones(5, 1, 2), torch.ones(1, 5, 1, 2)
    params = [t.double().requires_grad_() for t in (dt, A, B, C)]
    exact = s4d_conv_kernel(*params, 4)
    args = [t.to(ACCELERATOR).requires_grad_() for t in (dt, A, B, C)]
    K = s4d_conv_kernel(*args, 4, fused_kernels=True)
    scale = exact.detach().abs().amax(-1, keepdim=True)
    assert_close(K.cpu() / scale, exact / scale, tol=1e-5)
    g = torch.randn_like(exact) / scale
    grads = torch.autograd.grad(K, args, g.float().to(ACCELERATOR))
    for x, x_ref in zip(grads, torch.autograd.grad(exact, params, g), strict=True):
        assert_close(x, x_ref, tol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reverse_fft_one_step(device):
    """Check reverse fft mode at L = 1.

    At L = 1, the kernel of the reverse convolution has no taps after tap 0.
    """
    u = torch.randn(3, 1, 4)
    dt, A, B, C, D = s4d_params()
    y_ref = ref.run(*ref.s4d_matrices(dt, A, B, C, "zoh"), D, u, None, True)[0]
    assert_close(s4d(*to(device, u, dt, A, B, C, D), reverse=True, mode="fft")[0], y_ref)
    dt, Lam, P, B, C, D = s4_params()
    y_ref = ref.run(*ref.s4_matrices(dt, Lam, P, B, C, 128), D, u, None, True)[0]
    y = s4(*to(device, u, dt, Lam, P, B, C, D), l_kernel=128, reverse=True, mode="fft")[0]
    assert_close(y, y_ref)


def test_s4_final_state():
    """Compare S4's final state in fft mode with the reference.

    For the 600 steps, S4 combines the states of the chunks of 256 steps over two levels.
    """
    params = [x.double() for x in s4_params()[:5]]
    u = torch.randn(2, 600, 4, dtype=torch.float64)
    x0 = torch.randn(2, 4, 8, 2, dtype=torch.float64)
    x = s4(u, *params, initial_state=x0, output_final_state=True)[1]
    x_ref = ref.run(*ref.s4_matrices(*params, 600), None, u, x0)[1]
    assert_close(x, x_ref, tol=1e-10)


def test_s4d_state_gradients():
    params = [x.double().requires_grad_() for x in s4d_params()]
    u = torch.randn(3, 100, 4, dtype=torch.float64, requires_grad=True)
    x0 = torch.randn(3, 4, 8, 2, dtype=torch.float64, requires_grad=True)
    outs = s4d(u, *params, initial_state=x0, output_final_state=True)
    refs = ref.run(*ref.s4d_matrices(*params[:4], "zoh"), params[4], u, x0)
    gs = [torch.randn_like(o) for o in outs]
    inputs = [u, x0, *params]
    grads = torch.autograd.grad(outs, inputs, gs), torch.autograd.grad(refs, inputs, gs)
    for x, x_ref in zip([*outs, *grads[0]], [*refs, *grads[1]], strict=True):
        assert_close(x, x_ref, tol=1e-10)


def test_fftconv_bidirectional():
    L = 65  # The convolution needs at least 2 L = 130 points. FFTs of 128 points would wrap it.
    shapes = [(2, L, 3), (4, 3, L), (4, 3, L), (4, 3)]
    inputs = [torch.randn(s, dtype=torch.float64, requires_grad=True) for s in shapes]
    u, k, k_rev, D = inputs
    y = fftconv(u, k, D, k_rev)
    t = torch.arange(L)
    i = t[:, None] - t[None, :]
    causal = torch.where(i >= 0, k[..., i.clamp(min=0)], 0)
    anti = torch.where(i <= -1, k_rev[..., (-i - 1).clamp(min=0)], 0)
    y_ref = torch.einsum("chts,bsh->btch", causal + anti, u) + u[:, :, None] * D
    g = torch.randn_like(y)
    grads = torch.autograd.grad(y, inputs, g), torch.autograd.grad(y_ref, inputs, g)
    for x, x_ref in zip([y, *grads[0]], [y_ref, *grads[1]], strict=True):
        assert_close(x, x_ref, tol=1e-12)


def test_low_precision_params():
    u = torch.randn(2, 50, 4)
    dt, A, B, C, D = (t.bfloat16() for t in s4d_params(Cc=4))
    ops = [
        lambda: s4d(u, dt, A, B, C, D, mode="chunk"),
        lambda: s4d_conv_kernel(dt, A, B, C, 50),
        lambda: s4(u, *(t.bfloat16() for t in s4_params()), mode="recurrent"),
        lambda: s4_conv_kernel(*(t.bfloat16() for t in s4_params()[:5]), 50),
    ]
    for op in ops:
        with pytest.raises(ValueError, match="float32 or float64"):
            op()


def test_too_many_channels():
    """Check that every op raises for more than 4 kernel channels."""
    u = torch.randn(2, 50, 4)
    dt, A, B, C, D = s4d_params(Cc=5)
    ops = [
        lambda: s4d(u, dt, A, B, C, D),
        lambda: s4d_conv_kernel(dt, A, B, C, 50),
        lambda: s4(u, *s4_params(Cc=5)),
        lambda: s4_conv_kernel(*s4_params(Cc=5)[:5], 50),
        lambda: fftconv(u, torch.randn(5, 4, 50)),
    ]
    for op in ops:
        with pytest.raises(ValueError, match="more than 4 kernel channels"):
            op()


def test_empty_inputs():
    u = torch.randn(1, 10, 4)
    params = s4d_params()
    for empty in (u[:, :0], u[:0]):
        with pytest.raises(ValueError, match="non-empty"):
            s4d(empty, *params)
        with pytest.raises(ValueError, match="non-empty"):
            fftconv(empty, torch.randn(2, 4, 8))
    with pytest.raises(ValueError, match="at least one step"):
        s4d(u, *params, cu_seqlens=torch.tensor([0, 4, 4, 10]))
    with pytest.raises(ValueError, match="at least one kernel channel"):
        fftconv(u, torch.randn(0, 4, 8))


def test_fused_kernels_cpu():
    with pytest.raises(ValueError, match="do not support cpu tensors"):
        s4d(torch.randn(2, 50, 4), *s4d_params(), fused_kernels=True)
    with pytest.raises(ValueError, match="unknown fused_kernels"):
        s4d(torch.randn(2, 50, 4), *s4d_params(), fused_kernels="False")


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
def test_fused_kernels(monkeypatch):
    """Check that fused_kernels=True runs the kernels while `shared.DISABLED` is set, or raises."""
    monkeypatch.setattr(shared, "DISABLED", True)
    u, *params = to(ACCELERATOR, torch.randn(2, 50, 4), *s4d_params())
    y, y_default, y_pytorch = (s4d(u, *params, fused_kernels=f)[0] for f in (True, None, False))
    assert not torch.equal(y, y_pytorch) and torch.equal(y_default, y_pytorch)
    # A reverse step does not raise, because its kernel has no taps after tap 0.
    s4d(u[:, :1], *params, reverse=True, mode="fft", fused_kernels=True)
    with pytest.raises(ValueError, match="more than 32 modes"):
        state_powers(*to(ACCELERATOR, *s4_params(N=64)[:4]), 100, True)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
@pytest.mark.usefixtures("fresh_dynamo")
def test_fused_kernels_compiled():
    """Check that a compiled call raises the error for fused_kernels=True.

    Dynamo raises it as an Unsupported error.
    """
    u, *params = to(ACCELERATOR, torch.randn(2, 50, 4), *s4_params(N=64))
    s4_states = functools.partial(s4, output_final_state=True, fused_kernels=True)
    with pytest.raises(Exception, match="more than 32 modes"):
        torch.compile(s4_states, fullgraph=True)(u, *params)


def run_with_grads(fn, inputs, grad):
    if not grad:
        with torch.no_grad():
            return fn(*inputs), ()
    out = fn(*inputs)
    gen = torch.Generator().manual_seed(1)
    gs = [torch.randn(o.shape, generator=gen, dtype=torch.float64) for o in out]
    loss = sum((o * g.to(o)).sum() for o, g in zip(out, gs, strict=True))
    return out, torch.autograd.grad(loss, inputs)


def rel_error(x, y):
    return ((x.cpu().double() - y).abs().max() / y.abs().max()).item()


def assert_as_accurate(fn, inputs, grad=True):
    """Assert that the kernels are as accurate as PyTorch's float32 implementation.

    The PyTorch implementation in float32 has a relative error E against float64 in each output
    and gradient, and the relative error of the kernels must be at most max(1e-5, 2 E).

    The gradient of dt has one value per feature. With few features, the maximum error of that
    gradient varies between seeds by more than a factor of 2, so the tests of `s4d_conv_kernel`
    use 64 features.
    """
    kernels, pytorch = (functools.partial(fn, fused_kernels=f) for f in (True, False))
    exact = run_with_grads(pytorch, [t.double().requires_grad_() for t in inputs], grad)
    ins = [t.to(ACCELERATOR).requires_grad_() for t in inputs]
    ours, torch_fp32 = run_with_grads(kernels, ins, grad), run_with_grads(pytorch, ins, grad)
    flat = [[*outputs, *grads] for outputs, grads in (ours, torch_fp32, exact)]
    for a, b, e in zip(*flat, strict=True):
        E = rel_error(b, e)
        # The bound on E makes an error in PyTorch's code for the accelerator fail the test instead
        # of raising the bound of the kernels. With bfloat16 inputs, E reaches about 6e-3.
        assert E <= 2e-2 and rel_error(a, e) <= max(2 * E, 1e-5)


# Bounds on the relative errors against float64. The six bounds are for K and for the gradients of
# dt, Lambda, P, B and C. At seed 0, each bound is at least 4 times the largest error of the kernels
# and of PyTorch on the CPU, CUDA, ROCm and MPS. Each bound also exceeds 99% of the errors over 480
# draws of dt and C.
S4_BUDGETS = {
    "legs": (2e-5, 1e-3, 1e-4, 2e-4, 1e-4, 1e-4),
    # At the nearly undamped modes of fout, float32 rounding makes the gradients of dt and Lambda
    # inaccurate in every implementation. A sign error in these gradients still exceeds the bounds.
    "hippo": (1e-4, 0.5, 1e-2, 1e-5, 1e-5, 2e-3),
}


def assert_within(fn, inputs, device, budgets, reference=None):
    exact = run_with_grads(reference or fn, [t.double().requires_grad_() for t in inputs], True)
    ours = run_with_grads(fn, [t.to(device).requires_grad_() for t in inputs], True)
    flat = [[*outputs, *grads] for outputs, grads in (ours, exact)]
    for a, e, budget in zip(*flat, budgets, strict=True):
        assert a.isfinite().all() and rel_error(a, e) <= budget


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("init", S4_BUDGETS)
def test_s4_conv_kernel_accuracy(device, init):
    params = s4_params(H=4, N=128, Cc=2, init=init)[:5]
    assert_within(lambda *p: [s4_conv_kernel(*p, 1024)], params, device, S4_BUDGETS[init])


@pytest.mark.parametrize("device", DEVICES)
def test_s4_conv_kernel_dc(device):
    """Compare the sum of K and its gradients with -C~ (Lambda - P P^*)^-1 B.

    The sum of K is the bin k = 0 of its spectrum, which does not depend on dt. Because an error
    in that bin spreads over the L steps of K, the bound on K in test_s4_conv_kernel_accuracy does
    not detect it.
    """
    dt, Lam, P, B, C = s4_params(H=4, N=128, Cc=2)[:5]
    # With hippo's B, the solution x of (Lambda - P P^*) x = B is nearly real. A random B makes the
    # terms of Lambda's gradient that contain Im(x) large enough to test.
    params = [Lam, P, torch.randn_like(B), C]

    def total(*p):
        return [s4_conv_kernel(dt.to(p[0].device), *p, 1024).cpu().double().sum(-1)]

    def solve(Lam, P, B, C):
        Pf = ref.full(ref.cz(P))
        A = torch.diag_embed(ref.full(ref.cz(Lam))) - Pf[..., :, None] * Pf.conj()[..., None, :]
        return [-(ref.full(ref.cz(C)) * torch.linalg.solve(A, ref.full(ref.cz(B)))).sum(-1).real]

    assert_within(total, params, device, (2e-5,) * 5, reference=solve)


@pytest.mark.parametrize("device", DEVICES)
def test_s4_conv_kernel_adjoint(device):
    """Check that K's backward is the adjoint of its forward in C at a resonant mode.

    Because K is linear in C, <g, K(V)> = <grad_C <g, K(C)>, V> for every V. In each feature,
    mode n is resonant at a different frequency k. If the forward and the backward round
    |w|^2 - nu^2 differently, the residual of the identity is about 1e-2, and when they round it
    alike, the residual stays below 5e-7. The larger residual depends on the rounding error of
    nu^2. That error depends on the bits of nu, which differ between frequencies and between
    platforms.
    """
    L, N, Cc, n = 1024, 128, 2, 105
    k = 300 + 50 * torch.arange(4, dtype=torch.float64)
    H = len(k)
    nu = 2 * torch.tan(math.pi / L * k)
    dt = torch.full((H,), 1 / 32)
    Lam = torch.stack((torch.full((H, N), -0.5), -torch.arange(1.0, N + 1).expand(H, N)), -1)
    # |w| = nu. The layers clamp the initial Re(Lambda) to at most -1e-4.
    Lam[:, n, 0], Lam[:, n, 1] = -1e-4, -nu / dt
    B, P = torch.zeros(H, N, 2), torch.zeros(H, N, 2)
    B[:, n, 0], P[:, n, 0] = 1, 1 / 64
    g = torch.cos(2 * math.pi / L * k[:, None] * torch.arange(L))
    params = to(device, dt, Lam, P, B)
    C = torch.zeros(Cc, H, N, 2, device=device, requires_grad=True)
    loss = (s4_conv_kernel(*params, C, L) * g.float().to(device)).sum()
    gC = torch.autograd.grad(loss, C)[0].cpu().double()
    for part in range(2):
        V = torch.zeros(Cc, H, N, 2)
        V[:, :, n, part] = 1
        terms = s4_conv_kernel(*params, V.to(device), L).cpu().double() * g
        lhs, rhs = terms.sum(-1), gC[:, :, n, part]
        assert ((lhs - rhs).abs() <= 2e-5 * torch.maximum(terms.abs().sum(-1), rhs.abs())).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# The Cauchy kernels load the tables of the 150 modes in two chunks.
@pytest.mark.parametrize("N", [8, 150])
def test_s4_conv_kernel_tiles_cuda(monkeypatch, N):
    """Check the Cauchy backward when one thread block sums several tiles of frequencies.

    A thread block sums several tiles when the GPU has few SMs per feature. With one block per
    row, one block sums the 4 tiles of L = 1024.
    """
    s4_launchers = importlib.import_module("s4ever._backends.tilelang.s4")
    monkeypatch.setattr(s4_launchers, "blocks_per_row", lambda H, device: 1)
    params = s4_params(H=4, N=N, Cc=2, init="legs")[:5]
    assert_within(lambda *p: [s4_conv_kernel(*p, 1024)], params, "cuda", S4_BUDGETS["legs"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "N,Cc,dtype,reverse,disc,with_D",
    [
        (4, 2, "float32", False, "zoh", True),  # The 4 modes fill part of a block.
        (16, 1, "float32", True, "zoh", True),  # The 16 modes fill one block.
        # The 40 modes fill one block of 32 modes and part of a second.
        (40, 1, "bfloat16", True, "bilinear", True),
        (16, 1, "float16", False, "bilinear", True),
        (40, 2, "float32", False, "zoh", False),  # The two channels form one group.
    ],
)
@pytest.mark.parametrize("L", [150, 64, 20])  # The last chunk is partial, full, or the only chunk.
def test_chunk_cuda(N, Cc, dtype, reverse, disc, with_D, L):
    dt, A, B, C, D = s4d_params(H=4, N=N, Cc=Cc)
    # The 20 batch rows fill one block of 16 rows and part of a second.
    u = torch.randn(20, L, 4).to(getattr(torch, dtype))
    x0 = torch.randn(20, 4, N, 2)

    def fn(u, dt, A, B, C, x0, D=None, **kw):
        kw |= dict(initial_state=x0, output_final_state=True, reverse=reverse, mode="chunk")
        return s4d(u, dt, A, B, C, D, disc=disc, **kw)

    assert_as_accurate(fn, [u, dt, A, B, C, x0] + ([D] if with_D else []))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunk_gradients_repeat_cuda():
    params = [t.cuda() for t in s4d_params(H=64, N=32, Cc=1)]
    u, x0 = torch.randn(64, 256, 64, device="cuda"), torch.randn(64, 64, 32, 2, device="cuda")
    inputs = [t.requires_grad_() for t in (u, *params, x0)]
    y, x = s4d(*inputs[:-1], initial_state=x0, output_final_state=True, mode="chunk")
    gy, gx = torch.randn_like(y), torch.randn_like(x)
    first, *rest = (torch.autograd.grad((y, x), inputs, (gy, gx), retain_graph=True) for _ in "123")
    for grads in rest:
        assert all(torch.equal(a, b) for a, b in zip(first, grads, strict=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# At N = 64, the modes fill two blocks. At Cc = 2, the channels form two groups.
@pytest.mark.parametrize("N,Cc", [(64, 1), (8, 2)])
def test_chunk_partial_sums_cuda(N, Cc, dtype):
    """Check that the chunk kernels round y and du to half precision only once.

    The sums of du over the modes with C = 3 and with C = -3 exceed the range of float16 and
    cancel. At Cc = 1, the two sums of y also exceed the range and cancel.
    """
    # With 32 features and 32 steps, the transpose kernel transposes y and du.
    H = L = 32
    dt, A = torch.full((H,), math.log(2)), torch.tensor([-1.0, 0]).repeat(H, N, 1)
    B = -A
    s = torch.ones(Cc * N)
    s[Cc * N // 2 :] = -1
    C = torch.stack((3 * s.view(Cc, 1, N).expand(Cc, H, N), torch.zeros(Cc, H, N)), -1)
    u, gy = torch.full((1, L, H), 1e3, dtype=dtype), torch.full((1, L, Cc, H), 1e4, dtype=dtype)

    def run(*inputs):
        u = inputs[0].requires_grad_()
        y, _ = s4d(*inputs, torch.ones(Cc, H).to(inputs[1]), mode="chunk")
        return y, torch.autograd.grad(y, u, gy.to(y))[0]

    ours = run(*(t.cuda() for t in (u, dt, A, B, C)))
    for a, e in zip(ours, run(*(t.double() for t in (u, dt, A, B, C))), strict=True):
        assert rel_error(a, e) <= torch.finfo(dtype).eps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_final_gradient_cuda(dtype):
    """Check that du is rounded to half precision only once when the final state has a gradient.

    At the last step, the gradients through y and through the final state exceed the range of
    float16 and cancel.
    """
    H = L = 32
    dt, A = torch.full((H,), math.log(2)), torch.tensor([-1.0, 0]).repeat(H, 8, 1)
    B = -A
    u, gy = torch.ones(1, L, H, dtype=dtype), torch.zeros(1, L, 1, H, dtype=dtype)
    gy[:, -1] = 8192
    gx = torch.tensor([-16384.0, 0]).repeat(1, H, 8, 1)

    def run(u, dt, A, B):
        u.requires_grad_()
        D = torch.full((1, H), 0.3).to(dt)
        y, x = s4d(u, dt, A, B, B[None], D, output_final_state=True, mode="chunk")
        return torch.autograd.grad((y, x), u, (gy.to(y), gx.to(x)))[0]

    du = run(*(t.cuda() for t in (u, dt, A, B)))
    assert rel_error(du, run(*(t.double() for t in (u, dt, A, B)))) <= torch.finfo(dtype).eps


def assert_tf32_close(fn, inputs, grad=True, blas=None):
    """Assert that results in TF32 are within about ten TF32 roundings of float64."""
    exact = run_with_grads(fn, [t.double().requires_grad_() for t in inputs], grad)
    with tf32(blas):
        ours = run_with_grads(fn, [t.cuda().requires_grad_() for t in inputs], grad)
    for a, e in zip([*ours[0], *ours[1]], [*exact[0], *exact[1]], strict=True):
        assert rel_error(a, e) <= 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunk_tf32_cuda():
    dt, A, B, C, D = s4d_params(H=4, N=16, Cc=2)
    u, x0 = torch.randn(3, 150, 4), torch.randn(3, 4, 16, 2)

    def fn(u, dt, A, B, C, D, x0):
        return s4d(u, dt, A, B, C, D, initial_state=x0, output_final_state=True, mode="chunk")

    assert_tf32_close(fn, [u, dt, A, B, C, D, x0])


def s4_final_state(C):
    def fn(u, dt, Lam, P, Bm, x0, **kw):
        return s4(u, dt, Lam, P, Bm, C.to(dt), initial_state=x0, output_final_state=True, **kw)[1:]

    return fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "B,L,N,dtype",
    [
        # Each batch row is split into two groups. The states of the 21 modes are padded.
        (2, 2100, 21, "float32"),
        (1, 4096, 21, "float32"),  # Each batch row is split into four groups.
        # The only group has one chunk, which is longer than the input. On AMD, the state dimension
        # of 16 is padded to 32.
        (3, 20, 8, "bfloat16"),
        (20, 1100, 32, "float32"),  # The 20 batch rows need two thread blocks.
    ],
)
def test_s4_final_state_cuda(monkeypatch, B, L, N, dtype):
    spy = mock.Mock(wraps=tilelang.s4_group_states)
    monkeypatch.setattr(tilelang, "s4_group_states", spy)
    dt, Lam, P, Bm, C, _ = s4_params(H=4, N=N)
    u, x0 = torch.randn(B, L, 4).to(getattr(torch, dtype)), torch.randn(B, 4, N, 2)
    assert_as_accurate(s4_final_state(C), [u, dt, Lam, P, Bm, x0], grad=False)
    spy.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# At 16 modes, the kernels compute the powers. At 64 modes, torch.linalg.matrix_power computes them
# with cuBLASLt. The NoTF32Guard of matrix_power has no effect on cuBLASLt.
@pytest.mark.parametrize("N,blas", [(16, None), (64, "cublaslt")])
def test_s4_final_state_tf32_cuda(N, blas):
    dt, Lam, P, Bm, C, _ = s4_params(H=4, N=N)
    u, x0 = torch.randn(3, 1100, 4), torch.randn(3, 4, N, 2)
    assert_tf32_close(s4_final_state(C), [u, dt, Lam, P, Bm, x0], grad=False, blas=blas)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.usefixtures("fresh_dynamo")
def test_s4_recurrent_fallback_tf32_cuda(compiled):
    """Check PyTorch's recurrent steps of S4 at B = 64 under TF32.

    At B = 64, cuBLAS would run the matmuls of the recurrent steps in TF32, and their errors would
    accumulate over the steps.
    """
    dt, Lam, P, Bm, C, D = s4_params(H=64, N=32)
    inputs = [torch.randn(64, 16, 64), dt, Lam, P, Bm, C, D]

    def fn(*args):
        kw = dict(l_kernel=256, mode="recurrent", output_final_state=True, fused_kernels=False)
        return s4(*args, **kw)

    exact = fn(*(t.double() for t in inputs))
    f = torch.compile(fn, fullgraph=True) if compiled else fn
    cuda = [t.cuda() for t in inputs]
    with tf32():
        ours = f(*cuda)
    for a, e in zip(ours, exact, strict=True):
        assert a.dtype == torch.float32 and rel_error(a, e) <= 1e-4


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize(
    "l_kernel,B,N",
    [
        # On CUDA, the 9 states need one launch of eight states and a partial second launch. N = 21
        # is not a multiple of the thread counts or of the vector width.
        (301, 9, 21),
        (333, 9, 21),
        (301, 1, 40),  # The single state has more modes than the 32 lanes.
    ],
)
def test_s4_state_response(l_kernel, B, N):
    """Check S4's fft mode with initial states on the accelerator.

    The final state is checked only up to 32 modes, which is the most that S4's state kernels
    support.
    """
    dt, Lam, P, Bm, C, D = s4_params(H=4, N=N, Cc=2)
    u, x0 = torch.randn(B, 301, 4), torch.randn(B, 4, N, 2)
    final = N <= 32

    def fn(u, dt, Lam, P, Bm, C, D, x0, **kw):
        kw |= dict(l_kernel=l_kernel, initial_state=x0, output_final_state=final)
        y, x = s4(u, dt, Lam, P, Bm, C, D, **kw)
        return (y, x) if final else (y,)

    assert_as_accurate(fn, [u, dt, Lam, P, Bm, C, D, x0], grad=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("L,combine", [(4097, True), (4097, False), (70000, True)])
@pytest.mark.usefixtures("fresh_launchers")
def test_s4_state_cancellation_cuda(monkeypatch, L, combine):
    """Check float16 outputs when the response to the initial state cancels the convolution.

    The convolution of u reaches 1e5, which exceeds the range of float16, and the response to the
    initial state nearly cancels it. Split transforms add the response before rounding. With
    COMBINE on, the thread blocks combine their outputs. With COMBINE off, as on AMD,
    `combine_kernel` sums the outputs. At L = 70000, where the transforms run in three passes,
    `columns_inverse_kernel` sums the outputs even with COMBINE on.
    """
    monkeypatch.setattr(
        importlib.import_module("s4ever._backends.tilelang.kernels.fftconv"), "COMBINE", combine
    )
    u = torch.zeros(1, L, 1, dtype=torch.float16)
    u[:, 0] = 1000
    dt, A, B = torch.tensor([2 / 3]), torch.tensor([[[-1.0, -1e-4]]]), torch.tensor([[[1.0, 0.0]]])
    args = [u, dt, A, torch.zeros_like(A), B, torch.tensor([[[[100.0, 0.0]]]]), torch.ones(1, 1)]
    x0 = torch.tensor([[[[-999.0, 0.0]]]])
    with torch.no_grad():
        y = s4(*(t.cuda() for t in args), initial_state=x0.cuda())[0]
        exact = s4(*(t.double() for t in args), initial_state=x0.double())[0]
    assert_close(y, exact.half(), tol=2e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "L,N,Cc,H",
    [
        (4001, 21, 3, 4),
        # The tables hold at most 128 modes. At L = 64, the spectrum has a Nyquist bin. Some threads
        # have indices past the last frequency.
        (64, 128, 4, 4),
        (2, 8, 1, 4),
        # At L = 65536, the backward kernel splits the frequencies into 128 parts. The inverse
        # transform runs in two passes.
        (65536, 32, 2, 4),
        # The 201 modes and one mode of padding form two chunks of 101 modes. The frequencies are
        # split into parts of two tiles.
        (5000, 201, 2, 32),
    ],
)
def test_s4_conv_kernel_metal(L, N, Cc, H):
    """Check the Metal kernels of `s4_conv_kernel` against the budgets of legs.

    At power-of-two lengths, the Metal kernels also compute the inverse transform. The kernels run
    the same code at every init, and at legs their errors stay near float32's rounding, so one set
    of budgets holds at every configuration.
    """
    params = s4_params(H=H, N=N, Cc=Cc, init="legs")[:5]
    assert_within(lambda *p: [s4_conv_kernel(*p, L)], params, "mps", S4_BUDGETS["legs"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "L,N,Cc,disc",
    [
        # L = 1100 is not a multiple of the tile length. N = 40 is not a multiple of the SIMD group
        # width.
        (1100, 40, 3, "zoh"),
        # The tables of the forward kernel hold 128 modes.
        (300, 200, 4, "bilinear"),
        (70000, 32, 1, "zoh"),
    ],
)
def test_s4d_conv_kernel_metal(L, N, Cc, disc):
    params = s4d_params(H=64, N=N, Cc=Cc, init="diag-lin")[:4]
    assert_as_accurate(lambda *p, **kw: (s4d_conv_kernel(*p, L, disc, **kw),), params)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "N,Cc,B,dtype,reverse,disc,stateful",
    [
        # The state tiles include one tile of padding. The 9 batch rows need two row blocks of 8.
        (4, 2, 9, "float32", False, "zoh", "in, out"),
        # The 32 batch rows need two forward threadgroups of 16 rows.
        (40, 3, 32, "float32", True, "bilinear", "in, out"),
        # A mode group has at most 128 modes. The forward splits the 32 tiles of the mode group over
        # 2 SIMD groups.
        (128, 1, 3, "float32", False, "zoh", "in, out"),
        (32, 4, 3, "bfloat16", True, "zoh", "out"),
        # The 150 modes form two groups. For bf16 inputs, the kernels add the outputs of the groups
        # in float32.
        (150, 2, 9, "bfloat16", True, "bilinear", "in, out"),
        (16, 1, 3, "float16", False, "bilinear", "in"),
    ],
)
@pytest.mark.parametrize("L", [150, 64, 7])  # The last chunk is partial, full, or the only chunk.
def test_chunk_metal(N, Cc, B, dtype, reverse, disc, stateful, L):
    dt, A, Bm, C, D = s4d_params(H=4, N=N, Cc=Cc, init="diag-lin")
    u, x0 = torch.randn(B, L, 4).to(getattr(torch, dtype)), torch.randn(B, 4, N, 2)
    final = "out" in stateful

    def fn(u, dt, A, Bm, C, D, x0=None, **kw):
        kw |= dict(initial_state=x0, output_final_state=final, reverse=reverse, mode="chunk")
        y, x = s4d(u, dt, A, Bm, C, D, disc=disc, **kw)
        return (y, x) if final else (y,)

    assert_as_accurate(fn, [u, dt, A, Bm, C, D] + [x0] * ("in" in stateful))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "N,L",
    [
        (32, 1024),  # The kernels support state dimensions up to 2 N = 64.
        (8, 70000),  # The state dimension of 16 is padded.
    ],
)
def test_state_powers_metal(N, L):
    params = s4_params(H=4, N=N)[:4]

    def fn(*p, fused_kernels):
        In, Es = state_powers(*p, L, fused_kernels)
        return [In, *Es]

    assert_as_accurate(fn, params, grad=False)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize(
    "op,N,Cc,B,L,reverse,dtype",
    [
        ("s4d", 33, 3, 6, 70, False, "float32"),
        ("s4d", 256, 4, 5, 70, True, "float32"),
        ("s4", 40, 3, 6, 70, True, "float32"),
        ("s4", 256, 4, 5, 70, False, "float32"),
        ("s4d", 40, 3, 6, 70, True, "float16"),  # S4's random C-tilde overflows float16
        ("s4", 40, 3, 6, 70, False, "bfloat16"),
        ("s4d", 40, 3, 3, 1, False, "float32"),
        ("s4", 40, 3, 3, 1, True, "float32"),
    ],
)
def test_recurrent_kernels(op, N, Cc, B, L, reverse, dtype):
    """Compare the recurrent kernels with the float64 reference.

    The cases with 33 and 40 modes have more modes than the 32 lanes, and on MPS the recurrent
    kernels support at most 256 modes. The cases also cover several thread blocks and a partial
    last block of 32 steps. On CUDA, the cases cover a partial last group of features, and the
    single steps also cover a partial last group of batch rows.
    """
    H = 6
    u, x0 = torch.randn(B, L, H).to(getattr(torch, dtype)), torch.randn(B, H, N, 2)
    kw = dict(initial_state=x0.to(ACCELERATOR), output_final_state=True, reverse=reverse)
    kw |= dict(fused_kernels=True)
    if op == "s4d":
        dt, A, Bm, C, D = s4d_params(H=H, N=N, Cc=Cc, init="diag-lin")
        D = None if L == 1 else D
        y_ref, x_ref = ref.run(*ref.s4d_matrices(dt, A, Bm, C, "bilinear"), D, u, x0, reverse)
        y, x = s4d(*to(ACCELERATOR, u, dt, A, Bm, C, D), disc="bilinear", mode="recurrent", **kw)
    else:
        *params, D = s4_params(H=H, N=N, Cc=Cc)
        D = None if L == 1 else D
        y_ref, x_ref = ref.run(*ref.s4_matrices(*params, L), D, u, x0, reverse)
        # Solving for c from random C-tildes is inaccurate in float32 at many modes.
        rec = (t.float() for t in s4_recurrence(*(t.double() for t in params), L))
        u, *params, D = to(ACCELERATOR, u, *params, D)
        y, x = run_s4_recurrence(u, params[:4], to(ACCELERATOR, *rec), D, **kw)
    assert y.dtype == u.dtype
    assert_close(y, y_ref, tol=2e-5 if u.dtype == torch.float32 else torch.finfo(u.dtype).eps)
    # The bound of 5e-5 covers the float32 error of the state after 70 steps of 256 modes. PyTorch's
    # state has the same error.
    assert_close(x, x_ref, tol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "L,N,Cc,disc",
    [
        # L = 2000 is not a multiple of the tile length and reaches the last rows of a tile. N = 40
        # is not a multiple of the thread counts. The backward threads handle 2 rows each.
        (2000, 40, 3, "zoh"),
        # The tiles have 64 rows. The backward threads handle 4 rows each.
        (5000, 32, 1, "zoh"),
        (300, 128, 4, "bilinear"),
    ],
)
def test_s4d_conv_kernel_cuda(L, N, Cc, disc):
    params = s4d_params(H=64, N=N, Cc=Cc, init="diag-lin")[:4]
    assert_as_accurate(lambda *p, **kw: (s4d_conv_kernel(*p, L, disc, **kw),), params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "L,B,C,rounds",
    # The cases with L > 4096 compile slowest. They are spaced apart in the list so that xdist does
    # not queue two of them on one worker. The thread block counts below are for NVIDIA GPUs, where
    # one thread block transforms at most 8192 points. On AMD, one thread block transforms at most
    # 4096 points, which doubles the thread blocks of each split transform.
    [
        (70000, 5, 1, 2),  # The transforms run in three passes over 32 thread blocks.
        (40, 3, 2, 1),
        (40000, 3, 1, 1),  # The transforms are split over 16 thread blocks.
        (300, 9, 2, 2),
        (20000, 1, 1, 1),  # The transforms are split over 8 thread blocks.
        (4096, 5, 2, 2),  # The transforms have 8192 points.
        (12000, 5, 1, 2),  # The transforms are split over 4 thread blocks.
        # The transforms have 32 points. On NVIDIA GPUs, they run in two stages. The kernels have
        # C H = 15 rows, which is more than the 10 taps.
        (10, 3, 3, 1),
        (5000, 3, 2, 2),  # The transforms are split over 2 thread blocks.
        (30, 3, 1, 2),
        (200, 5, 2, 1),
        (300000, 1, 2, 1),  # The transforms run in three passes over 128 thread blocks.
        (1000, 5, 2, 1),  # The transforms have 2048 points, as for L = 1024.
        # The transforms have 4096 points. The backward is one kernel up to 4096 points.
        (2000, 3, 1, 2),
    ],
)
@pytest.mark.usefixtures("fresh_launchers")
def test_fftconv_cuda(monkeypatch, L, B, C, rounds):
    """Check the FFT convolution kernels and their gradients against PyTorch.

    A first call on other inputs of these shapes runs before the checked call. A split transform
    that reused the counters and shares of an earlier call would then fail the check.
    """
    # `pairs_per_block` returns 1 for transforms split over more than two thread blocks.
    monkeypatch.setattr(
        fftconv_launchers,
        "pairs_per_block",
        lambda pairs, H, device, G, N1: 1 if N1 > 2 else rounds,
    )
    u, D = torch.randn(B, L, 5), torch.randn(C, 5)
    k, k_rev = torch.randn(C, 5, L), torch.randn(C, 5, L)
    first = [torch.randn_like(t, device="cuda", requires_grad=True) for t in (u, k, k_rev, D)]
    fftconv(first[0], first[1], first[3], first[2]).sum().backward()

    # Like the gradients of a mean-reduced loss, the output gradients are about 1e-6. An error that
    # does not scale with the output gradients is then large relative to the gradients.
    def fn(u, k, k_rev, D, **kw):
        return (1e-6 * fftconv(u, k, D, k_rev, **kw),)

    assert_as_accurate(fn, [u, k, k_rev, D])
    # When u does not need a gradient, the backward launches kernels that do not compute the
    # gradient of u.
    assert_as_accurate(lambda k, k_rev, D, **kw: fn(u.to(k), k, k_rev, D, **kw), [k, k_rev, D])
    # When no kernel needs a gradient, the backward launches kernels that compute only the gradient
    # of u.
    assert_as_accurate(lambda u, **kw: fn(u, *(t.to(u) for t in (k, k_rev, D)), **kw), [u])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("shape", [(3, 1000, 37), (2, 33, 4099), (1, 70000 * 32 + 5, 32)])
def test_transpose_cuda(dtype, shape):
    """Check the transpose kernel.

    The cases cover partial tiles on both axes and more than 65535 tiles along a sequence.
    """
    x = torch.randn(shape, device="cuda").to(getattr(torch, dtype))
    assert torch.equal(transpose.transpose(x), x.mT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_new_thread_cuda():
    """Check a launch from a thread that has made no CUDA call.

    Autograd's backward thread is such a thread.
    """
    x = torch.randn(2, 64, 64, device="cuda")
    # The output of this call is freed to the allocator's cache. The thread of the pool then
    # allocates without a CUDA call.
    transpose.transpose(x)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        assert torch.equal(pool.submit(transpose.transpose, x).result(), x.mT)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# Transforms of 300 steps run in one thread block. Transforms of 5000 steps are split.
@pytest.mark.parametrize("L,dtype", [(300, "bfloat16"), (5000, "float16")])
def test_fftconv_low_precision_cuda(L, dtype):
    u, D = torch.randn(3, L, 5).to(getattr(torch, dtype)), torch.randn(2, 5)
    k, k_rev = torch.randn(2, 5, L), torch.randn(2, 5, L)
    assert_as_accurate(
        lambda u, k, k_rev, D, **kw: (fftconv(u, k, D, k_rev, **kw),), [u, k, k_rev, D]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("L", [300, 5000])
def test_fftconv_without_D_cuda(L):
    u, k, k_rev = torch.randn(3, L, 5), torch.randn(2, 5, L), torch.randn(2, 5, L // 2)
    assert_as_accurate(lambda u, k, k_rev, **kw: (fftconv(u, k, None, k_rev, **kw),), [u, k, k_rev])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "L,B,C,rounds",
    [
        (70000, 3, 1, 1),  # The transforms run in three passes.
        (5000, 3, 2, 1),  # On NVIDIA GPUs, the transforms are split over 2 thread blocks.
        (12000, 5, 1, 2),  # On NVIDIA GPUs, the transforms are split over 4 thread blocks.
        (2000, 3, 1, 1),  # Transforms of 4096 points run in one thread block.
    ],
)
@pytest.mark.usefixtures("fresh_launchers")
def test_fftconv_leaves_cuda(monkeypatch, L, B, C, rounds):
    monkeypatch.setattr(
        importlib.import_module("s4ever._backends.tilelang.kernels.fftconv"), "COMBINE", False
    )
    test_fftconv_cuda(monkeypatch, L, B, C, rounds)


def assert_addend_accurate(L):
    """Assert that the kernels of `fftconv_with_addend` are as accurate as PyTorch.

    The inputs match S4's call with initial states and without gradients. The callers split the
    transforms, because `test_s4_state_response` covers the addend for unsplit ones.
    """
    # B, C and H exceed 1, because an addend read at the wrong batch, channel or feature equals the
    # right one when B, C or H is 1.
    u, D = torch.randn(3, L, 5), torch.randn(2, 5)
    # The causal kernel and the addend are longer than the input, as when l_kernel > L.
    k, ad = torch.randn(2, 5, L + 9), torch.randn(2, 3, 5, L + 9)

    def fn(u, k, D, ad, fused_kernels):
        return (fftconv_with_addend(u, k, D, None, ad, fused_kernels),)

    assert_as_accurate(fn, [u, k, D, ad], grad=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "L,combine",
    [
        (5000, True),  # On NVIDIA GPUs, the transforms are split over 2 thread blocks.
        (12000, False),  # On NVIDIA GPUs, the transforms are split over 4 thread blocks.
        (70000, True),  # The transforms run in three passes.
    ],
)
@pytest.mark.usefixtures("fresh_launchers")
def test_fftconv_addend_cuda(monkeypatch, L, combine):
    monkeypatch.setattr(
        importlib.import_module("s4ever._backends.tilelang.kernels.fftconv"), "COMBINE", combine
    )
    assert_addend_accurate(L)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_metal_compile_error():
    """Check that the ops warn when the Metal compiler rejects a library.

    They then use PyTorch unless fused_kernels=True.
    """
    with pytest.warns(UserWarning, match="test kernels cannot be built for Metal"):
        assert not metal.msl.builds("test kernels", metal.msl.library, "kernel void f(", {})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("shape", [(3, 1000, 37), (2, 33, 4099), (2, 100, 200)])
def test_transpose_metal(dtype, shape):
    """Check the Metal transpose kernel.

    The cases cover partial tiles on both axes and several tiles along both axes.
    """
    x = torch.randn(shape, device="mps").to(getattr(torch, dtype))
    assert torch.equal(metal.transpose.transpose(x), x.mT)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "L,B,C,dtype",
    [
        (40, 3, 2, "float32"),  # One threadgroup transforms several rows.
        (300, 9, 2, "float32"),
        (1000, 1, 1, "float32"),  # The single batch row has no second row to pair with.
        (3000, 5, 2, "float32"),  # The transforms have 2 NF points. The inputs are shorter than NF.
        (5000, 4, 2, "float32"),  # The transforms are split over 4 threadgroups.
        # The transforms run in three passes with columns of 16 points. The gradients are summed
        # over chunks of two pairs.
        (20000, 13, 2, "float32"),
        (300000, 3, 1, "float32"),  # The transforms run in three passes with columns of 256 points.
        (300, 9, 2, "bfloat16"),
        (4096, 5, 2, "float16"),
        (5000, 4, 1, "bfloat16"),
        (20000, 3, 2, "bfloat16"),
    ],
)
def test_fftconv_metal(L, B, C, dtype):
    u, D = torch.randn(B, L, 5).to(getattr(torch, dtype)), torch.randn(C, 5)
    k, k_rev = torch.randn(C, 5, L), torch.randn(C, 5, L)

    def fn(u, k, k_rev, D, **kw):
        return (fftconv(u, k, D, k_rev, **kw),)

    assert_as_accurate(fn, [u, k, k_rev, D])
    # When u does not need a gradient, the backward kernel does not compute the gradient of u.
    assert_as_accurate(lambda k, k_rev, D, **kw: fn(u.to(k), k, k_rev, D, **kw), [k, k_rev, D])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
@pytest.mark.parametrize(
    "L",
    [
        3000,  # The transforms have 2 NF points. The inputs are shorter than NF.
        5000,  # The transforms are split over 4 threadgroups.
        20000,  # The transforms run in three passes.
    ],
)
def test_fftconv_addend_metal(L):
    assert_addend_accurate(L)
