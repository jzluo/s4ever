import torch
from torch import Tensor

from s4ever._backends.shared import refuse_double_backward, register_op, tracks_grad

from . import fftconv
from .fftconv import (
    fftconv_backward_op,
    fftconv_op,
    spectra_conv_backward_op,
    spectra_conv_op,
    takes_spectra,
)
from .runtime import launcher, lazy, on_device, shape_cache, unused
from .s4 import (
    cauchy_kernel,
    s4_conv_kernel_backward_op,
    s4_conv_kernel_op,
    s4_spectrum_backward_op,
    s4_spectrum_op,
)
from .s4d import s4d_conv_kernel_backward_op, s4d_conv_kernel_op, vandermonde_kernels
from .transpose import transpose


def layer_outputs(p, A_real, A_imag, X, rate, dt_kind, real_kind, imag_kind, grad):
    """Allocate dt, A and the repeated X of `layer_params`."""
    shape = (p.shape[0], A_real.shape[1], 2)
    return [torch.empty_like(p), *(p.new_empty(shape) for _ in range(1 + len(X)))]


def threads_per_column(H: int, S: int) -> int:
    """Return how many threads sum one column in `kernels.parameter_transforms.backward_kernel`.

    A column holds the gradients of one entry of an SSM's parameter from the H / S features that
    share the SSM.
    """
    return min(32, 1 << (H // S - 1).bit_length())


@register_op("layer_params", layer_outputs)
def layer_params_op(
    p: Tensor,
    A_real: Tensor,
    A_imag: Tensor,
    X: list[Tensor],
    rate: float,
    dt_kind: str,
    real_kind: str,
    imag_kind: str,
    grad: bool,
) -> list[Tensor]:
    out = layer_outputs(p, A_real, A_imag, X, rate, dt_kind, real_kind, imag_kind, grad)
    X2, Y2 = (X[1], out[3]) if len(X) == 2 else (unused(p.device, (1, 1, 2)),) * 2
    column_threads = threads_per_column(p.shape[0], A_real.shape[0]) if grad else 0
    fn = transforms_kernel(p.device, (dt_kind, real_kind, imag_kind), len(X), column_threads)
    fn(p, A_real, A_imag, X[0], X2, *out[:3], Y2, rate)
    return out


@launcher
def transforms_kernel(device, kinds, nX, column_threads):
    """Return a function that launches the kernel of `layer_params_op`.

    With column_threads > 0, the launcher also builds `transforms_backward_kernel`, so that the
    forward and backward kernels compile in parallel.
    """
    from s4ever._backends.tilelang.kernels import parameter_transforms

    if column_threads:
        transforms_backward_kernel(device, kinds, nX, column_threads)
    return lazy(parameter_transforms.forward_kernel, kinds, nX)


@launcher
def transforms_backward_kernel(device, kinds, nX, column_threads):
    from s4ever._backends.tilelang.kernels import parameter_transforms

    return lazy(parameter_transforms.backward_kernel, kinds, nX, column_threads)


def layer_grads(p, A_real, A_imag, gdt, gA, gY, rate, dt_kind, real_kind, imag_kind):
    grads = [torch.empty_like(t) for t in (p, A_real, A_imag)]
    return grads + [A_real.new_empty(*A_real.shape, 2) for _ in gY]


@register_op("layer_params_backward", layer_grads)
def layer_params_backward_op(
    p: Tensor,
    A_real: Tensor,
    A_imag: Tensor,
    gdt: Tensor,
    gA: Tensor,
    gY: list[Tensor],
    rate: float,
    dt_kind: str,
    real_kind: str,
    imag_kind: str,
) -> list[Tensor]:
    grads = layer_grads(p, A_real, A_imag, gdt, gA, gY, rate, dt_kind, real_kind, imag_kind)
    gY2, gX2 = (gY[1], grads[4]) if len(gY) == 2 else (unused(p.device, (1, 1, 2)),) * 2
    column_threads = threads_per_column(p.shape[0], A_real.shape[0])
    kinds = (dt_kind, real_kind, imag_kind)
    fn = transforms_backward_kernel(p.device, kinds, len(gY), column_threads)
    fn(p, A_real, A_imag, gdt, gA, gY[0], gY2, *grads[:4], gX2, rate)
    return grads


class LayerParams(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kinds, rate, grad, p, A_real, A_imag, *X):
        ctx.save_for_backward(p, A_real, A_imag)
        ctx.args = rate, *kinds
        return tuple(layer_params_op(p, A_real, A_imag, list(X), rate, *kinds, grad))

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        p, A_real, A_imag = ctx.saved_tensors
        gdt, gA, *gY = (g.contiguous() for g in grads)
        grads = layer_params_backward_op(p, A_real, A_imag, gdt, gA, gY, *ctx.args)
        return None, None, None, *grads


@on_device
def layer_params(
    p: Tensor, A_real: Tensor, A_imag: Tensor, X: list[Tensor], rate: float, kinds: tuple
) -> list[Tensor]:
    """Return dt, A and the parameters X (S, N, 2) repeated over the features.

        dt = t(p) rate
        A = -(t(A_real) + i t(A_imag))

    The tuple `kinds` names the transforms t of dt, A_real and A_imag, and the parameters are
    float32.
    """
    grad = tracks_grad(p, A_real, A_imag, *X)
    # Without grad, the op runs without LayerParams. In torch 2.14, Dynamo binds ctx to the
    # Function's first argument when it inlines an autograd.Function whose forward receives two
    # tensors through its *args parameter, if grad mode is off or no input requires grad. With
    # S4's P and B as X, a fullgraph compile of LayerParams.apply then fails at
    # ctx.save_for_backward, because ctx is the tuple `kinds`. Torch 2.15 compiles it.
    if not grad:
        return layer_params_op(p, A_real, A_imag, X, float(rate), *kinds, False)
    return LayerParams.apply(kinds, float(rate), grad, p, A_real, A_imag, *X)


class LayerConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cfg, x, p, A_real, A_imag, C, D, *X):
        y, (x, dt, A, Y, Kf) = layer_conv_forward(cfg, x, p, A_real, A_imag, C, D, X)
        ctx.save_for_backward(x, p, A_real, A_imag, dt, A, *Y, C, Kf)
        ctx.cfg, ctx.has_D = cfg, D is not None
        return y

    @staticmethod
    def backward(ctx, *grads):
        refuse_double_backward()
        x, p, A_real, A_imag, dt, A, *Y, C, Kf = ctx.saved_tensors
        rate, kinds, (kind, arg), spectra, bidirectional, want_gp, want_gx = ctx.cfg
        (Bsz, H, L), channels = x.shape, Kf.shape[0]
        gy = transpose(grads[0]).view(Bsz, channels, H, L)
        flags = (ctx.has_D, want_gx, want_gp)
        if spectra:
            gx, gK, gD = spectra_conv_backward_op(x, gy, Kf, arg, bidirectional, *flags)
            grads_op = s4_spectrum_backward_op
        else:
            Lr = L if bidirectional else None
            gx, gk, gk_rev, gD = fftconv_backward_op(x, gy, Kf, L, Lr, *flags)
            gK = torch.cat((gk, gk_rev)) if bidirectional else gk
            grads_op = s4_conv_kernel_backward_op if kind == "s4" else s4d_conv_kernel_backward_op
        gx = transpose(gx) if want_gx else None
        if not want_gp:
            return None, gx, *[None] * (5 + len(Y))
        gdt, gA, *gY, gC = grads_op(dt, A, *Y, C, gK, arg)
        gp, gAr, gAi, *gX = layer_params_backward_op(p, A_real, A_imag, gdt, gA, gY, rate, *kinds)
        return None, gx, gp, gAr, gAi, gC, gD if ctx.has_D else None, *gX


def layer_conv_forward(cfg, x, p, A_real, A_imag, C, D, X):
    rate, kinds, (kind, arg), spectra, bidirectional, want_gp, want_gx = cfg
    dt, A, *Y = layer_params_op(p, A_real, A_imag, list(X), rate, *kinds, want_gp)
    x = transpose(x)
    L = x.shape[-1]
    if spectra:  # The kernel length arg is at least L.
        k_f = s4_spectrum_op(dt, A, *Y, C, arg, want_gp)
        y, Kf = spectra_conv_op(x, k_f, D, bidirectional, want_gx, want_gp)
    else:
        if kind == "s4":
            K = s4_conv_kernel_op(dt, A, *Y, C, arg, want_gp)[..., :L].contiguous()
        else:
            K = s4d_conv_kernel_op(dt, A, *Y, C, L, arg, want_gp)
        kernels = K.chunk(2) if bidirectional else (K, None)
        y, Kf = fftconv_op(x, *kernels, D, None, want_gx, want_gp)
    return transpose(y.flatten(1, 2)), (x, dt, A, Y, Kf)


@on_device
def layer_conv(
    x: Tensor,
    p: Tensor,
    A_real: Tensor,
    A_imag: Tensor,
    C: Tensor,
    D: Tensor | None,
    X: list[Tensor],
    rate: float,
    kinds: tuple,
    ssm: tuple,
    bidirectional: bool,
) -> Tensor:
    """Return a layer's fft-mode output (B, L, C H) from its float32 parameters.

    The tuple ssm is ("s4", the kernel length) or ("s4d", the discretization). In a bidirectional
    layer, the second half of the convolution kernels is k_rev of `s4ever.ops.fftconv`.
    """
    C = C.contiguous()  # Module.to(memory_format=...) changes the strides of 4-D parameters
    kind, arg = ssm
    Bsz, L, H = x.shape
    spectra = kind == "s4" and takes_spectra(L, arg, x.device)
    # Without want_gp, no parameter needs a gradient, and the backward computes only x's gradient.
    want_gp, want_gx = tracks_grad(p, A_real, A_imag, C, D, *X), tracks_grad(x)
    cfg = (float(rate), kinds, ssm, spectra, bidirectional, want_gp, want_gx)
    if not torch.compiler.is_compiling():  # Dynamo cannot trace the compile pool's locks
        args = ((Bsz, H, L), x.dtype, x.device, A_real.shape, C.shape[0], len(X), D is not None)
        prefetch_layer(kinds, ssm, spectra, bidirectional, want_gp, want_gx, *args)
    # Without grad, the forward runs without LayerConv, because torch 2.14's Dynamo cannot inline
    # LayerConv when gradients are not tracked (see `layer_params`).
    if not (want_gp or want_gx):
        return layer_conv_forward(cfg, x, p, A_real, A_imag, C, D, X)[0]
    return LayerConv.apply(cfg, x, p, A_real, A_imag, C, D, *X)


@shape_cache
def prefetch_layer(
    kinds, ssm, spectra, bidirectional, want_gp, want_gx, shape, dtype, device, SN, Cc, nX, has_D
):
    """Build the launchers of `LayerConv`'s ops, other than the transposes.

    The launchers are built with the arguments that the ops will pass, so that all the ops' kernels
    compile in parallel. Otherwise each op would start its compiles only after the previous op's
    kernels had run.
    """
    (kind, arg), (Bsz, H, L), (S, N) = ssm, shape, SN
    channels = Cc // (1 + bidirectional)
    conv_args = (dtype, has_D, bidirectional, want_gx, want_gp)
    if spectra:  # the spectra of length-arg kernels have arg // 2 + 1 frequencies
        fftconv.spectra(device, Bsz, channels, H, L, arg // 2 + 1, *conv_args)
    else:
        has = (True, bidirectional, has_D)
        fftconv.spectrum(device, Bsz, channels, H, L, dtype, has, False, want_gx, want_gp)
    if kind == "s4":
        cauchy_kernel(device, N, Cc, arg, want_gp)
    else:
        vandermonde_kernels(device, N, Cc, L, arg, want_gp)
    transforms_kernel(device, kinds, nX, threads_per_column(H, S) if want_gp else 0)
