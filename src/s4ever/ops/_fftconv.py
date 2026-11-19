"""FFT convolution of sequences with the convolution kernels of the SSMs."""

import torch
import torch.nn.functional as F
from torch import Tensor

from s4ever._backends import metal, shared, tilelang
from s4ever._backends.shared import ceil_div
from s4ever._backends.tilelang.fftconv import fftconv as tilelang_fftconv

from ._shared import check_channels, unsupported


def pair(x: Tensor) -> Tensor:
    """Return the two halves of the batch x as the real and imaginary parts of one tensor."""
    return torch.complex(*x.chunk(2))


@shared.register_op(
    "mps_unpair", lambda z: z.new_empty(2 * z.shape[0], *z.shape[1:], dtype=torch.float32), "MPS"
)
def unpair(z: Tensor) -> Tensor:
    """Return the real and imaginary parts of z as the two halves of a batch.

    The function is an op, so that compiled code reads the parts from a contiguous real tensor. If
    Inductor read the parts from z, a flip of the result would become a kernel whose loads index the
    rows backward with a column stride of 2, and Metal's compiler drops most of the stores of such
    a kernel. Reversed and bidirectional convolutions flip the result.
    `_backends/metal/chunk.metal` describes a kernel that loses stores in the same way when its
    stores index the rows backward.
    """
    return torch.cat((z.real, z.imag))


def fft_size(L: int, mps: bool) -> int:
    """Return the number of points n >= 2 L of the transforms for length L.

    On MPS, n is the smallest power of two that is at least 2 L, because the Metal kernels of
    `metal.fft` transform only powers of two. Elsewhere n is the smallest of 2^k, 3 2^k and 5 2^k
    that is at least 2 L, because cuFFT runs slower per point at sizes with more factors of 3 or 5.
    """
    if mps:
        return 2 << (L - 1).bit_length()
    return min(f << (ceil_div(2 * L, f) - 1).bit_length() for f in (1, 3, 5))


class Conv(torch.autograd.Function):
    """The convolution y (B, L, C, H) of x (B, L, H) with kernel (C, H, n).

    The input x is zero-padded to n >= 2 L points before the circular convolution.

    The transforms run over the rows of (B, H, n) and (B, C, H, n) tensors, because
    transforms along L in (B, L, H) are slower on the CPU and on CUDA. On MPS they run along L, and
    the two halves x_a and x_b of an even batch are transformed together as x_a + i x_b, which the
    convolution maps to y_a + i y_b because the kernel is real.
    """

    @staticmethod
    def forward(ctx, x, kernel, keep):
        (B, L, H), n = x.shape, kernel.shape[-1]
        ctx.mps, ctx.n = x.device.type == "mps", n
        # On MPS, complex transforms run faster than inverse real transforms.
        ctx.pairs = B % 2 == 0 and ctx.mps
        if not ctx.mps:
            # flatten makes the padded rows 2-D, because Inductor lays out a padded 3-D tensor
            # features-last and rfft would then copy the tensor to rows.
            X = torch.fft.rfft(F.pad(x.transpose(1, 2), (0, n - L)).flatten(0, 1)).view(B, H, -1)
            K = torch.fft.rfft(kernel)
            if keep:
                ctx.save_for_backward(X, K)
            Y = X[:, None] * K
            del X, K
            y = torch.fft.irfft(Y, n=n)[..., :L].permute(0, 3, 1, 2)
            # y is a view of (B, C, H, n) storage. In eager training it is copied to a contiguous
            # tensor, for two reasons. A view would keep the n-point storage allocated while later
            # ops save y, and PyTorch's CPU kernels run the backwards of those ops as strided scalar
            # loops when y is a view and the gradients are contiguous. Under torch.compile y stays
            # a view, because the backwards run Inductor's kernels instead of PyTorch's.
            return y.contiguous() if keep and not torch.compiler.is_compiling() else y
        kernel = kernel.permute(2, 0, 1)  # (n, C, H)
        if ctx.pairs:
            X, K = torch.fft.fft(pair(x), n=n, dim=1), torch.fft.fft(kernel, dim=0)
        else:
            X, K = torch.fft.rfft(x, n=n, dim=1), torch.fft.rfft(kernel, dim=0)
        if keep:
            ctx.save_for_backward(X, K)
        Y = X[:, :, None] * K
        del X, K
        if ctx.pairs:
            return unpair(torch.fft.ifft(Y, dim=1)[:, :L])
        return torch.fft.irfft(Y, n=n, dim=1)[:, :L]

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        (X, K), (g,), n = ctx.saved_tensors, grads, ctx.n
        B, L, C, H = g.shape
        need = ctx.needs_input_grad
        gx = gk = None
        if not ctx.mps:
            G = torch.fft.rfft(F.pad(g.permute(0, 2, 3, 1), (0, n - L)).flatten(0, 2))
            G = G.view(B, C, H, -1)
            if need[0]:
                gx = torch.fft.irfft((G * K.conj()).sum(1), n=n)[..., :L].transpose(1, 2)
            if need[1]:
                gk = torch.fft.irfft((G * X[:, None].conj()).sum(0), n=n)
            return gx, gk, None
        g = g.flatten(2)  # MPS pads 4-D tensors slowly.
        # Re sum_t g_z[t] conj(x_z[t - i]) = sum_t g_a[t] x_a[t - i] + g_b[t] x_b[t - i]
        if ctx.pairs:
            G = torch.fft.fft(pair(g), n=n, dim=1).unflatten(2, (C, H))
            if need[0]:
                gx = unpair(torch.fft.ifft((G * K.conj()).sum(2), dim=1)[:, :L])
            if need[1]:
                # The first n rows of the unpaired transform are its real part.
                gk = unpair(torch.fft.ifft((G * X[:, :, None].conj()).sum(0), dim=0))[:n]
        else:
            G = torch.fft.rfft(g, n=n, dim=1).unflatten(2, (C, H))
            if need[0]:
                gx = torch.fft.irfft((G * K.conj()).sum(2), n=n, dim=1)[:, :L]
            if need[1]:
                gk = torch.fft.irfft((G * X[:, :, None].conj()).sum(0), n=n, dim=0)
        return gx, None if gk is None else gk.permute(1, 2, 0), None


def fftconv(
    u: Tensor,
    k: Tensor | None,
    D: Tensor | None = None,
    k_rev: Tensor | None = None,
    *,
    fused_kernels: bool | None = None,
) -> Tensor:
    """Convolve u (B, L, H) with kernels (C, H, Lk) by FFT, and return y (B, L, C, H).

        y[t] = sum_i k[i] u[t - i] + sum_i k_rev[i] u[t + 1 + i] + D u[t]

    Taps past L are unused. y has u's dtype, and the convolution runs in float32 unless the
    kernels are float64. `fused_kernels` behaves as in `s4d`.
    """
    return fftconv_with_addend(u, k, D, k_rev, None, fused_kernels)


def fft_mode(
    u: Tensor, K: Tensor, D: Tensor | None, reverse: bool, fused_kernels: bool | None
) -> Tensor:
    """Run the ops' fft mode for u (B, L, H) and the SSM's convolution kernel K (C, H, >= L).

    With `reverse`, y[t] = sum_i K[i] u[t + i] + D u[t].
    """
    K = K[..., : u.shape[1]]
    if reverse:
        D = K[..., 0] if D is None else D + K[..., 0]
        return fftconv(u, None, D, K[..., 1:], fused_kernels=fused_kernels)
    return fftconv(u, K, D, fused_kernels=fused_kernels)


def fftconv_with_addend(
    u: Tensor,
    k: Tensor | None,
    D: Tensor | None,
    k_rev: Tensor | None,
    ad: Tensor | None,
    fused_kernels: bool | None,
) -> Tensor:
    """Run `fftconv`, then add ad (C, B, H, >= L) if given, as y[b, t, c, h] += ad[c, b, h, t]."""
    L = u.shape[1]
    if 0 in u.shape:
        raise ValueError(f"expected a non-empty input of shape (B, L, H), got {tuple(u.shape)}")
    ref = k if k is not None else k_rev if k_rev is not None else D
    if ref is None:
        raise ValueError("fftconv needs at least one of k, k_rev, D")
    check_channels(ref)
    # In a reverse scan of one step, k_rev has no taps.
    taps = any(x is not None and x.shape[-1] > 0 for x in (k, k_rev))
    fused = (
        taps
        # The TileLang kernels read tap 0 of k and k_rev.
        and all(x is None or x.shape[-1] > 0 for x in (k, k_rev))
        and ref.dtype == torch.float32
        and tilelang.enabled(u, fused_kernels)
        and u.shape[1] <= tilelang.max_length(u.device)
    )
    if fused:
        k, k_rev = (x if x is None or x.shape[-1] <= L else x[..., :L] for x in (k, k_rev))
        return tilelang_fftconv(u, k, D, k_rev, ad)
    ad = None if ad is None else ad[..., :L].permute(1, 3, 0, 2)
    n = fft_size(L, u.device.type == "mps")
    kernel = None if k is None else F.pad(k[..., :L], (0, n - min(k.shape[-1], L)))
    if k_rev is not None:
        # In the circular kernel, k_rev[i] is at index n - 1 - i.
        k_rev = F.pad(k_rev[..., :L].flip(-1), (n - min(k_rev.shape[-1], L), 0))
        kernel = k_rev if kernel is None else kernel + k_rev
    dtype = torch.promote_types(ref.dtype, torch.float32)
    if kernel is None:
        assert D is not None
        y = u[:, :, None].to(dtype) * D
        return (y if ad is None else y + ad).to(u.dtype)
    kernel = kernel.to(dtype)
    if D is not None:
        kernel[..., 0] += D
    keep = shared.tracks_grad(u, kernel)
    if dtype == torch.float32 and metal.enabled(u, fused_kernels) and metal.fft.fits(n, u.dtype):
        # The Metal kernels read u and write y in u's dtype.
        return metal.fft.conv(u, kernel, keep, ad)
    if taps and fused_kernels:
        # The call to `tilelang.max_length` needs TileLang and a CUDA device.
        runs = tilelang.enabled(u, True)
        longest = tilelang.max_length(u.device) if runs else metal.fft.MAX_LENGTH
        raise unsupported(
            fused_kernels, "the FFT convolution kernels", ref, u, length=L, longest=longest
        )
    y = Conv.apply(u.to(dtype), kernel, keep)
    return (y if ad is None else y + ad).to(u.dtype)
