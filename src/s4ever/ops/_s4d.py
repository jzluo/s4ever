from functools import partial
from typing import Literal, get_args

import torch
import torch.nn.functional as F
from torch import Tensor

from s4ever._backends import metal, shared, tilelang
from s4ever._backends.shared import ceil_div

from . import _complex as cx
from ._fftconv import fft_mode
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

type S4DMode = Literal["auto", "fft", "chunk", "recurrent"]
type Disc = Literal["zoh", "bilinear"]
S4D_MODES = get_args(S4DMode.__value__)
DISCS = get_args(Disc.__value__)
CHUNK = 64


def check_options(mode: str, disc: str) -> None:
    if mode not in S4D_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {S4D_MODES}")
    check_disc(disc)


def check_disc(disc: str) -> None:
    if disc not in DISCS:
        raise ValueError(f"unknown discretization {disc!r}; choose from {DISCS}")


def discretize(dt: Tensor, A: Tensor, B: Tensor, disc: str = "zoh") -> tuple[Tensor, Tensor]:
    """Return (log dA, dB), each (H, N, 2)."""
    w = dt[:, None, None] * A
    if disc == "zoh":
        return w, cx.mul(B, cx.div(cx.expm1(w), A))
    den = cx.cplx(1 - w[..., 0] / 2, -w[..., 1] / 2)
    return 2 * cx.atanh(w / 2), cx.div(B * dt[:, None, None], den)


def split(L: int) -> tuple[int, int]:
    """Return T and L0, both about sqrt(L), so that each l < L is i T + j with i < L0 and j < T."""
    # Under torch.compile, int(L) specializes the graph on L, because T and L0 are sizes of tensors.
    L = int(L)
    T = 1 << (max(L - 1, 0).bit_length() + 1) // 2
    return T, ceil_div(L, T)


def power_tables(w: Tensor, L: int) -> tuple[Tensor, Tensor]:
    """Return Y = z^j (H, N, T, 2) and Z = z^(i T) (H, N, L0, 2) for z = exp(w)."""
    T, L0 = split(L)
    Y = cx.powers(w, torch.arange(T, device=w.device, dtype=w.dtype))
    return Y, cx.powers(w, torch.arange(L0, device=w.device, dtype=w.dtype) * T)


def gemm(W: Tensor, Y: Tensor, Z: Tensor, L: int) -> Tensor:
    """Return Re sum_n W[c, h, n] z_n^l (C, H, L) for l < L, from the Y and Z of `power_tables`."""
    C, H, N, _ = W.shape
    T, L0 = Y.shape[2], Z.shape[2]
    W = W.permute(1, 2, 0, 3)
    Wm = torch.stack((cx.conj(W), W.flip(-1)), 2).view(H * N, 2 * C, 2)  # (Wr, -Wi), (Wi, Wr)
    WZ = (Wm @ Z.view(H * N, L0, 2).mT).view(H, 2 * N, C * L0)
    Ys = torch.stack((Y[..., 0], -Y[..., 1]), 2).view(H, 2 * N, T)  # Re(WZ Y) = WZr Yr - WZi Yi
    return torch.matmul(WZ.mT, Ys).view(H, C, L0 * T).transpose(0, 1)[..., :L]


def moments(G: Tensor, Y: Tensor, Z: Tensor) -> Tensor:
    """Return sum_l G[h, c, l] z_n^l (C, H, N, 2) for G (H, C, L0 T).

    Y and Z are the outputs of `power_tables`.
    """
    H, C, _ = G.shape
    N, T, L0 = Y.shape[1], Y.shape[2], Z.shape[2]
    YG = torch.matmul(Y.transpose(2, 3).reshape(H, 2 * N, T), G.reshape(H, C * L0, T).mT)
    M = (YG.view(H * N, 2 * C, L0) @ Z.view(H * N, L0, 2)).view(H, N, 2, C, 2)
    Mr, Mi = M.unbind(2)  # sum_i YG Z = (YGr Zr - YGi Zi, YGr Zi + YGi Zr)
    return cx.cplx(Mr[..., 0] - Mi[..., 1], Mr[..., 1] + Mi[..., 0]).permute(2, 0, 1, 3)


def weighted(G: Tensor) -> Tensor:
    """Return G and l G[l], concatenated along axis 1."""
    return torch.cat((G, G * torch.arange(G.shape[-1], device=G.device, dtype=G.dtype)), 1)


class Vandermonde(torch.autograd.Function):
    """`vandermonde` with a hand-written backward.

    The backward takes a few dozen ops where autograd's takes a few hundred, which matters on MPS,
    where each op runs in about the same time regardless of size. It computes the gradients gW and
    gw from the moments S_m of g, the gradient of the kernel K:

        K = 2 Re sum_n W z^l
        S_m = sum_l l^m g[l] z^l
        gW = 2 conj(S_0)
        gw = 2 conj(sum_c W S_1)
    """

    @staticmethod
    def forward(ctx, W, w, L):
        Y, Z = power_tables(w, L)
        ctx.save_for_backward(W, Y, Z)
        return gemm(2 * W, Y, Z, L)

    @staticmethod
    def backward(ctx, *grads):
        shared.refuse_double_backward()
        (W, Y, Z), (g,) = ctx.saved_tensors, grads
        G = F.pad(g.transpose(0, 1), (0, Y.shape[2] * Z.shape[2] - g.shape[-1]))
        # Under torch.compile, a backward is traced under the autocast that surrounds the compiled
        # call, so these matmuls would run in bf16. A `no_autocast` context would not prevent
        # that: in the traced backward torch.is_autocast_enabled() returns False, because autocast
        # is off when the Function is called, and `no_autocast` then does nothing.
        with torch.autocast(g.device.type, enabled=False):
            S0, S1 = moments(weighted(G), Y, Z).split(g.shape[0])
        return 2 * cx.conj(S0), 2 * cx.conj(cx.mul(W, S1).sum(0)), None


def vandermonde(W: Tensor, w: Tensor, L: int) -> Tensor:
    """Return 2 Re sum_n W[..., n] exp(w[n] l) (C, H, L) for l < L.

    The shapes are W (C, H, N, 2) and w (H, N, 2). With l = i T + j, one GEMM over i and one over
    j compute the sum without forming an (N, L) matrix.
    """
    return Vandermonde.apply(W, w, L)


def s4d_conv_kernel(
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    L: int,
    disc: Disc = "zoh",
    *,
    fused_kernels: bool | None = None,
) -> Tensor:
    """Return the convolution kernel (C, H, L) of the discretized SSM.

    The shapes and `fused_kernels` are as in `s4d`.
    """
    check_disc(disc)
    check_params(dt, A, B, C)
    check_channels(C)
    with no_autocast(dt):
        N = A.shape[1]
        fused = dt.dtype == torch.float32 and N <= shared.MAX_CONV_KERNEL_MODES
        if fused and tilelang.enabled(dt, fused_kernels):
            return tilelang.s4d_conv_kernel(dt, A, B, C, L, disc)
        if metal.enabled(dt, fused_kernels) and metal.vandermonde.fits(N, L, disc):
            return metal.vandermonde.s4d_conv_kernel(dt, A, B, C, L, disc)
        if fused_kernels:
            longest = None if dt.is_cuda else metal.vandermonde.MAX_LENGTH
            most = shared.MAX_CONV_KERNEL_MODES
            limits = dict(modes=N, most=most, length=L, longest=longest)
            raise unsupported(fused_kernels, "S4D's convolution kernels", dt, **limits)
        w, dB = discretize(dt, A, B, disc)
        return vandermonde(cx.mul(C, dB), w, L)


class Matmul(torch.autograd.Function):
    """a @ b for `chunk`, with a backward that keeps the gradient as the left operand."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, *grads):
        (a, b), (g,) = ctx.saved_tensors, grads
        need = ctx.needs_input_grad
        # Autocast is off for the reason given in `Vandermonde.backward`.
        with torch.autocast(g.device.type, enabled=False):
            # In the gradient of b, g is the left operand, because MKL's sgemm slows down when its
            # right operand contains denormals.
            return g @ b.mT if need[0] else None, (g.mT @ a).mT if need[1] else None


class Products(torch.autograd.Function):
    """U Kt + X Out by bmm over the features, for `chunk`."""

    @staticmethod
    def forward(ctx, U, Kt, X, Out):
        ctx.save_for_backward(U, Kt, X, Out)
        return torch.bmm(U, Kt) + torch.bmm(X, Out)

    @staticmethod
    def backward(ctx, *grads):
        (U, Kt, X, Out), (g,) = ctx.saved_tensors, grads
        # g is a permuted view of the gradient of y (B, L, C, H), in which neither stride of a
        # feature's matrix is 1. The CPU's bmm would copy such matrices one feature at a time, so
        # g is copied once here.
        g = g.contiguous()
        need = ctx.needs_input_grad
        # Autocast is off for the reason given in `Vandermonde.backward`, and g is the left operand
        # in the gradients of Kt and Out for the reason given in `Matmul.backward`: GELU's backward
        # produces denormals in the gradient of y for large negative inputs.
        with torch.autocast(g.device.type, enabled=False):
            return (
                torch.bmm(g, Kt.mT) if need[0] else None,
                torch.bmm(g.mT, U).mT if need[1] else None,
                torch.bmm(g, Out.mT) if need[2] else None,
                torch.bmm(g.mT, X).mT if need[3] else None,
            )


def stepper(z: Tensor):
    """Return the step (s, dx) -> z s + dx for z (H, N, 2) and states (H, B, N, 2).

    The product is written as z s = (Re z, Re z) s + (-Im z, Im z) flip(s), because with
    `_complex.mul` Inductor splits the steps into more kernels.
    """
    zr, zi = z[:, None].unbind(-1)
    zr, zi = torch.stack((zr, zr), -1), torch.stack((-zi, zi), -1)
    return lambda s, dx: torch.addcmul(dx, zr, s).addcmul_(zi, s.flip(-1))


def conj_dot(X: Tensor, G: Tensor) -> Tensor:
    """Return the sum of conj(X) G = (Xr Gr + Xi Gi, Xr Gi - Xi Gr) over axes 1 and 2."""
    P = (X * G).sum((1, 2))
    Q = (X * G.flip(-1)).sum((1, 2))
    return cx.cplx(P[..., 0] + P[..., 1], Q[..., 0] - Q[..., 1])


class States(torch.autograd.Function):
    """The states X (H, B, nC, N, 2) of `chunk` before each chunk.

    With `final`, the state after the last chunk is returned too. The first state s_0 is x0
    (H, B, N, 2), or zero if x0 is None. The backward computes the gradients g_k of the states
    s_k:

        s_k = X[:, :, k]
        s_(k+1) = z s_k + dX[:, :, k]
        g_k = gX[:, :, k] + conj(z) g_(k+1)
    """

    @staticmethod
    def forward(ctx, z, dX, x0, final):
        H, Bsz, nC, N, _ = dX.shape
        step = stepper(z)
        s = [dX.new_zeros(H, Bsz, N, 2) if x0 is None else x0.contiguous()]
        for k in range(nC - 1):
            s.append(step(s[-1], dX[:, :, k]))
        last = step(s[-1], dX[:, :, -1]) if final else None
        X = torch.stack(s, 2)
        ctx.save_for_backward(z, X)
        return X, last

    @staticmethod
    def backward(ctx, *grads):
        (z, X), (gX, g_last) = ctx.saved_tensors, grads
        H, Bsz, nC, N, _ = X.shape
        step = stepper(cx.conj(z))
        # The gradient g[i] of s_(nC - i) is also the gradient of dX[:, :, nC - 1 - i].
        g = [X.new_zeros(H, Bsz, N, 2) if g_last is None else g_last.contiguous()]
        for k in range(nC - 1, 0, -1):
            g.append(step(g[-1], gX[:, :, k]))
        need = ctx.needs_input_grad
        gx0 = step(g[-1], gX[:, :, 0]) if need[2] else None
        gdX = torch.stack(g[::-1], 2)
        gz = conj_dot(X, gdX) if need[0] else None  # the sum of conj(s_k) g_(k+1) over b and k
        return gz, gdX, gx0, None


def chunk(u, w, dB, C, D, x0, final):
    """Return y (B, L, C, H) and the final state of S4D's chunk mode.

    The sequence is split into chunks of T steps. Within a chunk, the outputs are the inputs times
    Kt, the T x T Toeplitz matrix of the convolution kernel, plus the response `Out` to the state
    at the start of the chunk. `States` carries the states from chunk to chunk through dA^T.
    """
    Bsz, L, H = u.shape
    Cc, N = C.shape[0], C.shape[2]
    T = min(CHUNK, L)
    nC = ceil_div(L, T)
    # On the CPU, the two copies below run faster than one transpose that puts the features
    # outermost.
    U = F.pad(u, (0, 0, 0, nC * T - L)).unflatten(1, (nC, T)).permute(0, 3, 1, 2).contiguous()
    U = U.transpose(0, 1).reshape(H, Bsz * nC, T)
    zp = cx.powers(w, torch.arange(T + 1, device=u.device, dtype=u.dtype))  # (H, N, T + 1, 2)
    W = cx.conj(cx.mul(C, dB)).transpose(0, 1).reshape(H, Cc, 2 * N)
    K = 2 * Matmul.apply(W, zp[..., :T, :].transpose(2, 3).reshape(H, 2 * N, T)).transpose(0, 1)
    if D is not None:
        K[..., 0] += D
    # Kt (C, H, T, T) holds K[i - j], and 0 for j > i. It is built with unfold, whose backward
    # scatters with a 1-D index, because the backward of a gather with a 2-D index fails in
    # Inductor's partitioned-scatter pass, which torch 2.14 enables by default on ROCm (fixed in
    # 2.15 by pytorch/pytorch#194366). At T = 1, Kt is K[..., None] instead. Inductor would fuse
    # the scatter in unfold's backward with a sum over the batch, and in PyTorch 2.15.0 and earlier
    # that fused kernel adds the sum once per thread on MPS (pytorch/pytorch#201455).
    Kt = K[..., None] if T == 1 else F.pad(K, (T - 1, 0)).unfold(-1, T, 1).flip(-1)
    In = cx.mul(zp[..., :T, :].flip(-2), dB[..., None, :])  # dA^(T-1-j) dB: input j -> chunk state
    In = In.transpose(1, 2).reshape(H, T, 2 * N)
    Out = 2 * cx.conj(cx.mul(C[..., None, :], zp[None, ..., 1:, :]))  # 2 conj(C dA^(i+1))
    dX = Matmul.apply(U, In).view(H, Bsz, nC, N, 2)
    r = L - (nC - 1) * T  # the number of steps in the last chunk
    x0 = None if x0 is None else x0.transpose(0, 1)
    X, x = States.apply(zp[..., T, :], dX, x0, final and r == T)
    Kt = Kt.permute(1, 3, 2, 0).reshape(H, T, T * Cc)
    Out = Out.permute(1, 2, 4, 3, 0).reshape(H, 2 * N, T * Cc)
    y = Products.apply(U, Kt, X.view(H, Bsz * nC, 2 * N), Out)
    y = y.view(H, Bsz, nC * T, Cc).permute(1, 2, 3, 0)[:, :L]
    if not final:
        return y, None
    if r < T:
        dx = Matmul.apply(U.view(H, Bsz, nC, T)[:, :, -1, :r], In[:, T - r :]).view(H, Bsz, N, 2)
        x = cx.mul(zp[:, None, :, r, :], X[:, :, -1]) + dx
    return y, x.transpose(0, 1).contiguous()


def s4d_recurrence(
    dt: Tensor, A: Tensor, B: Tensor, C: Tensor, disc: str = "zoh"
) -> tuple[Tensor, Tensor, Tensor]:
    """Return dA, dB and 2 conj(C), the factors of S4D's recurrent steps."""
    with no_autocast(dt):
        w, dB = discretize(dt, A, B, disc)
        return cx.exp(w), dB, 2 * cx.conj(C)


def s4d_steps(u, dA, dB, W, D, x0):
    """Run the steps x <- dA x + dB u, and return the outputs and the final states (B, H, N, 2).

    The outputs are the sums of W x over the modes and the (re, im) parts, which are 2 Re(C x).
    """
    Bsz, L, H = u.shape
    x = u.new_zeros(Bsz, H, *dB.shape[1:]) if x0 is None else x0.contiguous()
    compiled = torch.compiler.is_compiling()
    # Inductor fuses the compiled steps into one kernel. On MPS,
    # a fused kernel of many steps can need more than the 31 buffers that Metal allows. Compiled
    # steps there compute the outputs with an einsum, because Inductor does not fuse ops across
    # the matmul of an einsum.
    mps = compiled and u.device.type == "mps" and L > 1
    z = dA if compiled else torch.view_as_complex(dA)
    ys = []
    for t in range(L):
        x = cx.mul(z, x) if compiled else torch.view_as_real(torch.view_as_complex(x) * z)
        x = x.addcmul_(dB, u[:, t, :, None, None])
        ys.append(torch.einsum("chnr,bhnr->bch", W, x) if mps else (W * x[:, None]).sum((-2, -1)))
    # Stacking a single step would copy it.
    y = ys[0][:, None] if L == 1 else torch.stack(ys, dim=1)
    if D is not None:
        y = torch.addcmul(y, u[:, :, None], D)
    return y, x


def run_s4d_recurrence(
    u,
    params,
    make,
    D,
    *,
    disc="zoh",
    initial_state=None,
    output_final_state=False,
    reverse=False,
    fused_kernels,
):
    """Run S4D's recurrent mode for params (dt, A, B, C), without gradients.

    The PyTorch path gets the `s4d_recurrence` of params from make(), so that a layer can cache it.
    """
    dtype = u.dtype
    dt, A, B, C = params
    with no_autocast(u):
        args = (initial_state, output_final_state, reverse, disc)
        if dt.dtype == torch.float32 and tilelang.enabled(u, fused_kernels):
            return tilelang.s4d_recurrent(u, dt, A, B, C, D, *args)
        if metal.enabled(u, fused_kernels) and metal.recurrent.fits(A.shape[1], u.dtype):
            return metal.recurrent.s4d_recurrent(u, dt, A, B, C, D, *args)
        if fused_kernels:
            most = None if u.is_cuda else metal.recurrent.MAX_MODES
            raise unsupported(
                fused_kernels, "S4D's recurrent kernels", dt, u, modes=A.shape[1], most=most
            )
        u = u.flip(1) if reverse else u
        u = u.to(torch.promote_types(dt.dtype, torch.float32))
        x0 = None if initial_state is None else initial_state.to(u.dtype)
        dA, dB, W = make()
        y, x = s4d_steps(u, dA, dB, W, D, x0)
        y = y.flip(1) if reverse else y
    return y.to(dtype), x if output_final_state else None


def run_chunk(u, params, D, *, disc, initial_state, output_final_state, reverse, fused_kernels):
    """Run S4D's chunk mode for params (dt, A, B, C)."""
    dtype = u.dtype
    dt, A, B, C = params
    args = (initial_state, output_final_state, reverse, disc)
    if dt.dtype == torch.float32 and tilelang.enabled(u, fused_kernels):
        return tilelang.s4d_chunk(u, dt, A, B, C, D, *args)
    if metal.enabled(u, fused_kernels) and metal.chunk.fits(A.shape[1], len(C), disc, u.dtype):
        return metal.chunk.s4d_chunk(u, dt, A, B, C, D, *args)
    if fused_kernels:
        most = None if u.is_cuda else metal.chunk.MAX_MODES
        raise unsupported(fused_kernels, "S4D's chunk kernels", dt, u, modes=A.shape[1], most=most)
    if reverse:
        u = u.flip(1)
    u = u.to(torch.promote_types(dt.dtype, torch.float32))
    x0 = None if initial_state is None else initial_state.to(u.dtype)
    w, dB = discretize(dt, A, B, disc)
    y, x = chunk(u, w, dB, C, D, x0, output_final_state)
    if reverse:
        y = y.flip(1)
    return y.to(dtype), x


def s4d(
    u: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    *,
    disc: Disc = "zoh",
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    reverse: bool = False,
    mode: S4DMode = "auto",
    cu_seqlens: Tensor | None = None,
    fused_kernels: bool | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Run an S4D SSM over u (B, L, H), and return y (B, L, C, H) and the final state.

    The output is y = 2 Re(C x) + D u, in u's dtype. The shapes are dt (H,), A and B (H, N, 2),
    C (C, H, N, 2), D (C, H) and states (B, H, N, 2), where N counts one mode of each conjugate
    pair and the trailing axis is (re, im).

    Args:
        reverse: whether the sequence is processed from the end, so that each y[t] depends on
            u[t:]. The initial state then precedes u[L - 1], and the final state follows u[0].
        mode: 'auto', 'fft', 'chunk' or 'recurrent'. The default, 'auto', picks recurrent mode
            for a single step without gradients. Otherwise it picks chunk mode when states are
            given or requested, and fft mode when they are not. The fft mode does not support
            states, and recurrent mode does not support gradients.
        cu_seqlens: the boundaries of the sequences packed into u, which then has batch size 1.
            Sequence i is u[:, cu_seqlens[i] : cu_seqlens[i + 1]] and has at least one step.
            `initial_state` and the final state have one entry per sequence. The sequences run
            one after another, each as its own call, in the one mode that 'auto' picks for all of
            them.
        fused_kernels: whether s4ever's GPU kernels (TileLang on CUDA and ROCm, Metal on MPS) run
            in place of PyTorch. With None, the default, they run where they support the inputs,
            unless the environment variable S4EVER_DISABLE_KERNELS is set to a value other than
            0. With True, they run even when that variable is set, and the ops raise ValueError
            where the kernels do not support the inputs. With False, PyTorch runs.
    """
    check_options(mode, disc)
    check_params(dt, A, B, C, D)
    check_channels(C)
    bounds = packed(u, cu_seqlens)
    check_shapes(u, A.shape[0], A.shape[1], initial_state, bounds)
    with no_autocast(u):
        stateful = initial_state is not None or output_final_state
        grad = shared.tracks_grad(u, dt, A, B, C, D, initial_state)
        if mode == "auto":
            mode = auto_mode(u.shape[1], grad, stateful)
        if mode == "fft" and stateful:
            raise ValueError("S4D's fft mode does not support states; use mode='chunk'")
        if mode == "recurrent" and grad:
            raise ValueError(
                "S4D's recurrent mode does not support gradients; use mode='chunk' or 'fft'"
            )
        params = (dt, A, B, C)
        kw = dict(
            disc=disc,
            output_final_state=output_final_state,
            reverse=reverse,
            fused_kernels=fused_kernels,
        )
        if mode == "fft":
            K = s4d_conv_kernel(
                *params, longest_sequence(u, bounds), disc, fused_kernels=fused_kernels
            )

            def run(u, x0):
                return fft_mode(u, K, D, reverse, fused_kernels), None
        elif mode == "recurrent":
            make = partial(s4d_recurrence, *params, disc)

            def run(u, x0):
                return run_s4d_recurrence(u, params, make, D, initial_state=x0, **kw)
        else:

            def run(u, x0):
                return run_chunk(u, params, D, initial_state=x0, **kw)

        if bounds is None:
            return run(u, initial_state)
        return varlen(run, u, bounds, initial_state, output_final_state)
