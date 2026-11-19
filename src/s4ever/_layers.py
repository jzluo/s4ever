"""The S4 and S4D layers, and `param_groups` for their optimizer settings."""

import contextlib
import math
import threading
import warnings
import weakref
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal, get_args

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from s4ever._backends import shared, tilelang
from s4ever.ops import _complex as cx
from s4ever.ops._fftconv import fftconv
from s4ever.ops._s4 import (
    S4_MODES,
    S4Mode,
    dense,
    matrix_power,
    run_fft,
    run_s4_recurrence,
    s4_conv_kernel,
    s4_recurrence,
    state_powers,
)
from s4ever.ops._s4d import (
    S4D_MODES,
    Disc,
    S4DMode,
    check_disc,
    run_s4d_recurrence,
    s4d,
    s4d_conv_kernel,
    s4d_recurrence,
)
from s4ever.ops._shared import auto_mode, check_params, check_shapes, no_autocast

from ._initialization import S4DInit, S4Init, s4_init, s4d_init

type Transform = Literal["none", "exp", "relu", "sigmoid", "softplus"]
TRANSFORMS = get_args(Transform.__value__)
SSM_LR = {"dt": 1e-3, "A": 1e-3, "B": 1e-3}
GROWTH = threading.RLock()


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def softplus(p: Tensor) -> Tensor:
    """Return softplus(p) in a form that stays accurate under torch.compile on MPS.

    Softplus is max(p, 0) + log1p(u) for u = exp(-|p|). In PyTorch 2.15.0 and earlier,
    torch.compile on MPS computes log1p(u) as log(1 + u) (pytorch/pytorch#201377), which is
    inaccurate for small u because 1 + u is rounded. The value is therefore computed as
    log(w) + (u - (w - 1)) / w for w = 1 + u, where the second term corrects log(w) for the
    rounding error of w.
    """
    with torch.no_grad():
        u = (-p.abs()).exp()
        w = 1 + u
        exact = p.clamp_min(0) + w.log() + (u - (w - 1)) / w
    s = F.softplus(p)
    # The result has the gradient of F.softplus, because autograd through `exact` would add several
    # ops to each backward.
    return s + (exact - s.detach())


def transform(p: Tensor, kind: str) -> Tensor:
    """Map an unconstrained parameter to a value that is positive unless `kind` is 'none'."""
    if kind == "none":
        return p
    if kind == "exp":
        return p.exp()
    if kind == "relu":
        return F.relu(p) + 1e-4
    if kind == "sigmoid":
        return p.sigmoid()
    if kind == "softplus":
        return softplus(p)
    raise ValueError(f"unknown transform {kind!r}; choose from {TRANSFORMS}")


def inverse_transform(p: Tensor, kind: str) -> Tensor:
    """Return the parameter that `transform` maps to max(p, 1e-4), or 1e-4 above it for 'relu'."""
    p = p.clamp(min=1e-4)
    if kind in ("none", "relu"):
        return p
    if kind == "exp":
        return p.log()
    if kind == "sigmoid":
        if (p >= 1).any():
            raise ValueError("the sigmoid transform does not support initial values of 1 or more")
        return p.logit()
    if kind == "softplus":
        return p + (-p).expm1().neg().log()  # in float32, log(expm1(p)) overflows for p > 88
    raise ValueError(f"unknown transform {kind!r}; choose from {TRANSFORMS}")


def hyperparams(value: float | Mapping[str, float] | None, name: str) -> dict[str, float | None]:
    """Return the lr or wd setting of each group as a dict keyed by 'dt', 'A', 'B' and 'C'."""
    keys = ("dt", "A", "B", "C")
    if value is None or isinstance(value, (int, float)):
        return {k: None if k == "C" else value for k in keys}
    if unknown := set(value) - set(keys):
        raise ValueError(f"unknown {name} keys {sorted(unknown)}; choose from {keys}")
    return {k: value.get(k) for k in keys}


class FeatureDropout(nn.Dropout):
    """Dropout of entire features of (B, L, H) inputs."""

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout1d(input.mT, self.p, self.training).mT


def make_dropout(p: float, tie: bool) -> nn.Module:
    if p == 0:
        return nn.Identity()
    return FeatureDropout(p) if tie else nn.Dropout(p)


class SSM(nn.Module):
    """The base class of `S4` and `S4D`."""

    inv_dt: Tensor
    A_real: Tensor
    A_imag: Tensor
    B: Tensor
    C: Tensor
    _repeated: tuple[str, ...]  # the per-SSM parameters other than A_real and A_imag
    mode: S4DMode
    _modes: tuple[str, ...]
    _custom_params = False

    def __init_subclass__(cls, **kwargs):
        """Mark the subclasses of S4 and S4D that override `ssm_params`.

        Such a layer may compute its parameters from tensors added by the subclass. Its cache
        therefore tracks every tensor of the module, and its fft mode calls `ssm_params` instead
        of running `tilelang.layer_conv` on the stored parameters.
        """
        super().__init_subclass__(**kwargs)
        if "ssm_params" in cls.__dict__ and SSM not in cls.__bases__:
            cls._custom_params = True

    def ssm_params(self, rate: float = 1.0) -> tuple[Tensor, ...]: ...

    def __init__(
        self,
        d_model,
        d_state,
        *,
        channels,
        bidirectional,
        n_ssm,
        dt_min,
        dt_max,
        dt_transform,
        real_transform,
        imag_transform,
        lr,
        wd,
        dropout,
        tie_dropout,
        mode,
        fused_kernels,
        keep_cache,
    ):
        super().__init__()
        sizes = {"d_model": d_model, "d_state": d_state, "channels": channels, "n_ssm": n_ssm}
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f"{name}={size} must be positive")
        if d_model % n_ssm:
            raise ValueError(f"n_ssm={n_ssm} must divide d_model={d_model}")
        if d_state % 2:
            raise ValueError(f"d_state={d_state} must be even (states come in conjugate pairs)")
        most = shared.MAX_CHANNELS // (1 + bidirectional)
        if channels > most:
            when = " when bidirectional" if bidirectional else ""
            raise ValueError(
                f"{type(self).__name__} does not support more than {most} channels{when}, got "
                f"{channels}; increase d_model instead"
            )
        shared.check_fused_kernels(fused_kernels)
        self.d_model, self.d_state, self.channels = d_model, d_state, channels
        self.bidirectional, self.n_ssm, self.mode = bidirectional, n_ssm, mode
        self.fused_kernels, self.keep_cache = fused_kernels, keep_cache
        if mode != "auto":
            # This call only validates: `_mode` raises ValueError for an unknown mode, and for a
            # mode other than fft in a bidirectional layer.
            self._mode(1, False)
        self.dt_transform, self.real_transform, self.imag_transform = (
            dt_transform,
            real_transform,
            imag_transform,
        )
        self._lr, self._wd = hyperparams(lr, "lr"), hyperparams(wd, "wd")
        self._overrides: dict[str, dict] = {}
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self._register("inv_dt", inverse_transform(log_dt.exp(), dt_transform), "dt")
        C = torch.randn(channels * (1 + bidirectional), d_model, d_state // 2, dtype=torch.cfloat)
        self._register("C", torch.view_as_real(C), "C")
        self.D = nn.Parameter(torch.randn(channels, d_model))
        self.dropout = make_dropout(dropout, tie_dropout)
        # `_cached` stores each (key, stream, value) entry under key[0], such as "params" or
        # "kernel", and `_end` stores a weak reference to the last call's final state under "state".
        self._cache: dict = {}
        self._cache_versions: list = []
        # Captured CUDA graphs read cached values, so the layer keeps a reference to each of them
        # after the cache is cleared.
        self._captured: list = []

    def _register(self, name: str, tensor: Tensor, group: str) -> None:
        """Register `tensor` as a parameter of `group`, or as a buffer if the group's lr is 0.

        The group's lr and weight decay are stored in `_overrides` under the parameter's name,
        which stays valid when FSDP or a copy of the module replaces the parameter. They are also
        set as the parameter's `_optim` attribute, which the training scripts of the original S4
        read.
        """
        lr, wd = self._lr[group], self._wd[group]
        if lr == 0.0:
            self.register_buffer(name, tensor)
            return
        self.register_parameter(name, nn.Parameter(tensor))
        optim = {k: v for k, v in (("lr", lr), ("weight_decay", wd)) if v is not None}
        if optim:
            self._overrides[name] = optim
            getattr(self, name)._optim = optim

    def _register_A(self, A: Tensor) -> None:
        self._register("A_real", inverse_transform(-A.real, self.real_transform), "A")
        self._register("A_imag", inverse_transform(-A.imag, self.imag_transform), "A")

    def _A(self) -> Tensor:
        A = cx.cplx(
            transform(self.A_real, self.real_transform),
            transform(self.A_imag, self.imag_transform),
        )
        return self._tile(-A)

    def _tile(self, x: Tensor) -> Tensor:
        """Tile x so that feature h uses SSM h % n_ssm, which is the order of the original S4."""
        n = self.d_model // self.n_ssm
        return x if n == 1 else x.expand(n, *x.shape).flatten(0, 1)

    def _dt(self, rate: float) -> Tensor:
        dt = transform(self.inv_dt, self.dt_transform)
        return dt if rate == 1 else dt * rate

    def _begin(self, x: Tensor, state: Tensor | None) -> None:
        """Clear the cache at the start of a call, unless the call may reuse it.

        A call may reuse the cache only with grad mode off, and only if `keep_cache` is set or
        `state` is the final state of the layer's last call. The version counters alone cannot
        tell whether the parameters changed, because some in-place updates do not advance them
        (see the notes of `S4D`). A CUDA graph capture also keeps the cache, so that the graph
        reads the values computed before the capture.
        """
        if torch.compiler.is_compiling() or not self._cache:
            return
        last = self._cache.get("state")
        continues = last is not None and state is not None and last() is state
        if (self.keep_cache or continues) and not torch.is_grad_enabled():
            return
        if not (x.is_cuda and torch.cuda.is_current_stream_capturing()):
            self._cache.clear()

    def _end(self, state: Tensor | None) -> Tensor | None:
        """Return `state` after noting it as this call's final state, for the next `_begin`."""
        if state is not None and not torch.compiler.is_compiling():
            self._cache["state"] = weakref.ref(state)
        return state

    def __getstate__(self) -> dict:
        """Return the module's state without its cache, whose weak reference cannot be pickled."""
        return {**super().__getstate__(), "_cache": {}, "_captured": []}

    def _cached(self, key: tuple, make: Callable[[], Any]) -> Any:
        """Return make(), cached under `key` while grad mode is off.

        On CUDA, an entry is reused only on the stream that computed it. Kernels on another stream
        could otherwise read an entry that has been dropped, after the caching allocator has given
        its memory to new tensors on the entry's own stream.

        A CUDA graph capture records make(), unless some stream already has the value cached. The
        graph then reads the cached value, which was computed from the parameters before the
        capture. Reading across streams is safe there, because torch.cuda.graph synchronizes
        before it captures.
        """
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return make()
        if self._custom_params:
            tensors = [*self.parameters(), *self.buffers()]
        else:
            tensors = [
                t for t in (*self._parameters.values(), *self._buffers.values()) if t is not None
            ]
        # Parameters made under torch.inference_mode have no version counter, so the values
        # computed from them are not cached.
        try:
            versions: list = [(t.data_ptr(), t._version) for t in tensors]
        except RuntimeError:
            if not any(t.is_inference() for t in tensors):
                raise
            return make()
        versions.append(self.fused_kernels)
        if versions != self._cache_versions:
            self._cache, self._cache_versions = {}, versions
        x = tensors[0]
        stream = torch.accelerator.current_stream(x.get_device()).stream_id if x.is_cuda else None
        # CUDA graphs capture only streams other than the default stream, whose id is 0.
        capturing = bool(stream) and torch.cuda.is_current_stream_capturing()
        kept = self._cache.get(key[0])
        if kept is not None and kept[0] == key and (kept[1] == stream or capturing):
            if capturing:
                self._captured.append(kept[2])
            return kept[2]
        if capturing:
            return make()  # a capture records the kernels of make() without running them
        kept = self._cache[key[0]] = key, stream, make()
        return kept[2]

    def _transformed(self, rate: float) -> Any:
        """Return `ssm_params` from the TileLang transforms kernel, or None where it cannot run."""
        inv_dt, A_real, A_imag = self.inv_dt, self.A_real, self.A_imag
        repeated = [getattr(self, n) for n in self._repeated]
        fp32 = all(t.dtype == torch.float32 for t in (A_real, *repeated))
        if not (tilelang.enabled(inv_dt, self.fused_kernels) and fp32):
            return None
        kinds = (self.dt_transform, self.real_transform, self.imag_transform)
        return (*tilelang.layer_params(inv_dt, A_real, A_imag, repeated, rate, kinds), self.C)

    def _params(self, rate: float) -> Any:

        def make():
            check_params(*self._parameters.values(), *self._buffers.values())
            return [p.contiguous() for p in self.ssm_params(rate)]

        return self._cached(("params", rate), make)

    @property
    def d_output(self) -> int:
        return self.channels * self.d_model

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, channels={self.channels}, "
            f"bidirectional={self.bidirectional}, n_ssm={self.n_ssm}, mode={self.mode!r}"
        )

    def _mode(self, L: int, grad: bool, stateful: bool = False) -> S4DMode:
        if self.mode == "auto":
            return "fft" if self.bidirectional else auto_mode(L, grad, stateful)
        if self.mode not in self._modes:
            raise ValueError(f"unknown mode {self.mode!r}; choose from {self._modes}")
        if self.bidirectional and self.mode != "fft":
            raise ValueError(
                f"bidirectional layers do not support {self.mode} mode; use mode='fft'"
            )
        return self.mode

    def _finish(self, y: Tensor) -> Tensor:
        return self.dropout(y.flatten(2))

    def _fused(self, x: Tensor) -> bool:
        """Return whether fft mode runs as one `tilelang.layer_conv` call.

        `tilelang.layer_conv` reads the stored parameters instead of calling `ssm_params`, so a
        layer with a custom `ssm_params` does not use it. With grad mode off, fft mode reads the
        cached convolution kernel instead.
        """
        if self._custom_params:
            return False
        tensors = (self.inv_dt, self.A_real, self.A_imag, self.C, self.D)
        tensors += tuple(getattr(self, n) for n in self._repeated)
        return (
            torch.is_grad_enabled()
            and tilelang.enabled(x, self.fused_kernels)
            and x.shape[1] <= tilelang.max_length(x.device)
            and self.d_state // 2 <= shared.MAX_CONV_KERNEL_MODES
            and all(t.dtype == torch.float32 for t in tensors)
        )

    def _layer_conv(self, x: Tensor, rate: float, ssm: tuple) -> Tensor:
        X = [getattr(self, n) for n in self._repeated]
        kinds = (self.dt_transform, self.real_transform, self.imag_transform)
        args = (self.C, self.D, X, rate, kinds, ssm, self.bidirectional)
        return self.dropout(tilelang.layer_conv(x, self.inv_dt, self.A_real, self.A_imag, *args))

    def _conv(self, x: Tensor, K: Tensor) -> Tensor:
        """Return the fft-mode output for the convolution kernels K (C, H, >= L)."""
        fk = self.fused_kernels
        if not self.bidirectional:
            return self._finish(fftconv(x, K, self.D, fused_kernels=fk))
        k, k_rev = K.chunk(2)
        return self._finish(fftconv(x, k, self.D, k_rev, fused_kernels=fk))


class S4D(SSM):
    """S4D layer: a diagonal SSM for each of the d_model features.

    The layer maps (B, L, d_model) to (B, L, channels * d_model), with the output features laid
    out as (channels, d_model).

    Args:
        d_state: the state size. It is even, because the modes come in conjugate pairs.
        channels: the number of output channels, each with its own C and D. A layer has at most 4
            channels, and a bidirectional layer at most 2. Recommended not to change this.
        bidirectional: whether a second SSM, with its own C, runs backward over the sequence.
        init: the initialization of A and B: 'diag-lin', 'diag-inv', 'diag-real', 'legs' (the
            diagonal part of S4's 'legs'), or 'diag', which uses 'diag-inv' for the first half of
            the SSMs and 'diag-lin' for the second half.
        n_ssm: the number of independent sets of A and B (and of P, in S4). Each set is shared by
            d_model / n_ssm features, while every feature has its own dt, C and D.
        disc: the discretization, 'zoh' or 'bilinear'.
        dt_min, dt_max: the range from which the initial dt is drawn log-uniformly.
        dt_transform, real_transform, imag_transform: the maps from the stored parameters to dt,
            -Re(A) and -Im(A). Each is 'none', 'exp', 'relu', 'sigmoid' or 'softplus'.
        lr, wd: the optimizer settings that `param_groups` applies. Each is a scalar, which
            applies to dt, A and B, or a dict keyed by 'dt', 'A', 'B' and 'C'. The settings for
            'A' also apply to S4's P. An lr of 0 freezes the tensor, and None leaves the
            optimizer's default in place.
        dropout, tie_dropout: dropout of the output. `tie_dropout` drops entire features.
        mode: 'auto', 'fft', 'chunk' or 'recurrent'. The default, 'auto', picks recurrent mode
            for a single step without gradients. Otherwise it picks chunk mode when states are
            given or requested, and fft mode when they are not. Bidirectional layers support only
            fft mode, which 'auto' picks for them at every length. Recurrent mode does not support
            gradients.
        fused_kernels: whether s4ever's GPU kernels run in place of PyTorch. `s4ever.ops.s4d`
            describes the values None, True and False.
        keep_cache: whether calls without gradients keep what they compute from the parameters,
            such as the convolution kernel, for later calls. Set it for inference with fixed
            parameters. By default, these values are kept only between the calls of one sequence
            of states.

    Notes:
        With `keep_cache`, the kept values are recomputed when a parameter's version counter
        advances, as it does after `load_state_dict`. Fused optimizers write the parameters
        without advancing the counters, as do AveragedModel's updates on CUDA. Calls after such
        an update still use the values from before it.
    """

    _repeated = ("B",)
    _modes = S4D_MODES

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        *,
        channels: int = 1,
        bidirectional: bool = False,
        init: S4DInit = "diag-lin",
        n_ssm: int = 1,
        disc: Disc = "zoh",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_transform: Transform = "softplus",
        real_transform: Transform = "exp",
        imag_transform: Transform = "none",
        lr: float | Mapping[str, float] | None = SSM_LR,
        wd: float | Mapping[str, float] | None = 0.0,
        dropout: float = 0.0,
        tie_dropout: bool = False,
        mode: S4DMode = "auto",
        fused_kernels: bool | None = None,
        keep_cache: bool = False,
    ):
        super().__init__(
            d_model,
            d_state,
            channels=channels,
            bidirectional=bidirectional,
            n_ssm=n_ssm,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_transform=dt_transform,
            real_transform=real_transform,
            imag_transform=imag_transform,
            lr=lr,
            wd=wd,
            dropout=dropout,
            tie_dropout=tie_dropout,
            mode=mode,
            fused_kernels=fused_kernels,
            keep_cache=keep_cache,
        )
        check_disc(disc)
        self.disc = disc
        A, B = s4d_init(init, d_state, n_ssm)
        self._register_A(A)
        self._register("B", torch.view_as_real(B), "B")

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, disc={self.disc!r}"

    def ssm_params(self, rate: float = 1.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return (dt, A, B, C) in the layout of `s4ever.ops.s4d`.

        A subclass may override this method, and the layer then uses the override's parameters on
        every device.
        """
        params = self._transformed(rate)
        if params is not None:
            return params
        return self._dt(rate), self._A(), self._tile(self.B), self.C

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        *,
        rate: float = 1.0,
        output_final_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Return (y, state) for x (B, L, d_model).

        The final state (B, d_model, d_state / 2, 2) is returned if a state is passed in or
        `output_final_state` is set, and is None otherwise. To step the layer, call it with L = 1.
        `rate` multiplies dt.
        """
        check_shapes(x, self.d_model, self.d_state // 2, state)
        self._begin(x, state)
        output_final_state = output_final_state or state is not None
        # Grad mode is checked first, so that calls with grad mode off do not traverse the
        # parameters.
        grad = torch.is_grad_enabled() and shared.tracks_grad(x, state, *self.parameters())
        mode = self._mode(x.shape[1], grad, output_final_state)
        if self.bidirectional and output_final_state:
            raise ValueError("bidirectional layers do not support state passing")
        if mode == "fft" and not output_final_state and self._fused(x):
            return self._layer_conv(x, rate, ("s4d", self.disc)), None
        dt, A, B, C = self._params(rate)
        fk = self.fused_kernels
        if mode == "fft" and not output_final_state:
            L = x.shape[1]
            make = partial(s4d_conv_kernel, dt, A, B, C, L, self.disc, fused_kernels=fk)
            return self._conv(x, self._cached(("kernel", rate, L), make)), None
        kw = dict(initial_state=state, output_final_state=output_final_state, fused_kernels=fk)
        if mode == "recurrent" and not grad:
            make = partial(s4d_recurrence, dt, A, B, C, self.disc)
            cached = partial(self._cached, ("recurrence", rate), make)
            y, state = run_s4d_recurrence(x, (dt, A, B, C), cached, self.D, disc=self.disc, **kw)
            return self._finish(y), self._end(state)
        y, state = s4d(x, dt, A, B, C, self.D, disc=self.disc, mode=mode, **kw)
        return self._finish(y), self._end(state)


class S4(SSM):
    """S4 layer: a diagonal plus low-rank SSM for each of the d_model features.

    The arguments other than `l_max` and `init` are those of `S4D`. S4 has no `disc` argument,
    because its discretization is always bilinear, and it has no chunk mode: 'auto' picks
    recurrent mode for a single step without gradients and fft mode otherwise.

    Args:
        l_max: the initial kernel length `l_kernel`. By default, the first call sets l_kernel to
            ceil(rate * L). C is stored as the C-tilde for l_kernel, so when an input is longer,
            l_kernel doubles until it covers the input and C is rewritten at each doubling. An
            l_max that covers the inputs avoids rewrites.
        init: the initialization of A, P and B: 'legs', 'fout', or 'hippo', which uses 'legs' for
            the first half of the SSMs and 'fout' for the second half.

    Notes:
        A rewrite changes the parameter C inside a forward call. It raises RuntimeError during a
        CUDA graph capture outside torch.compile, and a backward pass raises if C was rewritten
        after its forward pass. Rewrites are not safe across threads: while one call rewrites C,
        calls from other threads can read a C that does not match l_kernel, and two compiled calls
        that grow l_kernel at the same time can rewrite C twice, after which every output uses a
        C that does not match l_kernel. In distributed training each process would rewrite C for
        its own inputs, so the layer warns.
    """

    P: Tensor
    _repeated = ("P", "B")
    _modes = S4_MODES

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        *,
        l_max: int | None = None,
        channels: int = 1,
        bidirectional: bool = False,
        init: S4Init = "legs",
        n_ssm: int = 1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_transform: Transform = "softplus",
        real_transform: Transform = "exp",
        imag_transform: Transform = "none",
        lr: float | Mapping[str, float] | None = SSM_LR,
        wd: float | Mapping[str, float] | None = 0.0,
        dropout: float = 0.0,
        tie_dropout: bool = False,
        mode: S4Mode = "auto",
        fused_kernels: bool | None = None,
        keep_cache: bool = False,
    ):
        super().__init__(
            d_model,
            d_state,
            channels=channels,
            bidirectional=bidirectional,
            n_ssm=n_ssm,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_transform=dt_transform,
            real_transform=real_transform,
            imag_transform=imag_transform,
            lr=lr,
            wd=wd,
            dropout=dropout,
            tie_dropout=tie_dropout,
            mode=mode,
            fused_kernels=fused_kernels,
            keep_cache=keep_cache,
        )
        A, P, B = s4_init(init, d_state, n_ssm)
        self._register_A(A)
        self._register("P", torch.view_as_real(P), "A")
        self._register("B", torch.view_as_real(B), "B")
        self.l_kernel = 0
        if l_max:
            self._update_C(l_max, -1.0)

    def get_extra_state(self) -> dict:
        return {"l_kernel": self.l_kernel}

    def set_extra_state(self, state: dict) -> None:
        self.l_kernel = state["l_kernel"]

    def ssm_params(self, rate: float = 1.0) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return (dt, Lambda, P, B, C-tilde) in the layout of `s4ever.ops.s4`.

        A subclass may override this method, and the layer then uses the override's parameters on
        every device. The override's C is taken as the C-tilde for `l_kernel`.
        """
        params = self._transformed(rate)
        if params is not None:
            return params
        return self._dt(rate), self._A(), self._tile(self.P), self._tile(self.B), self.C

    @torch.no_grad()
    def _update_C(self, L: int, sign: float) -> None:
        """Multiply the stored C by (I + sign dA^L) in place.

        With sign = -1, the update converts the readout c into the C-tilde for length L. With
        sign = +1, it converts the C-tilde for length L into the C-tilde for length 2 L.
        """
        sync = self.C.is_cuda and not torch.compiler.is_compiling()
        # Every replay of a captured graph would update C again.
        if sync and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "S4 does not support inputs longer than its kernel length during a CUDA graph "
                "capture; pass l_max or call the layer at this length before capturing"
            )
        dt, A, P, B, _ = self.ssm_params()
        C = self.C  # the stored C-tilde rather than an override's C
        check_params(dt, A, P, B, C)
        # The device is synchronized before and after the update, so that calls on other streams do
        # not read C while it changes.
        if sync:
            torch.cuda.synchronize(C.device)
        with no_autocast(C):
            dA_L = matrix_power(dense(dt, A, P, B)[0], L)
            c_real = (2 * cx.conj(C)).flatten(-2)
            c_real = c_real + sign * torch.einsum("hst,chs->cht", dA_L, c_real)
        self.C.copy_(cx.conj(c_real.unflatten(-1, (-1, 2))) / 2)
        if sync:
            torch.cuda.synchronize(C.device)
        self.l_kernel = 2 * self.l_kernel if sign > 0 else L

    def _prepare(self, L: int, rate: float) -> int:
        """Grow l_kernel until it covers L steps at this rate, and return it in steps at this rate.

        GROWTH serializes the growth across threads, so that each rewrite of C happens once.
        """
        if self.l_kernel < rate * L:
            # Dynamo cannot trace the warning and the lock.
            compiling = torch.compiler.is_compiling()
            if not compiling and self.C.requires_grad and distributed():
                warnings.warn(
                    "S4 does not support inputs longer than its kernel length in distributed "
                    "training, because each process rewrites C for its own inputs; pass an l_max "
                    "that covers the inputs",
                    stacklevel=2,
                )
            with contextlib.nullcontext() if compiling else GROWTH:
                if self.l_kernel == 0:
                    self._update_C(math.ceil(rate * L), -1.0)
                while self.l_kernel < rate * L:
                    self._update_C(self.l_kernel, 1.0)
        return round(self.l_kernel / rate)

    def forward(
        self,
        x: Tensor,
        state: Tensor | None = None,
        *,
        rate: float = 1.0,
        output_final_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Return (y, state) for x (B, L, d_model), with the arguments of `S4D.forward`.

        The state is in the basis in which the state matrix is Lambda - P P^*. S4 does not support
        states while gradients are tracked.
        """
        check_shapes(x, self.d_model, self.d_state // 2, state)
        self._begin(x, state)
        L = x.shape[1]
        l_kernel = self._prepare(L, rate)
        output_final_state = output_final_state or state is not None
        grad = torch.is_grad_enabled() and shared.tracks_grad(x, *self.parameters())
        mode = self._mode(L, grad)
        if self.bidirectional and output_final_state:
            raise ValueError("bidirectional layers do not support state passing")
        if mode == "fft" and not output_final_state and self._fused(x):
            return self._layer_conv(x, rate, ("s4", l_kernel)), None
        dt, A, P, B, C = self._params(rate)
        fk = self.fused_kernels
        kw = dict(initial_state=state, fused_kernels=fk)
        if mode == "fft":
            make = partial(s4_conv_kernel, dt, A, P, B, C, l_kernel, fused_kernels=fk)
            K = self._cached(("kernel", rate, l_kernel), make)
            if not output_final_state:
                return self._conv(x, K), None
            make = partial(state_powers, dt, A, P, B, l_kernel, fk)
            powers = self._cached(("powers", rate, l_kernel), make)
            y, state = run_fft(x, (dt, A, P, B, C), K, powers, self.D, l_kernel, **kw)
            return self._finish(y), self._end(state)
        make = partial(s4_recurrence, dt, A, P, B, C, l_kernel)
        rec = self._cached(("recurrence", rate, l_kernel), make)
        y, state = run_s4_recurrence(
            x, (dt, A, P, B), rec, self.D, output_final_state=output_final_state, **kw
        )
        return self._finish(y), self._end(state)


def param_groups(model: nn.Module) -> list[dict]:
    """Return optimizer parameter groups that apply the lr and wd settings of S4 and S4D layers.

    The groups also apply the `_optim` dict of any other parameter, which is where the original
    S4 stores its overrides. The first group holds the parameters without overrides, so in
    `torch.optim.AdamW(param_groups(model), lr=1e-2, weight_decay=0.05)` these parameters use
    lr=1e-2 and weight_decay=0.05.
    """
    by_layer = {
        id(getattr(m, name)): optim
        for m in model.modules()
        if isinstance(m, SSM)
        for name, optim in m._overrides.items()
    }
    groups: dict[tuple, list[nn.Parameter]] = {(): []}
    for p in model.parameters():
        optim = by_layer.get(id(p), getattr(p, "_optim", {}))
        groups.setdefault(tuple(sorted(optim.items())), []).append(p)
    return [{"params": params, **dict(hp)} for hp, params in groups.items() if params]
