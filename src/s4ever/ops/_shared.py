import contextlib
from typing import Literal, overload

import torch
from torch import Tensor

from s4ever._backends import metal, tilelang
from s4ever._backends.shared import INPUT_DTYPES, MAX_CHANNELS, check_fused_kernels

PARAM_DTYPES = (torch.float32, torch.float64)


def check_params(*params: Tensor | None) -> None:
    """Raise ValueError for SSM parameters that are not float32 or float64.

    Below float32, the rounding error of dt and A shifts the phase of each mode's l-th power in
    proportion to l, which makes long kernels inaccurate. The other parameters have the same
    restriction so that a layer's parameters can share one dtype.
    """
    for p in params:
        if p is not None and p.dtype not in PARAM_DTYPES:
            raise ValueError(
                f"the ops do not support {p.dtype} SSM parameters; use float32 or float64 "
                f"parameters with {p.dtype} inputs or autocast"
            )


@overload
def auto_mode(L: int, grad: bool) -> Literal["fft", "recurrent"]: ...
@overload
def auto_mode(L: int, grad: bool, stateful: bool) -> Literal["fft", "chunk", "recurrent"]: ...
def auto_mode(L: int, grad: bool, stateful: bool = False) -> str:
    """Return the mode that mode='auto' picks.

    `stateful` says whether states of S4D are given or requested.
    """
    if L == 1 and not grad:
        return "recurrent"
    return "chunk" if stateful else "fft"


def check_channels(C: Tensor) -> None:
    if len(C) == 0:
        raise ValueError("the ops need at least one kernel channel")
    if len(C) > MAX_CHANNELS:
        raise ValueError(
            f"the ops do not support more than {MAX_CHANNELS} kernel channels, got {len(C)}; "
            "split the channels over several calls"
        )


def check_shapes(
    u: Tensor, H: int, N: int, state: Tensor | None, bounds: list[int] | None = None
) -> None:
    if u.ndim != 3 or u.shape[2] != H or 0 in u.shape:
        raise ValueError(f"expected a non-empty input of shape (B, L, {H}), got {tuple(u.shape)}")
    expected = (u.shape[0] if bounds is None else len(bounds) - 1, H, N, 2)
    if state is not None and state.shape != expected:
        raise ValueError(f"expected a state of shape {expected}, got {tuple(state.shape)}")


def unsupported(
    fused_kernels: bool | None,
    kernels: str,
    p: Tensor,
    u: Tensor | None = None,
    *,
    modes: int = 0,
    most: int | None = None,
    length: int = 0,
    longest: int | None = None,
) -> ValueError:
    """Return the ValueError for fused_kernels=True when `kernels` do not support the inputs.

    The dtypes are checked before the toolchain, because the toolchain check compiles probe
    kernels, which can take seconds.
    """
    check_fused_kernels(fused_kernels)
    device = p.device
    if device.type not in ("cuda", "mps"):
        reason = f"{device.type} tensors"
    elif u is not None and u.dtype not in INPUT_DTYPES:
        reason = f"{u.dtype} inputs"
    elif p.is_cuda and p.dtype != torch.float32:
        reason = f"{p.dtype} parameters"
    elif problem := (tilelang.unavailable(device) if p.is_cuda else metal.unavailable()):
        reason = f"{device} ({problem})"
    elif most is not None and modes > most:
        reason = f"more than {most} modes"
    elif longest is not None and length > longest:
        reason = f"more than {longest} steps"
    else:
        reason = f"these inputs on {device}"
    return ValueError(f"{kernels} do not support {reason}; use fused_kernels=None")


def no_autocast(x: Tensor):
    kind = x.device.type
    return (
        torch.autocast(kind, enabled=False)
        if torch.is_autocast_enabled(kind)
        else contextlib.nullcontext()
    )


def packed(u: Tensor, cu_seqlens: Tensor | None) -> list[int] | None:
    """Return the bounds of the sequences packed into u as a list, or None without `cu_seqlens`."""
    if cu_seqlens is None:
        return None
    if u.shape[0] != 1:
        raise ValueError(f"packed sequences need batch size 1, got {u.shape[0]}")
    bounds = cu_seqlens.tolist()
    if any(e <= s for s, e in zip(bounds[:-1], bounds[1:], strict=True)):
        raise ValueError("packed sequences need at least one step each")
    return bounds


def longest_sequence(u: Tensor, bounds: list[int] | None) -> int:
    if bounds is None:
        return u.shape[1]
    return max(e - s for s, e in zip(bounds[:-1], bounds[1:], strict=True))


def varlen(run, u, bounds, initial_state, output_final_state):
    """Run the op on each packed sequence in turn, and concatenate the outputs and the states."""
    ys, states = [], []
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        x0 = None if initial_state is None else initial_state[i : i + 1]
        y, x = run(u[:, s:e], x0)
        ys.append(y)
        states.append(x)
    return torch.cat(ys, dim=1), torch.cat(states) if output_final_state else None
