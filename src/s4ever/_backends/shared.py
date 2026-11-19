"""Constants and helpers that the backends, the ops and the layers share."""

import functools
import math
import os

import torch
from torch import Tensor

INPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# PI2 is 2 pi rounded to float32, and PI2_LO is its rounding error 2 pi - PI2 in float32. The
# kernels reduce a phase x by k periods as x - k PI2 - k PI2_LO, which subtracts k 2 pi more
# accurately than PI2 alone.
PI2 = torch.tensor(2 * math.pi, dtype=torch.float32, device="cpu").item()
PI2_LO = torch.tensor(2 * math.pi - PI2, dtype=torch.float32, device="cpu").item()
# The fused convolution-kernel ops support at most this many modes, as does the CUDA extension of
# the original S4.
MAX_CONV_KERNEL_MODES = 1024
# The layers accept a C of at most this many rows, and a bidirectional layer has two rows of C per
# channel. The authors of the original S4 recommend one channel, and their only experiment with
# more uses 2 bidirectional channels, which are 4 rows of C.
MAX_CHANNELS = 4
DISABLED = os.environ.get("S4EVER_DISABLE_KERNELS", "") not in ("", "0")
FUSED_KERNELS = (None, True, False)

LIB = torch.library.Library("s4ever", "DEF")


def ceil_div(a, b):
    """Return a / b rounded up, for integers."""
    return -(-a // b)


def check_fused_kernels(fused_kernels: bool | None) -> None:
    if fused_kernels not in FUSED_KERNELS:
        raise ValueError(f"unknown fused_kernels {fused_kernels!r}; choose from {FUSED_KERNELS}")


def wanted(fused_kernels: bool | None) -> bool:
    if fused_kernels is None:
        return not DISABLED
    check_fused_kernels(fused_kernels)
    return bool(fused_kernels)


def dynamo_constant(fn):
    """Mark fn as `torch.compiler.assume_constant_result` does.

    Dynamo then calls fn while it traces and uses the result as a constant. PyTorch's decorator
    would import Dynamo when s4ever is imported, so this function sets the mark itself.
    """
    vars(fn)["_dynamo_marked_constant"] = True
    return fn


def register_op(name: str, fake, device: str = "CUDA"):
    """Register the decorated function fn as the op `s4ever::<name>`, with `fake` for tracing.

    The returned function calls fn directly in eager mode, so that eager calls skip the op's
    dispatch, and it calls the registered op only while Dynamo traces. The op is registered for the
    dispatch key `device` with the Library API, which dispatches faster than
    torch.library.custom_op.
    """

    def register(fn):
        LIB.define(name + torch.library.infer_schema(fn, mutates_args=()))
        LIB.impl(name, fn, device)
        torch.library.register_fake(f"s4ever::{name}", fake, lib=LIB)
        op = getattr(torch.ops.s4ever, name).default

        @functools.wraps(fn)
        def call(*args):
            return op(*args) if torch.compiler.is_compiling() else fn(*args)

        return call

    return register


def contiguous_f32(*params):
    """Return the params as contiguous float32 tensors, with None passed through.

    A param that is already a contiguous float32 tensor is returned without a dispatcher call.
    """
    f32 = torch.float32
    return [
        p if p is None or p.dtype == f32 and p.is_contiguous() else p.float().contiguous()
        for p in params
    ]


def recurrent_outputs(u: Tensor, A: Tensor, C: Tensor) -> tuple[Tensor, Tensor]:
    """Allocate y (B, L, C, H) and the final states (B, H, N, 2) of a recurrent scan of u."""
    return u.new_empty(*u.shape[:2], C.shape[0], u.shape[2]), A.new_empty(u.shape[0], *A.shape)


def tracks_grad(*tensors: Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)


def refuse_double_backward() -> None:
    """Raise RuntimeError if a backward pass runs with grad mode on, as create_graph=True does.

    Autograd cannot differentiate s4ever's backward passes, because the backward kernels have no
    derivatives and the saved forward results have no autograd history. PyTorch's
    `once_differentiable` raises only when the gradient that the backward receives requires grad.
    For other gradients, such as the gradient of y.sum(), second derivatives would lack the terms
    of the layer without an error.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("s4ever's backward passes do not support create_graph=True")
