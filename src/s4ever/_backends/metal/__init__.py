"""Metal kernels for Apple GPUs.

`torch.mps.compile_shader` compiles each library on first use. The kernels compute in float32 and
read and write sequences in the dtype of the input. The chunk kernels write y and du in float32
instead when the modes form several groups (`chunk.out_dtype`).
"""

import functools
import warnings

import torch
from torch import Tensor

from s4ever._backends import shared

from . import cauchy, chunk, fft, powers, recurrent, transpose, vandermonde


def enabled(x: Tensor, fused_kernels: bool | None) -> bool:
    return (
        x.device.type == "mps"
        and x.dtype in shared.INPUT_DTYPES
        and shared.wanted(fused_kernels)
        and not unavailable()
    )


@shared.dynamo_constant
def unavailable() -> str:
    """Return the error message of the probe, or "" if the probe runs.

    The function is a Dynamo constant because Dynamo cannot trace the probe.
    """
    return probe()


@functools.cache
def probe() -> str:
    try:
        transpose.transpose(torch.ones(1, 2, 2, device="mps"))
    except Exception as e:  # Metal compilers and drivers fail in version-specific ways
        warnings.warn(
            "s4ever: the ops use PyTorch unless fused_kernels=True, because the Metal kernels "
            f"cannot be compiled or run: {e}",
            # The warning names the line in s4ever that first asked whether the kernels are
            # available.
            stacklevel=3,
        )
        # The repr is not empty when the exception has no message.
        return str(e) or repr(e)
    return ""


__all__ = [
    "cauchy",
    "chunk",
    "enabled",
    "fft",
    "powers",
    "recurrent",
    "transpose",
    "unavailable",
    "vandermonde",
]
