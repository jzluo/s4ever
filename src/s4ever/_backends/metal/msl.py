"""Helpers that build the Metal libraries from the .metal files in this package."""

import functools
import warnings
from pathlib import Path

import torch

from s4ever._backends.shared import PI2, PI2_LO, dynamo_constant

TYPES = {torch.float32: "float", torch.bfloat16: "bfloat", torch.float16: "half"}
# A dispatch that splits a sum into partial sums aims for about this many threadgroups, so that
# every core of the GPU has work.
GROUPS = 512


def source(*parts: str) -> str:
    """Join common.metal and `parts` into one MSL source.

    A part is the name of a .metal file of this package or MSL that the caller generated. The
    #line directives make the compiler's messages name the file and line of each part.
    """
    here = Path(__file__).parent
    texts = []
    for part in ("common.metal", *parts):
        is_file = part.endswith(".metal")
        text = (here / part).read_text() if is_file else part
        texts.append(f'#line 1 "{part if is_file else "generated"}"\n{text}')
    return "\n".join(texts)


def discretize(disc: str) -> str:
    """Return the prelude that discretize.metal needs.

    The prelude defines the flag ZOH for `disc` and the floats PI2 and PI2_LO, whose sum is 2 pi
    to about twice the float precision.
    """
    return (
        f"constant constexpr bool ZOH = {str(disc == 'zoh').lower()};\n"
        f"constant constexpr float PI2 = {PI2!r}f, PI2_LO = {PI2_LO!r}f;\n"
    )


def library(src: str, threads: dict[str, int]):
    """Compile src, and raise RuntimeError if a kernel cannot run its threads per threadgroup.

    `threads` maps kernel names to their threads per threadgroup. The Metal compiler lowers a
    kernel's limit on threads per threadgroup when the kernel uses many registers.
    """
    lib = torch.mps.compile_shader(src)
    for name, n in threads.items():
        if getattr(lib, name).max_threads_per_threadgroup < n:
            raise RuntimeError(
                f"{name} uses too many registers for {n} threads per threadgroup on this GPU"
            )
    return lib


def builds(what: str, build, *args) -> bool:
    """Return whether build(*args) runs without an error, and warn if it does not."""
    # torch.mps.compile_shader raises SyntaxError for a source that the Metal compiler rejects.
    try:
        build(*args)
    except (RuntimeError, SyntaxError) as e:
        # The warning skips five frames, which are this function, a module's `builds`, the wrapper
        # from `constant`, the module's `fits` and the ops function that calls `fits`. For S4D's
        # chunk mode, it then names the line of `s4ever.ops.s4d` that chose the mode.
        warnings.warn(
            "s4ever: the ops use PyTorch unless fused_kernels=True, because "
            f"{what} cannot be built for Metal: {e}",
            stacklevel=6,
        )
        return False
    return True


def constant(check):
    """Cache `check` and mark it so that Dynamo calls it while it traces and uses the result.

    Dynamo can trace neither the build nor a function that functools.cache wraps, so the mark is on
    a plain function that calls the cached check.
    """
    cached = functools.cache(check)

    @dynamo_constant
    @functools.wraps(check)
    def call(*args):
        return cached(*args)

    return call
