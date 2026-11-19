"""The original S4 from baseline/s4 (built by setup.sh), without its training utilities."""

import logging
import sys
import types
from pathlib import Path

import torch

from s4ever import S4, S4D

REPO = Path(__file__).resolve().parents[2] / "baseline" / "s4"
sys.path.insert(0, str(REPO))
import src  # noqa: E402  (namespace package of the original repo)

# The kernel modules of the original S4 need only `get_logger` from `src.utils.train`.
src.utils = types.SimpleNamespace(train=types.SimpleNamespace(get_logger=logging.getLogger))
sys.modules["src.utils"] = src.utils  # pyrefly: ignore[unsupported-operation]
sys.modules["src.utils.train"] = src.utils.train

from src.models.hippo import hippo  # noqa: E402
from src.models.sequence.kernels import dplr, ssm  # noqa: E402
from src.models.sequence.kernels.fftconv import FFTConv  # noqa: E402

__all__ = ["dplr", "hippo", "original"]

assert ssm.has_cuda_extension, "the original CUDA extension failed to import"


def original(s4ever: S4 | S4D, l_max: int | None = None) -> FFTConv:
    """Return the original FFTConv layer with the configuration and parameters of `s4ever`."""
    s4 = isinstance(s4ever, S4)
    layer = FFTConv(
        s4ever.d_model,
        l_max=l_max,
        channels=s4ever.channels,
        bidirectional=s4ever.bidirectional,
        mode="dplr" if s4 else "diag",
        d_state=s4ever.d_state,
        dt_transform=s4ever.dt_transform,
        real_transform=s4ever.real_transform,
        imag_transform=s4ever.imag_transform,
        n_ssm=s4ever.n_ssm,
        verbose=False,
        **({} if s4 else {"disc": s4ever.disc}),
    )
    k = layer.kernel
    with torch.no_grad():
        k.inv_dt.copy_(s4ever.inv_dt[:, None])
        for name in ("A_real", "A_imag", "C"):
            getattr(k, name).copy_(getattr(s4ever, name))
        k.B.copy_(s4ever.B[None])
        layer.D.copy_(s4ever.D)
        if s4:
            k.P.copy_(s4ever.P[None])
            k.l_kernel.fill_(s4ever.l_kernel)
    return layer
