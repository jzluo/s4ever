import contextlib

import torch

ACCELERATOR = torch.accelerator.current_accelerator(check_available=True)
DEVICES = ["cpu"] + ([ACCELERATOR.type] if ACCELERATOR else [])


def assert_close(x, y, tol=2e-5):
    x, y = x.cpu().double(), y.cpu().double()  # MPS has no float64
    err, scale = (x - y).abs().max().item(), y.abs().max().item()
    assert err <= tol * scale, f"max error {err:.3g} vs scale {scale:.3g}"


@contextlib.contextmanager
def tf32(blas=None):
    precision = torch.backends.cuda.matmul.fp32_precision
    library = torch.backends.cuda.preferred_blas_library()
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        if blas is not None:
            torch.backends.cuda.preferred_blas_library(blas)
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
        if blas is not None:
            torch.backends.cuda.preferred_blas_library(library)
