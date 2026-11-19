"""Initializations of the state matrices.

S4 uses the normal-plus-low-rank (NPLR) form of the HiPPO matrices, A = V (Lambda - P P^*) V^*.
The tensors that `s4_init` and `s4d_init` return contain one mode of each conjugate pair.
"""

import math
from typing import Literal, get_args

import torch
from torch import Tensor

type S4Init = Literal["legs", "fout", "hippo"]
type S4DInit = Literal["diag-lin", "diag-inv", "diag-real", "legs", "diag"]
S4_INITS = get_args(S4Init.__value__)
S4D_INITS = get_args(S4DInit.__value__)
COMBINATIONS = {"hippo": ("legs", "fout"), "diag": ("diag-inv", "diag-lin")}


def transition(measure: str, N: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return the HiPPO matrices (A, B, P) of a measure in float64.

    A + P P^T is skew-symmetric up to a multiple of the identity.
    """
    q = torch.arange(N, dtype=torch.float64)
    if measure == "legs":
        r = (2 * q + 1).sqrt()
        A = -torch.tril(r[:, None] * r[None, :], diagonal=-1) - torch.diag(q + 1)
        return A, r, (0.5 + q).sqrt()
    if measure == "fout":
        freqs = torch.arange(N // 2, dtype=torch.float64)
        d = torch.stack([torch.zeros_like(freqs), freqs], dim=-1).reshape(-1)[1:]
        A = math.pi * (torch.diag(d, -1) - torch.diag(d, 1))
        B = torch.zeros(N, dtype=torch.float64)
        B[0::2] = 2**0.5
        B[0] = 1
        return A - B[:, None] * B[None, :], B, B
    raise ValueError(f"unknown HiPPO measure {measure!r}; choose from ('legs', 'fout')")


def nplr(measure: str, N: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return the NPLR form (Lambda, P, B) of a HiPPO matrix, each as an (N // 2,) tensor."""
    A, B, P = transition(measure, N)
    AP = A + P[:, None] * P[None, :]
    W_re = AP.diagonal().mean()
    # Because AP is a skew-symmetric matrix plus W_re * I, -1j * AP is Hermitian except for its
    # imaginary diagonal, which eigh treats as zero.
    W_im, V = torch.linalg.eigh(AP * -1j)
    idx = W_im.argsort()
    W = (W_re + 1j * W_im)[idx][: N // 2]  # keep eigenvalues with negative imaginary part
    V = V[:, idx][:, : N // 2]
    if W[-1].abs() < 1e-4:
        # A zero eigenvalue has no conjugate partner among eigh's eigenvectors, so its eigenvector
        # is set to (e_0 + i e_1) / sqrt(2), whose conjugate is the other eigenvector for 0.
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j
    Vh = V.conj().T
    B = Vh @ B.to(V)
    P = Vh @ P.to(V)
    B = torch.complex(B.real, B.imag.clamp(-2, 2))  # as the original S4 clips it
    return W.to(torch.complex64), P.to(torch.complex64), B.to(torch.complex64)


def diag(init: str, N: int) -> tuple[Tensor, Tensor]:
    """Return (A, B) for a diagonal init, each as an (N // 2,) tensor."""
    n = torch.arange(N // 2, dtype=torch.float64)
    real = torch.full_like(n, 0.5)
    if init == "diag-lin":
        imag = math.pi * n
    elif init == "diag-inv":
        imag = N / math.pi * (N / (1 + 2 * n) - 1)
    else:  # diag-real
        real, imag = 1 + n, torch.zeros_like(n)
    A = torch.complex(-real, -imag).to(torch.complex64)
    return A, torch.ones_like(A)


def expand(init: str, n_ssm: int) -> tuple[str, ...]:
    """Return the inits that `init` combines, after checking that they divide n_ssm evenly."""
    inits = COMBINATIONS.get(init, (init,))
    if n_ssm % len(inits):
        raise ValueError(
            f"n_ssm={n_ssm} must be a multiple of {len(inits)} for init={init!r}; increase n_ssm "
            "or pick a single initialization"
        )
    return inits


def s4_init(init: str, d_state: int, n_ssm: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return (Lambda, P, B) for an S4 init, each as an (n_ssm, d_state // 2) tensor."""
    inits = expand(init, n_ssm)
    if unknown := set(inits) - set(S4_INITS):
        raise ValueError(f"unknown S4 init {unknown.pop()!r}; choose from {S4_INITS}")
    params = [torch.stack(nplr(i, d_state)) for i in inits]
    A, P, B = torch.cat([p[:, None].expand(-1, n_ssm // len(inits), -1) for p in params], dim=1)
    return A, P, B


def s4d_init(init: str, d_state: int, n_ssm: int) -> tuple[Tensor, Tensor]:
    """Return (A, B) for an S4D init, each as an (n_ssm, d_state // 2) tensor."""
    inits = expand(init, n_ssm)
    if unknown := set(inits) - set(S4D_INITS):
        raise ValueError(f"unknown S4D init {unknown.pop()!r}; choose from {S4D_INITS}")
    params = []
    for i in inits:
        if i == "legs":
            A, _, B = nplr(i, d_state)
        else:
            A, B = diag(i, d_state)
        params.append(torch.stack([A, B]))
    A, B = torch.cat([p[:, None].expand(-1, n_ssm // len(inits), -1) for p in params], dim=1)
    return A, B
