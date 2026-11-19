"""Float64 references for the ops.

The references compute the recurrences explicitly in complex arithmetic. Their states have 2 N
modes, which are the N modes and their conjugates.
"""

import torch


def cz(x):
    return torch.view_as_complex(x.double().contiguous())


def full(z):
    return torch.cat([z, z.conj()], dim=-1)


def s4d_matrices(dt, A, B, C, disc):
    dtA = dt.double()[:, None] * cz(A)
    if disc == "zoh":
        dA = dtA.exp()
        dB = (dA - 1) / cz(A) * cz(B)
    else:
        dA = (1 + dtA / 2) / (1 - dtA / 2)
        dB = dt.double()[:, None] * cz(B) / (1 - dtA / 2)
    eye = torch.eye(dA.shape[-1] * 2, dtype=torch.cdouble)
    return eye * full(dA)[..., None, :], full(dB), full(cz(C))


def s4_matrices(dt, Lam, P, B, C, l_kernel):
    A = (
        torch.diag_embed(full(cz(Lam)))
        - full(cz(P))[..., :, None] * full(cz(P)).conj()[..., None, :]
    )
    eye = torch.eye(A.shape[-1], dtype=torch.cdouble)
    d = dt.double()[:, None, None]
    dA = torch.linalg.solve(eye - d / 2 * A, eye + d / 2 * A)
    dB = torch.linalg.solve(eye - d / 2 * A, d * full(cz(B))[..., None])[..., 0]
    M = eye - torch.linalg.matrix_power(dA, l_kernel)
    Ct = full(cz(C))
    Ctrue = torch.linalg.solve(M.mT, Ct.permute(1, 2, 0)).permute(2, 0, 1)
    return dA, dB, Ctrue


def conv_kernel(dA, dB, C, L):
    ks, v = [], dB
    for _ in range(L):
        ks.append(torch.einsum("chn,hn->ch", C, v))
        v = torch.einsum("hmn,hn->hm", dA, v)
    return torch.stack(ks, -1).real


def run(dA, dB, C, D, u, x0=None, reverse=False):
    """Return y (B, L, C, H) and the final state (B, H, N, 2) of the recurrence x' = dA x + dB u.

    The output is y = C x + D u. The final state has only the first N of the 2 N modes.
    """
    u = u.double().flip(1) if reverse else u.double()
    N = dA.shape[-1] // 2
    x = (
        torch.zeros(u.shape[0], u.shape[2], 2 * N, dtype=torch.cdouble)
        if x0 is None
        else full(cz(x0))
    )
    ys = []
    for t in range(u.shape[1]):
        x = torch.einsum("hmn,bhn->bhm", dA, x) + dB * u[:, t, :, None]
        ys.append(torch.einsum("chn,bhn->bch", C, x).real)
    y = torch.stack(ys, 1)
    if D is not None:
        y = y + u[:, :, None] * D.double()
    return (y.flip(1) if reverse else y), torch.view_as_real(x[..., :N])
