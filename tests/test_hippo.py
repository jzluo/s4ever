import pytest
import torch

from s4ever._initialization import nplr, transition


@pytest.mark.parametrize("measure", ["legs", "fout"])
@pytest.mark.parametrize("N", [2, 32])
def test_nplr_is_unitarily_equivalent(measure, N):
    """Compare the singular values and power traces of Lambda - P P^* with those of the HiPPO A.

    The eigenvalues of A and A_hippo are too ill-conditioned to compare.
    """
    Lam, P, _ = (x.to(torch.cdouble) for x in nplr(measure, N))
    Lam, P = torch.cat([Lam, Lam.conj()]), torch.cat([P, P.conj()])
    A = torch.diag(Lam) - P[:, None] * P.conj()[None, :]
    A_hippo = transition(measure, N)[0].to(torch.cdouble)
    sv = torch.linalg.svdvals
    torch.testing.assert_close(sv(A), sv(A_hippo), rtol=1e-5, atol=1e-4)
    for k in (1, 2, 3):
        tr, tr_hippo = (torch.linalg.matrix_power(M, k).trace() for M in (A, A_hippo))
        # Lambda and P from nplr are complex64.
        torch.testing.assert_close(tr, tr_hippo, rtol=3e-4, atol=1e-3)
