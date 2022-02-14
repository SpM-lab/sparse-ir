import sparse_ir
from sparse_ir.dlr_like import DLRLike
from sparse_ir.sampling import MatsubaraSampling
import numpy as np
import pytest


def _to_IR(basis, poles, coeffs):
    weight_func = basis.kernel.weight_func(basis.statistics)
    rhol = np.einsum(
        'lp,p,p->l',
        basis.v(poles),
        coeffs,
        weight_func(basis.beta * poles/basis.wmax)
    )
    return -basis.s * rhol


@pytest.mark.parametrize("stat", ["F", "B"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-12
    basis = sparse_ir.FiniteTempBasis(
        stat, beta, wmax, eps=eps, kernel=sparse_ir.KernelFFlat(beta*wmax))
    dlr = DLRLike(basis)

    np.random.seed(4711)

    num_poles = 1
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = _to_IR(basis, poles, coeffs)

    g_dlr = dlr.from_IR(Gl)

    smpl = MatsubaraSampling(basis)
    giv = dlr.evaluate_matsubara(g_dlr, smpl.sampling_points)

    giv_ref = smpl.evaluate(Gl, axis=0)

    np.testing.assert_allclose(giv, giv_ref, atol=0, rtol=1e+4*eps)
