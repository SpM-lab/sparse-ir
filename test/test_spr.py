import sparse_ir
from sparse_ir.spr import SparsePoleRepresentation
from sparse_ir.sampling import MatsubaraSampling
import numpy as np
import pytest


def _to_IR(basis, poles, coeffs):
    rhol = np.einsum(
        'lp,p,p->l',
        basis.v(poles),
        coeffs,
        basis.kernel.weight_func(basis.beta * poles/basis.wmax)
    )
    return -basis.s * rhol


@pytest.mark.parametrize("stat", ["F", "B"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-12
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=eps)
    spr = SparsePoleRepresentation(basis)

    np.random.seed(4711)

    num_poles = 1
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = spr.to_IR(coeffs)

    g_spr = spr.from_IR(Gl)

    smpl = MatsubaraSampling(basis)
    giv = spr.evaluate_matsubara(g_spr, smpl.sampling_points)

    giv_ref = smpl.evaluate(Gl, axis=0)

    np.testing.assert_allclose(giv, giv_ref, atol=300*eps, rtol=0)
