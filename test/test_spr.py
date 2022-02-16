import sparse_ir
from sparse_ir.spr import SparsePoleRepresentation
from sparse_ir.sampling import MatsubaraSampling, TauSampling
import numpy as np
import pytest



@pytest.mark.parametrize("stat", ["F", "B"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-12
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=eps)
    spr = SparsePoleRepresentation(basis)

    np.random.seed(4711)

    num_poles = 10
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = SparsePoleRepresentation(basis, poles).to_IR(coeffs)

    g_spr = spr.from_IR(Gl)

    # Comparison on Matsubara frequencies
    smpl = MatsubaraSampling(basis)
    smpl_for_spr = MatsubaraSampling(spr, smpl.sampling_points)

    giv = smpl_for_spr.evaluate(g_spr)
    giv2 = spr.evaluate_matsubara(g_spr, smpl.sampling_points)

    giv_ref = smpl.evaluate(Gl, axis=0)

    np.testing.assert_allclose(giv, giv_ref, atol=300*eps, rtol=0)
    np.testing.assert_allclose(giv2, giv_ref, atol=300*eps, rtol=0)

    # Comparison on tau
    smpl_tau = TauSampling(basis)
    gtau = smpl_tau.evaluate(Gl)

    smpl_tau_for_spr= TauSampling(spr)
    gtau2 = smpl_tau_for_spr.evaluate(g_spr)

    np.testing.assert_allclose(gtau, gtau2, atol=300*eps, rtol=0)