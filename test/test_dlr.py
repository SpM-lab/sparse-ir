import sparse_ir
from sparse_ir.dicrete_lehmann import to_discrete_lehmann_rep, eval_matsubara_from_discrete_lehmann_rep
from sparse_ir.sampling import MatsubaraSampling
import numpy as np
import pytest

@pytest.mark.parametrize("stat", ["F"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-15
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=eps)

    np.random.seed(4711)

    num_poles = 100
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    rhol = basis.v(poles) @ coeffs
    Gl = -basis.s * rhol

    g_dlr = to_discrete_lehmann_rep(basis, Gl)

    vsample = basis.default_matsubara_sampling_points()
    giv = eval_matsubara_from_discrete_lehmann_rep(beta, vsample, g_dlr)

    smpl_matsu = MatsubaraSampling(basis, vsample)
    giv_ref = smpl_matsu.evaluate(Gl, axis=0)

    np.testing.assert_allclose(giv, giv_ref)
    print()
    for x_, y_ in zip(giv, giv_ref):
        print(x_.imag, y_.imag)