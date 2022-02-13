import sparse_ir
from sparse_ir.dicrete_lehmann import to_discrete_lehmann_rep, eval_matsubara_from_discrete_lehmann_rep
from sparse_ir.sampling import MatsubaraSampling
import numpy as np
import pytest

scale_func = {
        "F": lambda omega: np.ones_like(omega),
        "B": lambda omega: 1/np.tanh(omega)
    }

def _to_IR(basis, poles, coeffs):
    rhol = np.einsum(
        'lp,p,p->l',
        basis.v(poles),
        coeffs,
        scale_func[basis.statistics](poles)
    )
    return -basis.s * rhol

@pytest.mark.parametrize("stat", ["F"])
def test_compression(stat):
    beta = 1e+4
    wmax = 1
    eps = 1e-12
    basis = sparse_ir.FiniteTempBasis(
        stat, beta, wmax, eps=eps, kernel=sparse_ir.KernelFFlat(beta*wmax))

    np.random.seed(4711)

    num_poles = 100
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = _to_IR(basis, poles, coeffs)

    g_dlr = to_discrete_lehmann_rep(basis, Gl)

    vsample = basis.default_matsubara_sampling_points()
    giv = eval_matsubara_from_discrete_lehmann_rep(beta, vsample, g_dlr)

    smpl_matsu = MatsubaraSampling(basis, vsample)
    giv_ref = smpl_matsu.evaluate(Gl, axis=0)

    Gl_reconst = _to_IR(basis, *g_dlr)
    #for x_, y_ in zip(Gl, Gl_reconst):
        #print(x_.real, y_.real)

    #print()
    #for x_, y_ in zip(giv, giv_ref):
        #print(x_.imag, y_.imag)
    np.testing.assert_allclose(giv, giv_ref, atol=0, rtol=1e+4*eps)