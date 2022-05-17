# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import sparse_ir
from sparse_ir.spr import SparsePoleRepresentation
from sparse_ir.sampling import MatsubaraSampling, TauSampling
import numpy as np
import pytest


@pytest.mark.parametrize("stat", ["F", "B"])
def test_compression(sve_logistic, stat):
    beta = 10_000
    wmax = 1
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax,
                                      sve_result=sve_logistic[beta*wmax])
    spr = SparsePoleRepresentation(basis)

    np.random.seed(4711)

    num_poles = 10
    poles = wmax * (2*np.random.rand(num_poles) - 1)
    coeffs = 2*np.random.rand(num_poles) - 1
    assert np.abs(poles).max() <= wmax

    Gl = SparsePoleRepresentation(basis, poles).to_IR(coeffs)

    g_spr = spr.from_IR(Gl)
    eps = basis.accuracy * np.linalg.norm(g_spr)
    print(f"eps = {eps:.2g}")

    # Comparison on Matsubara frequencies
    smpl = MatsubaraSampling(basis)
    smpl_for_spr = MatsubaraSampling(spr, smpl.sampling_points)

    giv = smpl_for_spr.evaluate(g_spr)
    giv_ref = smpl.evaluate(Gl, axis=0)
    np.testing.assert_allclose(giv, giv_ref, atol=1e4*eps, rtol=0)

    # Comparison on tau
    smpl_tau = TauSampling(basis)
    gtau = smpl_tau.evaluate(Gl)

    smpl_tau_for_spr= TauSampling(spr)
    gtau2 = smpl_tau_for_spr.evaluate(g_spr)

    np.testing.assert_allclose(gtau, gtau2, atol=1e4*eps, rtol=0)


def test_boson(sve_logistic):
    beta = 2
    wmax = 21
    eps = 1e-7
    basis_b = sparse_ir.FiniteTempBasis("B", beta, wmax, eps=eps,
                                        sve_result=sve_logistic[beta * wmax])

    coeff = np.array([1.1, 2.0])
    omega_p = np.array([2.2, -1.0])

    rhol_pole = np.einsum('lp,p->l', basis_b.v(omega_p), coeff/np.tanh(0.5*beta*omega_p))
    gl_pole = - basis_b.s * rhol_pole

    sp = SparsePoleRepresentation(basis_b, omega_p)
    gl_pole2 = sp.to_IR(coeff)

    np.testing.assert_allclose(gl_pole, gl_pole2, atol=300*eps, rtol=0)
