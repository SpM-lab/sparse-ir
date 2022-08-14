# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import sparse_ir
from sparse_ir import poly
from sparse_ir import augment

import pytest


def test_augmented_bosonic_basis():
    """Augmented bosonic basis"""
    wmax = 2
    beta = 1000
    basis = sparse_ir.FiniteTempBasis("B", beta, wmax, eps=1e-6)
    basis_comp = augment.AugmentedBasis(basis, augment.TauConst, augment.TauLinear)

    # G(tau) = c - e^{-tau*pole}/(1 - e^{-beta*pole})
    pole = 1.0
    const = 1e-2
    tau_smpl = sparse_ir.TauSampling(basis_comp)
    assert tau_smpl.sampling_points.size == basis_comp.size
    gtau = const + basis.u(tau_smpl.tau).T @ (-basis.s * basis.v(pole))
    magn = np.abs(gtau).max()

    # This illustrates that "naive" fitting is a problem if the fitting matrix
    # is not well-conditioned.
    gl_fit_bad = np.linalg.pinv(tau_smpl.matrix) @ gtau
    gtau_reconst_bad = tau_smpl.evaluate(gl_fit_bad)
    assert not np.allclose(gtau_reconst_bad, gtau, atol=1e-13 * magn, rtol=0)
    np.testing.assert_allclose(gtau_reconst_bad, gtau,
                               atol=5e-16 * tau_smpl.cond * magn, rtol=0)

    # Now do the fit properly
    gl_fit = tau_smpl.fit(gtau)
    gtau_reconst = tau_smpl.evaluate(gl_fit)
    np.testing.assert_allclose(gtau_reconst, gtau, atol=1e-14 * magn, rtol=0)


@pytest.mark.parametrize("stat", ["F", "B"])
def test_vertex_basis(stat):
    """Vertex basis"""
    wmax = 2
    beta = 1000
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=1e-6)
    basis_comp = augment.AugmentedBasis(basis, augment.MatsubaraConst)
    assert basis_comp.uhat is not None

    # G(iv) = c + 1/(iv-pole)
    pole = 1.0
    c = 1.0
    matsu_smpl = sparse_ir.MatsubaraSampling(basis_comp)
    giv = c  + 1/(1J*matsu_smpl.sampling_points * np.pi/beta - pole)
    gl = matsu_smpl.fit(giv)

    giv_reconst = matsu_smpl.evaluate(gl)

    np.testing.assert_allclose(giv, giv_reconst,
                               atol=np.abs(giv).max() * 1e-7, rtol=0)
