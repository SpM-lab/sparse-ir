# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import sparse_ir
from sparse_ir import augment
from scipy.special import eval_legendre, spherical_jn

import pytest

def compute_Tnl(vsample, n_legendre):
    """
    Compute transformation matrix from Legendre to fermionic/bosonic Matsubara frequency
    Implement Eq. (4.5) in the Boehnke's  Ph.D thesis
    """
    Tnl = np.zeros((vsample.size, n_legendre), dtype=np.complex128)
    for idx_n, v in enumerate(vsample):
        abs_v = abs(v)
        sph_jn = np.array(
            [spherical_jn(l, 0.5*abs_v*np.pi) for l in range(n_legendre)])
        for il in range(n_legendre):
            Tnl[idx_n, il] = (1J**(abs_v+il)) * np.sqrt(2*il + 1.0) * sph_jn[il]
        if v < 0:
            Tnl[idx_n, :] = Tnl[idx_n, :].conj()
    return Tnl

@pytest.mark.parametrize("stat", ["F", "B"])
def test_legendre_basis(stat):
    beta = 10.0
    Nl = 100
    cl = np.sqrt(2*np.arange(Nl) + 1)
    basis = augment.LegendreBasis(stat, beta, Nl, cl=cl)

    tau = np.array([0, 0.1*beta, 0.4*beta, beta])
    uval = basis.u(tau)
    l = 0
    for l in range(Nl):
        np.testing.assert_allclose(
            uval[l,:],
            cl[l] * (np.sqrt(2*l+1)/beta) * eval_legendre(l, 2*tau/beta-1)
            )

    zeta = {"F":  1, "B": 0}[stat]
    sign = {"F": -1, "B": 1}[stat]
    w = 2*np.arange(-10, 10) + zeta

    uhat_val = basis.uhat(w)
    uhat_val_ref = compute_Tnl(w, Nl)
    np.testing.assert_allclose(uhat_val.T, uhat_val_ref * cl[None,:])

    # G(iv) = 1/(iv-pole)
    # G(tau) = -e^{-tau*pole}/(1 + e^{-beta*pole}) [F]
    #        = -e^{-tau*pole}/(1 - e^{-beta*pole}) [B]
    pole = 1.0
    tau_smpl = sparse_ir.TauSampling(basis)
    gtau = -np.exp(-tau_smpl.sampling_points * pole)/(1 - sign * np.exp(-beta * pole))
    gl_from_tau = tau_smpl.fit(gtau)

    matsu_smpl = sparse_ir.MatsubaraSampling(basis)
    giv = 1/(1J*matsu_smpl.sampling_points*np.pi/beta - pole)
    gl_from_matsu = matsu_smpl.fit(giv)

    np.testing.assert_allclose(gl_from_tau, gl_from_matsu,
                               atol=1e-10*np.abs(gl_from_matsu).max(), rtol=0)


@pytest.mark.parametrize("stat", ["F", "B"])
def test_const_basis(stat):
    beta = 100
    value = 2.001
    basis = augment.MatsubaraConstBasis(stat, beta, value)

    zeta = {"F":  1, "B": 0}[stat]
    v = 2*np.arange(10)+zeta
    uhat_ = basis.uhat(v)
    assert uhat_.shape == (v.size, 1)
    np.testing.assert_array_equal(uhat_.ravel(), np.ones_like(v) * value)
