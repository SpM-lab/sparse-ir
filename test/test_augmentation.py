# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import augmentation
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
    beta = 2.0
    Nl = 10
    basis = augmentation.LegendreBasis(stat, beta, Nl)

    tau = np.array([0, 0.4*beta, beta])
    uval = basis.u(tau)
    for l in range(Nl):
        np.testing.assert_allclose(
            uval[l,:],
            (np.sqrt(2*l+1)/beta) * eval_legendre(l, 2*tau/beta-1)
            )
    
    zeta = {"F": 1, "B": 0}[stat]
    w = 2*np.arange(-10, 10) + zeta

    uhat_val = basis.uhat(w)
    uhat_val_ref = compute_Tnl(w, Nl)
    np.testing.assert_allclose(uhat_val.T, uhat_val_ref)

