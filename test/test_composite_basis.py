# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import irbasis3
from irbasis3 import sve
from irbasis3 import kernel
from irbasis3 import composite_basis

import pytest

@pytest.fixture(scope="module")
def basis():
    return sve.compute(kernel.KernelFFlat(42))

def _check_composite_poly(u_comp, u_list, test_points):
    assert u_comp.size == np.sum((u.size for u in u_list))
    assert u_comp.shape == (u_comp.size,)
    np.testing.assert_allclose(u_comp(test_points), np.vstack((u(test_points) for u in u_list)))
    idx = 0
    for isub in range(len(u_list)):
        for ip in range(u_list[isub].size):
            np.testing.assert_allclose(u_comp[idx](test_points), u_list[isub][ip](test_points))
            idx += 1

def test_composite_poly(basis):
    u, s, v = basis
    l = s.size

    u_comp = composite_basis.CompositePiecewiseLegendrePoly([u, u])
    _check_composite_poly(u_comp, [u, u], np.linspace(-1, 1, 10))

    uhat = u.hat("odd")
    uhat_comp = composite_basis.CompositePiecewiseLegendreFT([uhat, uhat])
    _check_composite_poly(uhat_comp, [uhat, uhat], np.array([-3, 1, 5]))

def test_composite_basis():
    lambda_ = 99
    beta = 10
    wmax = lambda_/beta
    K = irbasis3.KernelBFlat(lambda_)
    basis = irbasis3.FiniteTempBasis(K, "F", beta, eps=1e-6)
    basis2 = irbasis3.FiniteTempBasis(K, "F", beta, eps=1e-3)
    basis_comp = composite_basis.CompositeBasis([basis, basis2])
    _check_composite_poly(basis_comp.u, [basis.u, basis2.u], np.linspace(0, beta, 10))
    _check_composite_poly(basis_comp.uhat, [basis.uhat, basis2.uhat], np.array([1,3]))
    _check_composite_poly(basis_comp.v, [basis.v, basis2.v], np.linspace(-wmax, -wmax, 10))