# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import sve
from irbasis3 import kernel
from irbasis3 import composite_basis

import pytest

@pytest.fixture(scope="module")
def basis():
    return sve.compute(kernel.KernelFFlat(42))

def test_composite_poly(basis):
    u, s, v = basis
    l = s.size

    u_comp = composite_basis.CompositePiecewiseLegendrePoly([u, u])
    assert u_comp.size == 2*u.size
    assert u_comp.shape == (2*u.size,)
    x = np.linspace(-1, 1, 10)
    np.testing.assert_allclose(u_comp(x), np.vstack(2*(u(x),)))

    for p in range(2):
        for l in range(s.size):
            assert u_comp._internal_pos(l + p*s.size) == (p, l)

    for i in range(s.size):
        np.testing.assert_allclose(u_comp[i](x), u[i](x))
        np.testing.assert_allclose(u_comp[i+s.size](x), u[i](x))

    uhat = u.hat("odd")
    uhat_comp = composite_basis.CompositePiecewiseLegendreFT([uhat, uhat])
    n = np.array([-3, 1, 5])
    np.testing.assert_allclose(uhat_comp(n), np.vstack((uhat(n),uhat(n))))
