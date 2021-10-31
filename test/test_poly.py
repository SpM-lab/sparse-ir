# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import sve
from irbasis3 import kernel
from irbasis3 import poly

import pytest


@pytest.fixture(scope="module")
def basis():
    return sve.compute(kernel.KernelFFlat(42))


def test_shape(basis):
    u, s, v = basis
    l = s.size
    assert u.shape == (l,)

    assert u[3].shape == ()
    assert u[2:5].shape == (3,)


def test_eval(basis):
    u, s, v = basis
    l = s.size

    # evaluate
    np.testing.assert_array_almost_equal_nulp(
            u(0.4), [u[i](0.4) for i in range(l)])
    np.testing.assert_array_almost_equal_nulp(
            u([0.4, -0.2]),
            [[u[i](x) for x in (0.4, -0.2)] for i in range(l)])


def test_broadcast(basis):
    u, s, v = basis

    x = [0.3, 0.5]
    l = [2, 7]
    np.testing.assert_array_almost_equal_nulp(
            u.value(l, x), [u[ll](xx) for (ll, xx) in zip(l, x)])


def test_overlap(basis):
    u, s, v = basis

    ref = np.ones(s.size)
    ref[0] = 0
    np.testing.assert_allclose(
       np.abs(u.overlap(u[0])-1), ref, rtol=0, atol=1e-14
    )

    # Axis
    trans_f = lambda x: u[0](x).T
    np.testing.assert_allclose(
       np.abs(u.overlap(trans_f, axis=0)-1), ref, rtol=0, atol=1e-14
    )

    # Matrix-valued functions
    np.testing.assert_allclose(
        u.overlap(u), np.identity(s.size), rtol=0, atol=1e-14
    )