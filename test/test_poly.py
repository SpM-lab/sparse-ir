# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
from _pytest.mark import param
import numpy as np
from sparse_ir import sve
from sparse_ir import kernel
from sparse_ir import poly

import pytest


@pytest.fixture(scope="module")
def basis():
    return sve.compute(kernel.LogisticKernel(42))


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


def test_matrix_hat(basis):
    u, s, v = basis
    uhat = u.hat('odd')

    n = np.array([1, 3, 5, -1, -3, 5])
    result = uhat(n.reshape(3, 2))
    result_iter = uhat(n).reshape(-1, 3, 2)
    assert result.shape == result_iter.shape
    np.testing.assert_array_almost_equal_nulp(result, result_iter)


def test_overlap(basis):
    u, s, v = basis
    atol = max(s[-1] / s[0], 2.2e-16 * s.size)

    # Keep only even number of polynomials
    np.testing.assert_allclose(u[0].overlap(u[0]), 1, rtol=0, atol=atol)

    ref = (np.arange(s.size) == 0).astype(int)
    np.testing.assert_allclose(u.overlap(u[0]), ref, rtol=0, atol=atol)
