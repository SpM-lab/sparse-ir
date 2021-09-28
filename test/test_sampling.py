# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import sampling


def test_decomp():
    rng = np.random.RandomState(4711)
    A = rng.randn(49, 39)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]
    np.testing.assert_allclose(A, np.asarray(Ad), atol=1e-15 * norm_A, rtol=0)

    x = rng.randn(39)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    x = rng.randn(39, 3)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49, 2)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)


def test_axis():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(2, 21, 4, 7)
    ref = np.tensordot(A, x, (-1,1)).transpose((1,0,2,3))
    np.testing.assert_allclose(
            Ad.matmul(x, axis=1), ref,
            atol=1e-13 * norm_A, rtol=0)

def test_axis0():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(21, 2)

    np.testing.assert_allclose(
            Ad.matmul(x, axis=0), A@x,
            atol=1e-13 * norm_A, rtol=0)

    np.testing.assert_allclose(
            Ad.matmul(x), A@x,
            atol=1e-13 * norm_A, rtol=0)
