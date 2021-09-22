# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import sampling


def test_decomp():
    rng = np.random.RandomState(4711)
    A = rng.random(size=(49, 39))

    Ad = sampling.DecomposedMatrix.from_matrix(A)
    np.testing.assert_allclose(A, np.asarray(Ad), atol=1e-13, rtol=0)

    x = rng.random(size=39)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-13, rtol=0)

    x = rng.random(size=(39, 3))
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-13, rtol=0)

    y = rng.random(size=49)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-13, rtol=0)

    y = rng.random(size=(49, 2))
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-13, rtol=0)
