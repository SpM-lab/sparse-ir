# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from sparse_ir import kernel
from sparse_ir import gauss

KERNELS = [
    kernel.LogisticKernel(9),
    kernel.RegularizedBoseKernel(8),
    kernel.LogisticKernel(120_000),
    kernel.RegularizedBoseKernel(127_500),
    kernel.LogisticKernel(40_000).get_symmetrized(-1),
    kernel.RegularizedBoseKernel(35_000).get_symmetrized(-1),
    ]


@pytest.mark.parametrize("K", KERNELS)
def test_accuracy(K):
    dtype = np.float32
    dtype_x = np.float64

    rule = gauss.legendre(10, dtype)
    hints = K.sve_hints(2.2e-16)
    gauss_x = rule.piecewise(hints.segments_x)
    gauss_y = rule.piecewise(hints.segments_y)
    eps = np.finfo(dtype).eps
    tiny = np.finfo(dtype).tiny / eps

    result = kernel.matrix_from_gauss(K, gauss_x, gauss_y)
    result_x = kernel.matrix_from_gauss(
                    K, gauss_x.astype(dtype_x), gauss_y.astype(dtype_x))
    magn = np.abs(result_x).max()
    np.testing.assert_allclose(result, result_x, atol = 2 * magn * eps, rtol=0,
                               err_msg="absolute precision too poor")

    with np.errstate(invalid='ignore'):
        reldiff = np.where(np.abs(result_x) < tiny, 1, result / result_x)
    np.testing.assert_allclose(reldiff, 1, atol=100 * eps, rtol=0,
                               err_msg="relative precision too poor")

@pytest.mark.parametrize("lambda_", [10, 42, 10_000])
def test_singularity(lambda_):
    x = np.random.rand(1000) * 2 - 1
    K = kernel.RegularizedBoseKernel(lambda_)
    np.testing.assert_allclose(K(x, [0.0]), 1 / lambda_)
