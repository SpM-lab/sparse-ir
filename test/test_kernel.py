# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from sparse_ir import kernel
from sparse_ir import gauss

KERNELS = [
    kernel.KernelFFlat(9),
    kernel.KernelBFlat(8),
    kernel.KernelFFlat(120_000),
    kernel.KernelBFlat(127_500),
    kernel.KernelFFlat(40_000).get_symmetrized(-1),
    kernel.KernelBFlat(35_000).get_symmetrized(-1),
    ]


@pytest.mark.parametrize("K", KERNELS)
def test_accuracy(K):
    dtype = np.float32
    dtype_x = np.float64

    rule = gauss.legendre(10, dtype)
    hints = K.hints(1e-16)
    gauss_x = rule.piecewise(hints.segments_x)
    gauss_y = rule.piecewise(hints.segments_y)
    eps = np.finfo(dtype).eps
    tiny = np.finfo(dtype).tiny / eps

    result = kernel.matrix_from_gauss(K, gauss_x, gauss_y)
    result_x = kernel.matrix_from_gauss(
                    K, gauss_x.astype(dtype_x), gauss_y.astype(dtype_x))
    magn = np.abs(result_x).max()
    np.testing.assert_allclose(result, result_x, atol = 2 * magn * eps, rtol=0,
                               err_msg="absolute precision tool poor")

    with np.errstate(invalid='ignore'):
        reldiff = np.where(np.abs(result_x) < tiny, 1, result / result_x)
    np.testing.assert_allclose(reldiff, 1, atol=100 * eps, rtol=0,
                               err_msg="relative precision too poor")
