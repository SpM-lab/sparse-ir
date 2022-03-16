# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
from _pytest.mark import param
import numpy as np

import sparse_ir
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


def test_slice(basis):
    sve_result = basis

    basis = sparse_ir.IRBasis('F', 42, sve_result=sve_result)
    assert basis[:5].size == 5

    basis = sparse_ir.FiniteTempBasis('F', 4.2, 10, sve_result=sve_result)
    assert basis[:4].size == 4


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


@pytest.mark.parametrize("lambda_, atol", [(42,1e-14), (1E+5,5e-12)])
def test_overlap(lambda_, atol):
    u, s, v = sve.compute(kernel.LogisticKernel(lambda_))

    # Keep only even number of polynomials
    u, s, v = u[:2*(s.size//2)], s[:2*(s.size//2)], v[:2*(s.size//2)]
    npoly = s.size

    np.testing.assert_allclose(u[0].overlap(u[0]), 1, rtol=0, atol=atol)

    ref = np.ones(s.size)
    ref[0] = 0
    np.testing.assert_allclose(
       np.abs(u.overlap(u[0])-1), ref, rtol=0, atol=atol
    )

    # Axis
    trans_f = lambda x: u[0](x).T
    np.testing.assert_allclose(
       np.abs(u.overlap(trans_f, axis=0)-1), ref, rtol=0, atol=atol
    )

    np.testing.assert_allclose(
        u.overlap(u, axis=-1), np.identity(s.size), rtol=0, atol=atol
    )

    u_tensor = poly.PiecewiseLegendrePoly(
                    u.data.reshape(u.data.shape[:2] + (npoly//2, 2)), u.knots)
    res = u_tensor.overlap(u_tensor, axis=-1)
    assert res.shape == (npoly//2, 2, npoly//2, 2)
    np.testing.assert_allclose(
        res.reshape(npoly, npoly), np.identity(s.size), rtol=0, atol=atol
    )


param_axis = []
shapes_axis = [(1,), (1,1), (1,2), (1,2,3)]
for shape in shapes_axis:
    for axis in range(len(shape)):
        param_axis.append((shape, axis))


@pytest.mark.parametrize("shape, axis", param_axis)
def test_overlap_axis(basis, shape, axis):
    u, s, v = basis
    def f(x):
        res = np.zeros((x.size, np.prod(shape)))
        for i in range(res.shape[1]):
            res[:,i] = i * x
        res = res.reshape((x.size,) + shape)
        return np.moveaxis(res, 0, axis)

    overlap = u.overlap(f, axis=axis)

    overlap_ref = np.empty((u.size, np.prod(shape)), dtype=overlap.dtype)
    for i in range(overlap_ref.shape[1]):
        overlap_ref[:,i] = u.overlap(lambda x: i*x)
    overlap_ref = overlap_ref.reshape((u.size,) + shape)

    np.testing.assert_allclose(overlap, overlap_ref, rtol=0,
                               atol=1e-10*np.abs(overlap_ref).max())
