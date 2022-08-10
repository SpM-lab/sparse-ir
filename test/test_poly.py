# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
from _pytest.mark import param
import numpy as np

import sparse_ir
from sparse_ir import poly
from scipy.integrate import quad

import pytest


def test_shape(sve_logistic):
    u, s, v = sve_logistic[42]
    l = s.size
    assert u.shape == (l,)

    assert u[3].shape == ()
    assert u[2:5].shape == (3,)


def test_slice(sve_logistic):
    sve_result = sve_logistic[42]

    basis = sparse_ir.DimensionlessBasis('F', 42, sve_result=sve_result)
    assert basis[:5].size == 5

    basis = sparse_ir.FiniteTempBasis('F', 4.2, 10, sve_result=sve_result)
    assert basis[:4].size == 4


def test_eval(sve_logistic):
    u, s, v = sve_logistic[42]
    l = s.size

    # evaluate
    np.testing.assert_array_equal(
            u(0.4), [u[i](0.4) for i in range(l)])
    np.testing.assert_array_equal(
            u([0.4, -0.2]),
            [[u[i](x) for x in (0.4, -0.2)] for i in range(l)])


def test_broadcast(sve_logistic):
    u, s, v = sve_logistic[42]

    x = [0.3, 0.5]
    l = [2, 7]
    np.testing.assert_array_equal(
            u.value(l, x), [u[ll](xx) for (ll, xx) in zip(l, x)])


def test_matrix_hat(sve_logistic):
    u, s, v = sve_logistic[42]
    uhat = poly.PiecewiseLegendreFT(u, "odd")

    n = np.array([1, 3, 5, -1, -3, 5])
    result = uhat(n.reshape(3, 2))
    result_iter = uhat(n).reshape(-1, 3, 2)
    assert result.shape == result_iter.shape
    np.testing.assert_array_equal(result, result_iter)


@pytest.mark.parametrize("lambda_, atol", [(42, 1e-13), (1E+4, 1e-13)])
def test_overlap(sve_logistic, lambda_, atol):
    u, s, v = sve_logistic[lambda_]

    # Keep only even number of polynomials
    u, s, v = u[:2*(s.size//2)], s[:2*(s.size//2)], v[:2*(s.size//2)]

    np.testing.assert_allclose(u[0].overlap(u[0]), 1, rtol=0, atol=atol)

    ref = (np.arange(s.size) == 0).astype(float)
    np.testing.assert_allclose(u.overlap(u[0]), ref, rtol=0, atol=atol)


@pytest.mark.parametrize("lambda_, atol", [(42, 1e-13), (1E+4, 1e-13)])
def test_overlap_break_points(sve_logistic, lambda_, atol):
    u, s, v = sve_logistic[lambda_]

    D = 0.5 * v.xmax
    rhow = lambda omega: np.where(abs(omega)<=D, 1, 0)
    rhol = v.overlap(rhow, points=[-D, D])
    rhol_ref = [quad(v[l], -D, D)[0] for l in range(v.size)]

    np.testing.assert_allclose(rhol, rhol_ref, rtol=0, atol=1e-12*np.abs(rhol_ref).max())


def test_eval_unique(sve_logistic):
    u, s, v = sve_logistic[42]
    uhat = poly.PiecewiseLegendreFT(u, "odd")

    # evaluate
    res1 = uhat(np.array([1, 3, 3, 1]))
    idx = np.array([0, 1, 1, 0])
    res2 = uhat(np.array([1,3]))[:,idx]
    np.testing.assert_array_equal(res1, res2)
