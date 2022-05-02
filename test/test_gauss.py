# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import numpy.polynomial.legendre as np_legendre

from sparse_ir import _gauss


def test_collocate():
    r = _gauss.legendre(20)
    cmat = _gauss.legendre_collocation(r)
    emat = np_legendre.legvander(r.x, r.x.size-1)
    np.testing.assert_allclose(emat.dot(cmat), np.eye(20), atol=1e-13, rtol=0)


def _gauss_validate(rule):
    if not (rule.a <= rule.b):
        raise ValueError("a,b must be a valid interval")
    if not (rule.x <= rule.b).all():
        raise ValueError("x must be smaller than b")
    if not (rule.x >= rule.a).all():
        raise ValueError("x must be larger than a")
    if not (rule.x[:-1] < rule.x[1:]).all():
        raise ValueError("x must be well-ordered")
    if rule.x.shape != rule.w.shape:
        raise ValueError("shapes are inconsistent")

    np.testing.assert_allclose(rule.x_forward, rule.x - rule.a)
    np.testing.assert_allclose(rule.x_backward, rule.b - rule.x)

def test_gauss_leg():
    rule = _gauss.legendre(200)
    _gauss_validate(rule)
    x, w = np.polynomial.legendre.leggauss(200)
    np.testing.assert_allclose(rule.x, x)
    np.testing.assert_allclose(rule.w, w)


def test_piecewise():
    edges = [-4, -1, 1, 3]
    rule = _gauss.legendre(20).piecewise(edges)
    _gauss_validate(rule)
