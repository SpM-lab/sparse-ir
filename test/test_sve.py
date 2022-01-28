# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import sparse_ir

import pytest

BASES = [
    ('F', 73),
    ('B', 37)
    ]


@pytest.fixture(scope="module")
def bases():
    return {(stat, lambda_): sparse_ir.IRBasis(stat, lambda_)
            for (stat, lambda_) in BASES}


def _check_smooth(u, s, uscale, fudge_factor):
    eps = np.finfo(s.dtype).eps
    x = u.knots[1:-1]

    jump = np.abs(u(x + eps) - u(x - eps))
    compare = np.abs(u(x + 3 * eps) - u(x + eps))
    compare = np.maximum(compare, uscale * eps)

    # loss of precision
    compare *= fudge_factor * (s[0] / s)[:, None]
    try:
        np.testing.assert_array_less(jump, compare)
    except:
        print((jump > compare).nonzero())
        raise


@pytest.mark.parametrize("stat,lambda_", BASES)
def test_smooth(bases, stat, lambda_):
    basis = bases[stat, lambda_]
    _check_smooth(basis.u, basis.s, 2*basis.u(1).max(), 24)
    _check_smooth(basis.v, basis.s, 50, 20)


@pytest.mark.parametrize("stat,lambda_", BASES)
def test_num_roots_u(bases, stat, lambda_):
    basis = bases[stat, lambda_]
    for i in range(basis.u.size):
        ui_roots = basis.u[i].roots()
        assert ui_roots.size == i


@pytest.mark.parametrize("stat,lambda_", BASES)
def test_num_roots_uhat(bases, stat, lambda_):
    basis = bases[stat, lambda_]
    for i in [0, 1, 7, 10]:
        x0 = basis.uhat[i].extrema()
        assert i + 1 <= x0.size <= i + 2
