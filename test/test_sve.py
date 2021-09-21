# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from irbasis3 import sve
from irbasis3 import kernel
from irbasis3 import poly

import pytest

BASES = [
    ('F', 73),
    ('B', 37)
    ]


@pytest.fixture(scope="module")
def bases():
    def _make_basis(stat, lambda_):
        K = {'F': kernel.KernelFFlat, 'B': kernel.KernelBFlat}[stat](lambda_)
        return sve.compute(K)

    return {p: _make_basis(*p) for p in BASES}


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
    u, s, v = bases[stat, lambda_]
    _check_smooth(u, s, 2*u(1).max(), 12)
    _check_smooth(v, s, 50, 10)


@pytest.mark.parametrize("stat,lambda_", BASES)
def test_num_roots_u(bases, stat, lambda_):
    u, s, v = bases[stat, lambda_]
    for i in range(u.size):
        ui_roots = u[i].roots()
        assert ui_roots.size == i


@pytest.mark.parametrize("stat,lambda_", BASES)
def test_num_roots_uhat(bases, stat, lambda_):
    u, s, v = bases[stat, lambda_]
    zeta = {'B': 0, 'F': 1}[stat]
    for i in [0, 1, 7, 10]:
        part = ['real', 'imag'][(i + zeta) % 2]
        x0 = poly.find_hat_extrema(u[i], part, zeta)
        assert i + 1 <= x0.size <= i + 2
