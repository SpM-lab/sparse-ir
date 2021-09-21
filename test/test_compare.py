# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from irbasis3 import adapter

try:
    import irbasis
except ImportError:
    pytest.skip("no irbasis library for comparison", allow_module_level=True)
    raise

COMPARE_PARAMS = [
    ('F', 10),
    ('F', 10_000),
    ('B', 10),
    ('B', 10_000),
    ]

# Return of fixture functions are passed as parameters to test functions
# if the parameter has the same name as the fixture function.  scope="module"
# ensures that fixtures are cached.

@pytest.fixture(scope="module")
def adapters():
    return {params: adapter.load(*params) for params in COMPARE_PARAMS}


@pytest.fixture(scope="module")
def references():
    return {params: irbasis.load(*params) for params in COMPARE_PARAMS}


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_svals(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared = slice(min(adapt.dim(), ref.dim()))
    eps = np.finfo(float).eps

    assert adapt.statistics == ref.statistics == stat
    assert adapt.Lambda == ref.Lambda == lambda_
    assert adapt.dim() > 10

    np.testing.assert_allclose(adapt.sl(shared), ref.sl(shared),
                               atol=10 * ref.sl(0) * eps, rtol=0)

@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_vly(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())

    y = [-1., -.5, -.1, -.01, -.001, -.0001, 0, .0001, .001, .01, .1, .5, 1.]
    for li in [0, 1, 2, shared_dim//2, shared_dim//2 + 1, shared_dim-1]:
        uly_adapt = adapt.vly(li, y)
        uly_ref = ref.vly(li, y)
        tol = 1e-10 * np.abs(uly_ref).max() * ref.sl(0) / ref.sl(li)
        np.testing.assert_allclose(uly_adapt, uly_ref, atol=tol, rtol=0)


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_ulx(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())

    x = [-1., -.9999, -.999, -.99, -.9, -.5, 0., .5, .9, .99, .999, .9999, 1.]
    for li in [0, 1, 2, shared_dim//2, shared_dim//2 + 1, shared_dim-1]:
        uly_adapt = adapt.ulx(li, x)
        uly_ref = ref.ulx(li, x)
        tol = 1e-12 * np.abs(uly_ref).max() * ref.sl(0) / ref.sl(li)
        np.testing.assert_allclose(uly_adapt, uly_ref, atol=tol, rtol=0)


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_ulx(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())

    n = [-20, -2, -1, 0, 1, 2, 20, 100, 300, 1000]
    for li in [0, 1, 2, shared_dim//2, shared_dim//2 + 1, shared_dim-1]:
        ulxhat_adapt = adapt.compute_unl(n, li)
        ulxhat_ref = ref.compute_unl(n, li).ravel()
        tol = 1e-13 * np.abs(ulxhat_ref).max() * ref.sl(0) / ref.sl(li)
        np.testing.assert_allclose(ulxhat_adapt, ulxhat_ref, atol=tol, rtol=0)
