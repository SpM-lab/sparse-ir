# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from sparse_ir import adapter

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
def adapters(sve_logistic, sve_reg_bose):
    table = {'F': sve_logistic, 'B': sve_reg_bose}
    return {(stat, lambda_): adapter.Basis(stat, lambda_, table[stat][lambda_])
            for (stat, lambda_) in COMPARE_PARAMS}


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
def test_ulx2(stat, lambda_, adapters, references):
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


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_matasubara_sampling(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())
    zeta = {'F': 1, 'B': 0}[stat]

    l = shared_dim - 1

    sp_adapt = adapt.sampling_points_matsubara(whichl=shared_dim-1)
    u_adapt = adapt.compute_unl(sp_adapt, shared_dim-1)
    u_adapt_real = u_adapt.real if l % 2 == zeta else u_adapt.imag

    sp_ref = ref.sampling_points_matsubara(whichl=shared_dim-1)
    u_ref = ref.compute_unl(sp_ref, shared_dim-1)
    u_ref_real = u_ref.real if l % 2 == zeta else u_ref.imag

    num_sign_changes_adapt = np.sum(u_adapt_real[:-1] * u_adapt_real[1:] < 0)
    num_sign_changes_ref = np.sum(u_ref_real[:-1] * u_ref_real[1:] < 0)
    assert num_sign_changes_adapt == num_sign_changes_ref


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_sampling_x(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())
    sp_adapt = adapt.sampling_points_x(whichl=shared_dim-1)
    sp_ref = ref.sampling_points_x(whichl=shared_dim-1)
    np.testing.assert_allclose(sp_adapt, sp_ref, rtol=0, atol=1e-8)


@pytest.mark.parametrize("stat,lambda_", COMPARE_PARAMS)
def test_sampling_y(stat, lambda_, adapters, references):
    adapt = adapters[stat, lambda_]
    ref = references[stat, lambda_]
    shared_dim = min(adapt.dim(), ref.dim())
    sp_adapt = adapt.sampling_points_y(whichl=shared_dim-1)
    sp_ref = ref.sampling_points_y(whichl=shared_dim-1)
    np.testing.assert_allclose(sp_adapt, sp_ref, rtol=0, atol=1e-8)
