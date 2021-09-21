from types import LambdaType
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
