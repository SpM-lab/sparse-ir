"""
Configuration and fixtures for pysparseir tests.

This file provides shared fixtures that are available to all tests.
Following the pattern from sparse-ir test suite.
"""

import pytest
import numpy as np
import pylibsparseir

from sparse_ir.kernel import LogisticKernel, RegularizedBoseKernel

@pytest.fixture(scope="session")
def sve_logistic():
    """Precomputed SVE results for logistic kernels with different Lambda values."""
    print("Precomputing SVEs for logistic kernel ...")
    kernels = {}
    for lambda_ in [10, 42, 1000]:
        try:
            kernel = LogisticKernel(lambda_)
            kernels[lambda_] = pylibsparseir.sve_result_new(kernel, 1e-12)
        except Exception as e:
            print(f"Failed to create SVE for lambda={lambda_}: {e}")
    return kernels


@pytest.fixture(scope="session")
def sve_reg_bose():
    """Precomputed SVE results for regularized Bose kernels."""
    print("Precomputing SVEs for regularized Bose kernel ...")
    kernels = {}
    for lambda_ in [10, 1000]:
        try:
            kernel = RegularizedBoseKernel(lambda_)
            kernels[lambda_] = pylibsparseir.sve_result_new(kernel, 1e-12)
        except Exception as e:
            print(f"Failed to create Bose SVE for lambda={lambda_}: {e}")
    return kernels


@pytest.fixture(scope="session")
def test_bases():
    """Precomputed test bases for common parameter sets."""
    print("Precomputing test bases ...")
    bases = {}

    test_params = [
        ('F', 1.0, 10.0, 1e-6),    # Small fermion
        ('F', 1.0, 42.0, 1e-8),    # Medium fermion
        ('B', 1.0, 10.0, 1e-6),    # Small boson
        ('F', 4.0, 20.0, 1e-6),    # Different beta
    ]

    for stat, beta, wmax, eps in test_params:
        try:
            basis = pylibsparseir.FiniteTempBasis(stat, beta, wmax, eps)
            bases[(stat, beta, wmax)] = basis
        except Exception as e:
            print(f"Failed to create basis {(stat, beta, wmax)}: {e}")

    return bases


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.RandomState(42)


# Test parameter sets following sparse-ir patterns
KERNEL_LAMBDAS = [10, 42, 1000]
BASIS_PARAMS = [
    ('F', 1.0, 10.0),
    ('F', 1.0, 42.0),
    ('B', 1.0, 10.0),
    ('F', 4.0, 20.0),
]
SAMPLING_PARAMS = [
    ('F', 1.0, 42.0, False),
    ('F', 1.0, 42.0, True),
    ('B', 1.0, 10.0, False),
]