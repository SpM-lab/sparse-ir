# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Configuration and fixtures for pysparseir tests.

This file provides shared fixtures that are available to all tests.
Following the pattern from sparse-ir test suite.
"""

import pytest
import numpy as np
import pylibsparseir

import sparse_ir
from sparse_ir import LogisticKernel, RegularizedBoseKernel


@pytest.fixture(scope="session")
def sve_logistic():
    """SVE of the logistic kernel for Lambda = 42"""
    print("Precomputing SVEs for logistic kernel ...")
    return {
        10:     sparse_ir.compute_sve(sparse_ir.LogisticKernel(10)),
        42:     sparse_ir.compute_sve(sparse_ir.LogisticKernel(42)),
        10_000: sparse_ir.compute_sve(sparse_ir.LogisticKernel(10_000))
        }


@pytest.fixture(scope="session")
def sve_reg_bose():
    """SVE of the logistic kernel for Lambda = 42"""
    print("Precomputing SVEs for regularized Bose kernel ...")
    return {
        10:     sparse_ir.compute_sve(sparse_ir.RegularizedBoseKernel(10)),
        10_000: sparse_ir.compute_sve(sparse_ir.RegularizedBoseKernel(10_000))
        }


@pytest.fixture(scope="session")
def test_bases():
    """Precomputed test bases for common parameter sets."""
    test_params = [
        ('F', 1.0, 10.0, 1e-6),    # Small fermion
        ('F', 1.0, 42.0, 1e-8),    # Medium fermion
        ('B', 1.0, 10.0, 1e-6),    # Small boson
        ('F', 4.0, 20.0, 1e-6),    # Different beta
    ]

    # A basis that cannot be constructed is a failure, not a missing
    # precondition: let the exception propagate instead of silently handing
    # tests an incomplete dict.
    return {
        (stat, beta, wmax):
            pylibsparseir.FiniteTempBasis(stat, beta, wmax, eps)
        for stat, beta, wmax, eps in test_params
    }


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