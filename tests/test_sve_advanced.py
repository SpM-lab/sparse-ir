# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test SVE (Singular Value Expansion) functionality following sparse-ir patterns.
"""

import pytest
import numpy as np
import pylibsparseir
from pylibsparseir.core import *
from .conftest import KERNEL_LAMBDAS
from sparse_ir.kernel import LogisticKernel, RegularizedBoseKernel

class TestSVEProperties:
    """Test SVE computation and properties."""

    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_sve_accuracy_vs_epsilon(self, lambda_):
        """Test that SVE accuracy matches requested epsilon."""
        kernel = LogisticKernel(lambda_)

        # Test different epsilon values
        epsilons = [1e-4, 1e-6, 1e-8, 1e-10]

        for eps in epsilons:
            # The C entry point takes the kernel *handle*, not the Python
            # wrapper object.
            sve = sve_result_new(kernel._ptr, eps)
            size = sve_result_get_size(sve)
            svals = sve_result_get_svals(sve)

            # Actual accuracy should be <= requested epsilon
            actual_accuracy = svals[-1] / svals[0]
            assert actual_accuracy <= eps, \
                f"Actual accuracy {actual_accuracy} > requested {eps}"

            # Should have at least 1 singular value
            assert size >= 1
            assert len(svals) == size
            assert np.all(svals > 0)

    @pytest.mark.parametrize("lambda_", [10, 42])
    def test_sve_convergence(self, lambda_):
        """Test SVE convergence as epsilon decreases."""
        kernel = LogisticKernel(lambda_)

        epsilons = [1e-4, 1e-6, 1e-8]
        sizes = []

        for eps in epsilons:
            sve = sve_result_new(kernel._ptr, eps)
            sizes.append(sve_result_get_size(sve))

        # Sizes should be non-decreasing as epsilon decreases
        for i in range(len(sizes) - 1):
            assert sizes[i+1] >= sizes[i], \
                f"Size decreased from {sizes[i]} to {sizes[i+1]} as epsilon decreased"


class TestBasisFromSVE:
    """Test basis construction from SVE results."""

    @pytest.mark.parametrize("statistics", ['F', 'B'])
    def test_basis_creation_from_sve(self, statistics):
        """Test creating basis from precomputed SVE."""
        lambda_ = 42.0
        beta = 1.0
        wmax = lambda_ / beta

        eps = 1e-6

        # Create kernel and SVE
        if statistics == 'F':
            kernel = LogisticKernel(lambda_)
        else:
            kernel = RegularizedBoseKernel(lambda_)

        sve = sve_result_new(kernel._ptr, eps)

        # Create basis using C API directly.  Both the kernel and the SVE are
        # passed as handles, and ``basis_new`` takes epsilon and max_size.
        stats_int = 1 if statistics == 'F' else 0
        basis_c = basis_new(stats_int, beta, wmax, eps, kernel._ptr, sve, -1)

        # Test basic properties
        size = basis_get_size(basis_c)
        assert size > 0

        svals = basis_get_svals(basis_c)
        assert len(svals) == size
        assert np.all(svals > 0)

        stats_recovered = basis_get_stats(basis_c)
        assert stats_recovered == stats_int


class TestSVEErrorHandling:
    """Test error handling in SVE computation."""

    def test_invalid_epsilon_values(self):
        """Test error handling for invalid epsilon values."""
        kernel = LogisticKernel(42.0)

        # Test negative epsilon
        with pytest.raises(RuntimeError, match="Failed to create"):
            sve_result_new(kernel._ptr, -1e-6)

        # Test zero epsilon
        with pytest.raises(RuntimeError, match="Failed to create"):
            sve_result_new(kernel._ptr, 0.0)