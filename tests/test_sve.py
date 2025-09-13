# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for SVE and basis accuracy, following sparse-ir test patterns.
"""

import pytest
import numpy as np
import sparse_ir
from .conftest import BASIS_PARAMS


class TestSVEAccuracy:
    """Test SVE accuracy and basis properties."""

    @pytest.mark.parametrize("stat,beta,wmax", BASIS_PARAMS)
    def test_accuracy_bounds(self, stat, beta, wmax):
        """Test that basis accuracy meets expected bounds."""
        eps = 1e-6
        basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps)

        # Basic properties
        assert 0 < basis.accuracy <= basis.significance[-1]
        assert basis.significance[0] == 1.0
        assert basis.accuracy <= basis.s[-1] / basis.s[0]

        # Accuracy should be better than requested epsilon (with some tolerance)
        assert basis.accuracy <= 10 * eps, f"Accuracy {basis.accuracy} should be close to eps {eps}"

    @pytest.mark.parametrize("stat,beta,wmax", BASIS_PARAMS)
    def test_singular_value_properties(self, stat, beta, wmax):
        """Test singular value properties."""
        basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, 1e-6)

        s = basis.s

        # All positive
        assert np.all(s > 0), "All singular values should be positive"

        # Monotonically decreasing
        assert np.all(s[:-1] >= s[1:]), "Singular values should be monotonically decreasing"

        # First value should be approximately 1 for logistic kernel
        # (this is a property of the normalized kernel)
        assert s[0] > 0.5, "First singular value should be reasonably large"

        # Check significance array
        sig = basis.significance
        assert sig[0] == 1.0, "First significance should be 1"
        assert np.all(sig >= 0), "All significance values should be non-negative"
        assert np.all(sig <= 1), "All significance values should be <= 1"
        np.testing.assert_allclose(sig, s / s[0], rtol=1e-14)

    @pytest.mark.parametrize("lambda_", [10, 42, 1000])
    def test_basis_size_scaling(self, lambda_):
        """Test that basis size scales appropriately with Lambda."""
        beta = 1.0
        wmax = lambda_ / beta
        eps = 1e-6

        basis = sparse_ir.FiniteTempBasis('F', beta, wmax, eps)

        # Basis size should be reasonable for the given Lambda
        # The exact scaling depends on the kernel and implementation
        expected_min_size = max(1, int(np.log(lambda_)))
        expected_max_size = max(50, int(lambda_))  # More generous upper bound

        assert expected_min_size <= basis.size <= expected_max_size, \
            f"Basis size {basis.size} not in expected range [{expected_min_size}, {expected_max_size}] for Lambda={lambda_}"

    def test_epsilon_vs_size(self):
        """Test that smaller epsilon gives larger basis size."""
        beta, wmax = 1.0, 42.0

        basis_loose = sparse_ir.FiniteTempBasis('F', beta, wmax, 1e-4)
        basis_medium = sparse_ir.FiniteTempBasis('F', beta, wmax, 1e-6)
        basis_tight = sparse_ir.FiniteTempBasis('F', beta, wmax, 1e-8)

        # Sizes should increase (or stay same) as epsilon decreases
        assert basis_loose.size <= basis_medium.size <= basis_tight.size, \
            f"Basis sizes should increase with smaller epsilon: {basis_loose.size} <= {basis_medium.size} <= {basis_tight.size}"

    @pytest.mark.parametrize("stat", ['F', 'B'])
    def test_statistics_consistency(self, stat):
        """Test that basis statistics are consistent."""
        basis = sparse_ir.FiniteTempBasis(stat, 1.0, 10.0, 1e-6)

        assert basis.statistics == stat

        # Both fermion and boson should have reasonable sizes
        assert 5 <= basis.size <= 100, f"Basis size {basis.size} seems unreasonable for stat={stat}"


class TestBasisConsistency:
    """Test consistency between different basis creation methods."""

    def test_same_parameters_same_result(self):
        """Test that same parameters give same results."""
        params = ('F', 2.0, 15.0, 1e-6)

        basis1 = sparse_ir.FiniteTempBasis(*params)
        basis2 = sparse_ir.FiniteTempBasis(*params)

        # Should have identical properties
        assert basis1.size == basis2.size
        assert basis1.lambda_ == basis2.lambda_
        assert basis1.beta == basis2.beta
        assert basis1.wmax == basis2.wmax

        # Singular values should be identical
        np.testing.assert_allclose(basis1.s, basis2.s, rtol=1e-15)

    def test_finite_temp_bases_factory(self):
        """Test finite_temp_bases factory function."""
        beta, wmax, eps = 2.0, 20.0, 1e-6

        f_basis, b_basis = sparse_ir.finite_temp_bases(beta, wmax, eps)

        # Check types
        assert f_basis.statistics == 'F'
        assert b_basis.statistics == 'B'

        # Check parameters
        for basis in [f_basis, b_basis]:
            assert basis.beta == beta
            assert basis.wmax == wmax
            assert basis.lambda_ == beta * wmax

        # Fermion and boson bases should have different sizes (typically)
        # This is not always guaranteed, but usually true
        # We just check they're both reasonable
        assert 5 <= f_basis.size <= 100
        assert 5 <= b_basis.size <= 100


