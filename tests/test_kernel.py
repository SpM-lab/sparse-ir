# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for kernel functionality, following sparse-ir test patterns.
"""

import pytest
import numpy as np
import pylibsparseir
from sparse_ir.kernel import LogisticKernel, RegularizedBoseKernel, kernel_domain
from .conftest import KERNEL_LAMBDAS


class TestLogisticKernel:
    """Test LogisticKernel functionality."""

    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_creation(self, lambda_):
        """Test kernel creation for various Lambda values."""
        kernel = LogisticKernel(lambda_)
        assert kernel is not None

        # Test domain properties
        xmin, xmax, ymin, ymax = kernel_domain(kernel)
        assert xmin < xmax
        assert ymin < ymax

        # For logistic kernel, domain should be [-1, 1] x [-1, 1]
        np.testing.assert_allclose([xmin, xmax, ymin, ymax], [-1, 1, -1, 1], atol=1e-14)

    @pytest.mark.parametrize("lambda_", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_lambda(self, lambda_):
        """The cutoff is validated before the C library is called."""
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            LogisticKernel(lambda_)


class TestRegularizedBoseKernel:
    """Test RegularizedBoseKernel functionality."""

    @pytest.mark.parametrize("lambda_", KERNEL_LAMBDAS)
    def test_creation(self, lambda_):
        """Test regularized Bose kernel creation."""
        kernel = RegularizedBoseKernel(lambda_)
        assert kernel is not None

        # Test domain properties
        xmin, xmax, ymin, ymax = kernel_domain(kernel)
        assert xmin < xmax
        assert ymin < ymax

        # For regularized Bose kernel, domain should be [-1, 1] x [-1, 1]
        np.testing.assert_allclose([xmin, xmax, ymin, ymax], [-1, 1, -1, 1], atol=1e-14)

    @pytest.mark.parametrize("lambda_", [0.0, -1.0, np.inf, np.nan])
    def test_invalid_lambda(self, lambda_):
        """The cutoff is validated before the C library is called."""
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            RegularizedBoseKernel(lambda_)

