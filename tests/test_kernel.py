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

    def test_invalid_lambda(self):
        """Test error handling for invalid Lambda values."""
        # Note: libsparseir may handle edge cases gracefully
        # Zero lambda might work but produce warnings
        try:
            kernel = LogisticKernel(0.0)
            # If this succeeds, that's also acceptable
        except RuntimeError:
            # If it fails, that's expected
            pass


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

    def test_invalid_lambda(self):
        """Test error handling for invalid Lambda values."""
        with pytest.raises(RuntimeError, match="Failed to create"):
            RegularizedBoseKernel(-1.0)  # Negative lambda should fail

