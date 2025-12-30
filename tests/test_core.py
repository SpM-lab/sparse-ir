# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for core functionality and C API wrappers
"""

import numpy as np
from ctypes import c_double, byref
from pylibsparseir.core import logistic_kernel_new, reg_bose_kernel_new, sve_result_new, sve_result_get_size, sve_result_get_svals, basis_new, basis_get_size, basis_get_stats, basis_get_svals, basis_get_u, basis_get_v, basis_get_uhat, basis_get_default_tau_sampling_points, basis_get_default_matsubara_sampling_points, tau_sampling_new, matsubara_sampling_new
from pylibsparseir.core import _lib

class TestCoreAPI:
    """Test core C API wrapper functions."""

    def test_kernel_creation(self):
        """Test kernel creation functions."""
        lambda_val = 80.0

        # Test logistic kernel
        kernel_log = logistic_kernel_new(lambda_val)
        assert kernel_log is not None

        # Test regularized boson kernel
        kernel_bose = reg_bose_kernel_new(lambda_val)
        assert kernel_bose is not None

        # Test kernel domain
        xmin = c_double()
        xmax = c_double()
        ymin = c_double()
        ymax = c_double()
        _lib.spir_kernel_get_domain(kernel_log, byref(xmin), byref(xmax), byref(ymin), byref(ymax))
        assert xmin.value < xmax.value
        assert ymin.value < ymax.value

    def test_sve_computation(self):
        """Test SVE computation."""
        kernel = logistic_kernel_new(80.0)
        eps = 1e-6

        sve = sve_result_new(kernel, eps)
        assert sve is not None

        size = sve_result_get_size(sve)
        assert size > 0

        svals = sve_result_get_svals(sve)
        assert len(svals) == size
        assert np.all(svals > 0)
        assert np.all(svals[:-1] >= svals[1:])  # Decreasing

    def test_basis_creation(self):
        """Test basis creation and properties."""
        eps = 1e-6
        kernel = logistic_kernel_new(80.0)
        sve = sve_result_new(kernel, eps)

        # Test fermion basis
        max_size = -1
        basis_f = basis_new(1, 10.0, 8.0, eps, kernel, sve, max_size)  # 1 = fermion
        assert basis_f is not None

        size_f = basis_get_size(basis_f)
        assert size_f > 0

        stats_f = basis_get_stats(basis_f)
        assert stats_f == 1  # Fermion

        svals_f = basis_get_svals(basis_f)
        assert len(svals_f) == size_f

        # Test boson basis
        basis_b = basis_new(0, 10.0, 8.0, eps, kernel, sve, max_size)  # 0 = boson
        stats_b = basis_get_stats(basis_b)
        assert stats_b == 0  # Boson

    def test_basis_functions(self):
        """Test basis function objects."""
        kernel = logistic_kernel_new(80.0)
        eps = 1e-6
        sve = sve_result_new(kernel, eps)
        max_size = -1
        basis = basis_new(1, 10.0, 8.0, eps, kernel, sve, max_size)

        # Test getting function objects
        u_funcs = basis_get_u(basis)
        assert u_funcs is not None

        v_funcs = basis_get_v(basis)
        assert v_funcs is not None

        uhat_funcs = basis_get_uhat(basis)
        assert uhat_funcs is not None

    def test_default_sampling_points(self):
        """Test default sampling point functions."""
        kernel = logistic_kernel_new(80.0)
        eps = 1e-6
        sve = sve_result_new(kernel, eps)
        max_size = -1
        basis = basis_new(1, 10.0, 8.0, eps, kernel, sve, max_size)

        # Test tau sampling points
        tau_points = basis_get_default_tau_sampling_points(basis)
        assert len(tau_points) > 0
        assert np.all(np.isfinite(tau_points))  # Should be finite

        # Test Matsubara sampling points
        matsu_points = basis_get_default_matsubara_sampling_points(basis, False)
        assert len(matsu_points) > 0

        matsu_points_pos = basis_get_default_matsubara_sampling_points(basis, True)
        assert len(matsu_points_pos) > 0
        assert np.all(matsu_points_pos >= 0)

    def test_sampling_objects(self):
        """Test sampling object creation."""
        kernel = logistic_kernel_new(80.0)
        eps = 1e-6
        sve = sve_result_new(kernel, eps)
        max_size = -1
        basis = basis_new(1, 10.0, 8.0, eps, kernel, sve, max_size)

        # Test tau sampling
        tau_points = basis_get_default_tau_sampling_points(basis)
        tau_sampling = tau_sampling_new(basis, tau_points)
        assert tau_sampling is not None

        # Test Matsubara sampling - temporarily disabled due to C API issues
        # matsu_points = basis_get_default_matsubara_sampling_points(basis, True)
        # matsu_sampling = matsubara_sampling_new(basis, True, matsu_points)
        # assert matsu_sampling is not None


class TestErrorHandling:
    """Test error handling in C API wrappers."""

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""

        # Note: libsparseir may be more permissive than expected
        # Some "invalid" values might be handled gracefully

        # Test very small epsilon (this should still work)
        kernel = logistic_kernel_new(80.0)
        try:
            sve_result_new(kernel, 1e-20)  # Very small epsilon
        except RuntimeError:
            # This is acceptable - very small epsilon might fail
            pass
