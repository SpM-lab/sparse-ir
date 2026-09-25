# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for FiniteTempBasis functionality
"""

import numpy as np
import sparse_ir


class TestFiniteTempBasis:
    """Test FiniteTempBasis class."""

    def test_basic_creation(self):
        """Test basic FiniteTempBasis creation."""
        beta = 10.0
        wmax = 8.0
        eps = 1e-6

        # Test fermion basis
        basis_f = sparse_ir.FiniteTempBasis('F', beta, wmax, eps)
        assert basis_f.statistics == 'F'
        assert basis_f.beta == beta
        assert basis_f.wmax == wmax
        assert basis_f.lambda_ == beta * wmax
        assert basis_f.size > 0
        assert len(basis_f.s) == basis_f.size

        # Test boson basis
        basis_b = sparse_ir.FiniteTempBasis('B', beta, wmax, eps)
        assert basis_b.statistics == 'B'
        assert basis_b.beta == beta
        assert basis_b.wmax == wmax

    def test_singular_values(self):
        """Test singular values properties."""
        basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

        # Singular values should be positive and decreasing
        s = basis.s
        assert np.all(s > 0)
        assert np.all(s[:-1] >= s[1:])  # Decreasing

        # Test significance
        sig = basis.significance
        assert sig[0] == 1.0  # First should be 1
        assert np.all(sig <= 1.0)  # All should be <= 1

        # Test accuracy
        acc = basis.accuracy
        assert 0 < acc <= 1.0

    def test_basis_function_evaluation(self):
        """Test basis function evaluation."""
        basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

        sign = (-1.0) ** np.arange(basis.size)[:, None]

        # u functions (imaginary time): u_l(beta - tau) = (-1)^l u_l(tau)
        tau_points = np.linspace(0, basis.beta, 5)
        u_vals = basis.u(tau_points)
        assert u_vals.shape == (basis.size, len(tau_points))
        assert np.linalg.norm(u_vals) > 0
        np.testing.assert_allclose(basis.u(basis.beta - tau_points),
                                   sign * u_vals, rtol=0, atol=1e-12)

        # v functions (real frequency): v_l(-w) = (-1)^l v_l(w)
        omega_points = np.linspace(-8, 8, 5)
        v_vals = basis.v(omega_points)
        assert v_vals.shape == (basis.size, len(omega_points))
        assert np.linalg.norm(v_vals) > 0
        np.testing.assert_allclose(basis.v(-omega_points), sign * v_vals,
                                   rtol=0, atol=1e-12)

        # uhat functions (fermionic, so odd reduced frequencies)
        n_points = np.array([1, 3, 5, 7, 9], dtype=np.int64)
        uhat_vals = basis.uhat(n_points)
        assert uhat_vals.shape == (basis.size, len(n_points))
        assert np.linalg.norm(uhat_vals) > 0
        np.testing.assert_allclose(basis.uhat(-n_points), np.conj(uhat_vals),
                                   rtol=0, atol=1e-14)

    def test_default_sampling_points(self):
        """Test default sampling points."""
        basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

        # Tau sampling points, folded to (0, beta) and sorted by default
        tau_points = basis.default_tau_sampling_points()
        assert len(tau_points) == basis.size
        assert np.all((tau_points > 0) & (tau_points < basis.beta))
        assert np.all(np.diff(tau_points) > 0)

        # Unfolded points lie in [-beta/2, beta/2]
        centered = basis.default_tau_sampling_points(use_positive_taus=False)
        assert np.all(np.abs(centered) <= basis.beta / 2)

        # Matsubara sampling points: odd (fermionic) and symmetric
        matsu_points = basis.default_matsubara_sampling_points()
        assert len(matsu_points) >= basis.size
        assert np.all(matsu_points % 2 == 1)
        np.testing.assert_array_equal(np.sort(matsu_points), np.sort(-matsu_points))

        matsu_points_pos = basis.default_matsubara_sampling_points(positive_only=True)
        assert len(matsu_points_pos) > 0
        assert np.all(matsu_points_pos >= 0)

    def test_repr(self):
        """Test string representation."""
        basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)
        repr_str = repr(basis)
        assert 'FiniteTempBasis' in repr_str
        assert 'F' in repr_str
        assert '10.0' in repr_str
        assert '8.0' in repr_str


def test_finite_temp_bases():
    """Test finite_temp_bases factory function."""
    beta = 5.0
    wmax = 4.0
    eps = 1e-8

    f_basis, b_basis = sparse_ir.finite_temp_bases(beta, wmax, eps)

    assert f_basis.statistics == 'F'
    assert b_basis.statistics == 'B'
    assert f_basis.beta == beta
    assert b_basis.beta == beta
    assert f_basis.wmax == wmax
    assert b_basis.wmax == wmax

class TestBasisFunctionEvaluation:
    """Test basis function evaluation accuracy."""
    def test_u_function_finite(self):
        """Test that u functions evaluate to finite values."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)

        # Test at various tau points
        tau_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        u_vals = basis.u(tau_points)

        assert u_vals.shape == (basis.size, len(tau_points))
        assert np.all(np.isfinite(u_vals)), "All u function values should be finite"

        # u functions should not be trivially zero
        assert np.any(np.abs(u_vals) > 1e-10), "u functions should not be all zero"

    def test_v_function_finite(self):
        """Test that v functions evaluate to finite values."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)

        # Test at various omega points
        omega_points = np.linspace(-8, 8, 9)
        v_vals = basis.v(omega_points)

        assert v_vals.shape == (basis.size, len(omega_points))
        assert np.all(np.isfinite(v_vals)), "All v function values should be finite"

        # v functions should not be trivially zero
        assert np.any(np.abs(v_vals) > 1e-10), "v functions should not be all zero"