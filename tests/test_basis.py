# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for FiniteTempBasis functionality
"""

import numpy as np
import pytest
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
    def test_u_function_values(self):
        """u_l has the reflection symmetry u_l(beta - tau) = (-1)^l u_l(tau)."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)

        tau_points = np.array([0.0, 0.25, 0.5, 0.75, 1.0])   # symmetric about beta/2
        u_vals = basis.u(tau_points)

        assert u_vals.shape == (basis.size, len(tau_points))
        sign = (-1.0) ** np.arange(basis.size)
        np.testing.assert_allclose(u_vals[:, ::-1], sign[:, None] * u_vals,
                                   rtol=0, atol=1e-12 * np.abs(u_vals).max())
        # u_0 has no sign change
        assert abs(np.sign(u_vals[0]).sum()) == len(tau_points)

    def test_v_function_values(self):
        """v_l has the parity v_l(-omega) = (-1)^l v_l(omega)."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)

        omega_points = np.linspace(-8, 8, 9)                  # symmetric about 0
        v_vals = basis.v(omega_points)

        assert v_vals.shape == (basis.size, len(omega_points))
        sign = (-1.0) ** np.arange(basis.size)
        np.testing.assert_allclose(v_vals[:, ::-1], sign[:, None] * v_vals,
                                   rtol=0, atol=1e-12 * np.abs(v_vals).max())

def test_basis_truncation():
    """basis[:n] keeps the n most significant singular values and functions."""
    basis = sparse_ir.FiniteTempBasis("F", 10.0, 1.0, eps=1e-6)
    part = basis[:3]
    assert isinstance(part, sparse_ir.FiniteTempBasis)
    assert part.size == 3
    np.testing.assert_array_equal(part.s, basis.s[:3])
    np.testing.assert_allclose(part.u(0.5), basis.u(0.5)[:3], rtol=1e-14, atol=0)
    np.testing.assert_allclose(part.uhat(3), basis.uhat(3)[:3], rtol=1e-14, atol=0)
    assert basis[:basis.size].size == basis.size
    for bad in (slice(1, 3), slice(0, 4, 2)):
        with pytest.raises(ValueError, match="truncation"):
            basis[bad]
    with pytest.raises(IndexError):
        basis[:basis.size + 1]
    with pytest.raises(TypeError, match="slice"):
        basis[2]


def test_basis_set_rescale_keeps_eps():
    """FiniteTempBasisSet.rescale uses the same eps, as FiniteTempBasis.rescale does."""
    bset = sparse_ir.FiniteTempBasisSet(10.0, 8.0, 1e-6)
    new = bset.rescale(20.0)
    for stat, basis in (("F", new.basis_f), ("B", new.basis_b)):
        ref = sparse_ir.FiniteTempBasis(stat, 20.0, 4.0, 1e-6)   # same lambda and eps
        assert basis.size == ref.size
        np.testing.assert_allclose(basis.s, ref.s, rtol=1e-12, atol=0)
    assert new.basis_f.size == bset.basis_f.rescale(20.0).size
