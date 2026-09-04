# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Test cases for sampling functionality
"""

import pytest
import numpy as np
import sparse_ir


class TestTauSampling:
    """Test TauSampling class."""

    @pytest.fixture
    def basis(self):
        """Create a test basis."""
        return sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

    def test_creation_default_points(self, basis):
        """Test TauSampling creation with default points."""
        sampling = sparse_ir.TauSampling(basis)

        assert len(sampling.tau) == basis.size
        # Note: tau points can extend beyond [0, beta] for numerical reasons
        assert np.all(np.isfinite(sampling.tau))  # Should be finite

    def test_creation_custom_points(self, basis):
        """Test TauSampling creation with custom points."""
        # Number of sampling points must be >= basis.size
        n_points = basis.size + 5
        custom_points = np.linspace(0, basis.beta, n_points)
        sampling = sparse_ir.TauSampling(basis, custom_points)

        assert len(sampling.tau) == n_points
        np.testing.assert_array_almost_equal(sampling.tau, custom_points)

    def test_evaluate_fit_roundtrip(self, basis):
        """Test evaluate/fit roundtrip accuracy."""
        sampling = sparse_ir.TauSampling(basis)

        # Test with different coefficient patterns
        test_cases = [
            np.array([1.0] + [0.0] * (basis.size - 1)),  # First coefficient only
            np.array([0.0, 1.0] + [0.0] * (basis.size - 2)),  # Second coefficient only
            np.random.random(basis.size),  # Random coefficients
        ]

        for al_original in test_cases:
            # Evaluate -> Fit cycle
            ax = sampling.evaluate(al_original)
            al_recovered = sampling.fit(ax)

            # Check roundtrip accuracy
            error = np.max(np.abs(al_original - al_recovered))
            assert error < 1e-12, f"Roundtrip error too large: {error}"

    def test_evaluate_fit_roundtrip_complex(self, basis):
        """Complex coefficients (e.g. off-diagonal G) must roundtrip with
        their imaginary part intact."""
        sampling = sparse_ir.TauSampling(basis)
        rng = np.random.default_rng(42)
        al_original = rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)

        ax = sampling.evaluate(al_original)
        assert np.iscomplexobj(ax)
        al_recovered = sampling.fit(ax)

        assert np.iscomplexobj(al_recovered)
        assert np.max(np.abs(al_original - al_recovered)) < 1e-12
        assert np.max(np.abs(al_original.imag - al_recovered.imag)) < 1e-12

    def test_evaluate_shape(self, basis):
        """Test evaluate output shape."""
        sampling = sparse_ir.TauSampling(basis)

        al = np.ones(basis.size)
        ax = sampling.evaluate(al)

        assert ax.shape == (len(sampling.tau),)
        assert np.all(np.isfinite(ax))

    def test_fit_shape(self, basis):
        """Test fit output shape."""
        sampling = sparse_ir.TauSampling(basis)

        ax = np.ones(len(sampling.tau))
        al = sampling.fit(ax)

        assert al.shape == (basis.size,)
        assert np.all(np.isfinite(al))

    def test_repr(self, basis):
        """Test string representation."""
        sampling = sparse_ir.TauSampling(basis)
        repr_str = repr(sampling)
        assert 'TauSampling' in repr_str
        assert str(len(sampling.tau)) in repr_str


class TestMatsubaraSampling:
    """Test MatsubaraSampling class."""

    @pytest.fixture
    def basis(self):
        """Create a test basis."""
        return sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

    def test_creation_default_points(self, basis):
        """Test MatsubaraSampling creation with default points."""
        # MatsubaraSampling creation works fine
        sampling = sparse_ir.MatsubaraSampling(basis)

        # Check that we have sampling points
        assert hasattr(sampling, 'wn')
        assert len(sampling.wn) > 0
        assert sampling.wn.dtype == np.int64

        # For fermionic, frequencies should be odd integers
        assert np.all(sampling.wn % 2 == 1)

    def test_creation_custom_points(self, basis):
        """Test MatsubaraSampling creation with custom points."""
        # Custom points for fermionic frequencies (odd integers)
        custom_wn = np.array([1, 3, 5, 7, 9], dtype=np.int64)
        sampling = sparse_ir.MatsubaraSampling(basis, custom_wn)

        assert len(sampling.wn) == len(custom_wn)
        np.testing.assert_array_equal(sampling.wn, custom_wn)

    def test_evaluate_fit_roundtrip_complex(self, basis):
        """Complex coefficients (e.g. off-diagonal G) must roundtrip with
        their imaginary part intact."""
        sampling = sparse_ir.MatsubaraSampling(basis)
        rng = np.random.default_rng(42)
        al_original = rng.normal(size=basis.size) + 1j * rng.normal(size=basis.size)

        ax = sampling.evaluate(al_original)
        al_recovered = sampling.fit(ax)

        assert np.iscomplexobj(al_recovered)
        assert np.max(np.abs(al_original - al_recovered)) < 1e-12
        assert np.max(np.abs(al_original.imag - al_recovered.imag)) < 1e-12

    def test_fit_real_input_matches_complex_reference(self, basis):
        """Regression test: MatsubaraSampling.fit with real float64 input
        must not be silently misread as complex128 data (out-of-bounds
        read producing garbage). The real-input result must agree with
        the complex128 reference obtained by fitting the same values cast
        to complex128."""
        sampling = sparse_ir.MatsubaraSampling(basis)
        rng = np.random.default_rng(42)
        al_original = rng.normal(size=basis.size)

        ax = sampling.evaluate(al_original)
        # ax is complex (Matsubara values are always complex); take the
        # real part to exercise the real-dtype input code path of fit().
        ax_real = np.ascontiguousarray(ax.real)
        ax_complex_ref = np.ascontiguousarray(ax_real, dtype=np.complex128)

        al_from_real = sampling.fit(ax_real)
        al_from_complex_ref = sampling.fit(ax_complex_ref)

        assert np.max(np.abs(al_from_real - al_from_complex_ref)) < 1e-12

    def test_repr(self, basis):
        """Test string representation."""
        sampling = sparse_ir.MatsubaraSampling(basis)
        repr_str = repr(sampling)
        assert 'MatsubaraSampling' in repr_str
        assert str(len(sampling.wn)) in repr_str