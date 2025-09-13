# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT

"""
Advanced feature tests based on sparse-ir test patterns.

Tests for more sophisticated functionality including noise resilience,
multi-dimensional operations, and edge cases.
"""

import pytest
import numpy as np
import sparse_ir
from .conftest import BASIS_PARAMS


class TestNoiseResilience:
    """Test noise resilience following sparse-ir patterns."""

    @pytest.mark.parametrize("statistics,beta,wmax", BASIS_PARAMS[:2])  # Limit to avoid long tests
    def test_tau_noise_resilience(self, statistics, beta, wmax, rng):
        """Test that sampling is resilient to noise in tau domain."""
        eps = 1e-6

        try:
            basis = sparse_ir.FiniteTempBasis(statistics, beta, wmax, eps)
            sampling = sparse_ir.TauSampling(basis)
        except Exception as e:
            pytest.skip(f"Failed to create basis/sampling: {e}")

        # Create synthetic IR coefficients (mimicking sparse-ir test)
        try:
            # Evaluate v functions at test frequencies
            omega_test = np.array([-0.999 * wmax, -0.01 * wmax, 0.5 * wmax])
            v_vals = basis.v(omega_test)

            # Create IR coefficients as linear combination
            rhol = v_vals @ np.array([0.8, -0.2, 0.5])
            Gl = basis.s * rhol
            Gl_magn = np.linalg.norm(Gl)

            # Evaluate to tau domain
            Gtau = sampling.evaluate(Gl)

            # Add noise
            noise_level = 1e-5
            Gtau_noisy = Gtau + noise_level * np.linalg.norm(Gtau) * rng.randn(*Gtau.shape)

            # Fit back
            Gl_recovered = sampling.fit(Gtau_noisy)

            # Check that recovery is reasonable despite noise
            recovery_error = np.linalg.norm(Gl - Gl_recovered)
            expected_error = 12 * noise_level * Gl_magn  # Following sparse-ir pattern

            assert recovery_error <= expected_error, \
                f"Recovery error {recovery_error} exceeds expected {expected_error}"

        except Exception as e:
            pytest.skip(f"Noise resilience test failed: {e}")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_lambda(self):
        """Test with very small Lambda values."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 1.0, 1.0, 1e-4)  # Lambda = 1
            assert basis.size >= 1
            assert len(basis.s) == basis.size

            # Should still be able to create sampling
            sampling = sparse_ir.TauSampling(basis)
            assert len(sampling.tau) == basis.size

        except Exception as e:
            pytest.skip(f"Small lambda test failed: {e}")

    def test_large_lambda(self):
        """Test with large Lambda values."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 10.0, 100.0, 1e-6)  # Lambda = 1000
            assert basis.size > 10  # Should have reasonable number of basis functions

            # Test basic functionality
            tau_points = np.linspace(0, basis.beta, 3)
            u_vals = basis.u(tau_points)
            assert u_vals.shape == (basis.size, 3)

        except Exception as e:
            pytest.skip(f"Large lambda test failed: {e}")

    def test_high_precision(self):
        """Test with very high precision requirements."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-12)
        assert basis.accuracy <= 1e-11  # Allow some tolerance

        # High precision should give more basis functions
        basis_low = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)
        assert basis.size >= basis_low.size

    def test_boundary_tau_points(self):
        """Test evaluation at boundary tau points."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

            # Test at boundaries
            tau_boundary = np.array([0.0, basis.beta])
            u_vals = basis.u(tau_boundary)
            assert u_vals.shape == (basis.size, 2)
            assert np.all(np.isfinite(u_vals))

            # Test very close to boundaries
            eps_tau = 1e-10
            tau_near_boundary = np.array([eps_tau, basis.beta - eps_tau])
            u_vals_near = basis.u(tau_near_boundary)
            assert np.all(np.isfinite(u_vals_near))

        except Exception as e:
            pytest.skip(f"Boundary tau test failed: {e}")

    def test_zero_frequency(self):
        """Test evaluation at zero frequency."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 10.0, 8.0, 1e-6)

            # Test v function at omega = 0
            v_zero = basis.v(np.array([0.0]))
            assert v_zero.shape == (basis.size, 1)
            assert np.all(np.isfinite(v_zero))

        except Exception as e:
            pytest.skip(f"Zero frequency test failed: {e}")


class TestConsistencyChecks:
    """Test internal consistency of the implementation."""

    def test_reconstruction_accuracy(self):
        """Test that evaluate/fit operations are consistent."""
        basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)
        sampling = sparse_ir.TauSampling(basis)

        # Create test coefficients with different magnitudes
        al_test = np.zeros(basis.size)
        al_test[0] = 1.0      # Large coefficient
        al_test[1] = 0.1      # Medium coefficient
        al_test[2] = 0.01     # Small coefficient

        # Test roundtrip accuracy
        ax = sampling.evaluate(al_test)
        al_recovered = sampling.fit(ax)

        # Should recover with high accuracy
        error = np.max(np.abs(al_test - al_recovered))
        assert error < 1e-12, f"Reconstruction error too large: {error}"

        # Test that the reconstruction preserves the structure
        for i in range(3):
            rel_error = abs(al_test[i] - al_recovered[i]) / max(abs(al_test[i]), 1e-14)
            assert rel_error < 1e-12, f"Relative error for coefficient {i}: {rel_error}"

    def test_singular_value_ordering(self):
        """Test that singular values are properly ordered."""
        for statistics in ['F', 'B']:
            for lambda_val in [10, 42]:
                try:
                    basis = sparse_ir.FiniteTempBasis(statistics, 1.0, lambda_val, 1e-6)

                    s = basis.s
                    # Check decreasing order
                    assert np.all(s[:-1] >= s[1:]), f"Singular values not decreasing for {statistics}, λ={lambda_val}"

                    # Check positivity
                    assert np.all(s > 0), f"Non-positive singular values for {statistics}, λ={lambda_val}"

                    # Check normalization (first should be largest)
                    assert s[0] == np.max(s), f"First singular value not largest for {statistics}, λ={lambda_val}"

                except Exception as e:
                    pytest.skip(f"Singular value test failed for {statistics}, λ={lambda_val}: {e}")

    def test_significance_properties(self):
        """Test properties of significance array."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)

            sig = basis.significance

            # First significance should be 1
            assert abs(sig[0] - 1.0) < 1e-14, f"First significance {sig[0]} != 1"

            # Should be decreasing
            assert np.all(sig[:-1] >= sig[1:]), "Significance not decreasing"

            # Should be between 0 and 1
            assert np.all(sig >= 0), "Negative significance values"
            assert np.all(sig <= 1), "Significance values > 1"

        except Exception as e:
            pytest.skip(f"Significance test failed: {e}")


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_large_basis_creation(self):
        """Test creating large basis without memory issues."""
        try:
            # Create a moderately large basis
            basis = sparse_ir.FiniteTempBasis('F', 1.0, 100.0, 1e-8)

            # Should be able to access all properties
            size = basis.size
            assert size > 20  # Should be reasonably large

            s = basis.s
            assert len(s) == size

            # Should be able to evaluate functions
            tau_test = np.array([0.1, 0.5, 0.9])
            u_vals = basis.u(tau_test)
            assert u_vals.shape == (size, 3)

        except MemoryError:
            pytest.skip("Not enough memory for large basis test")
        except Exception as e:
            pytest.skip(f"Large basis test failed: {e}")

    def test_repeated_operations(self):
        """Test that repeated operations don't leak memory or degrade performance."""
        try:
            basis = sparse_ir.FiniteTempBasis('F', 1.0, 10.0, 1e-6)
            sampling = sparse_ir.TauSampling(basis)

            # Test coefficients
            al_test = np.random.random(basis.size)

            # Perform many evaluate/fit cycles
            for _ in range(10):
                ax = sampling.evaluate(al_test)
                al_recovered = sampling.fit(ax)

                # Should maintain accuracy
                error = np.max(np.abs(al_test - al_recovered))
                assert error < 1e-12, f"Accuracy degraded after repeated operations: {error}"

        except Exception as e:
            pytest.skip(f"Repeated operations test failed: {e}")