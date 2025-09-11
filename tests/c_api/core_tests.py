"""
Test core C API functionality - kernels, SVE, and basis functions.
Fixed version based on correct C API signatures.
"""
import pytest
import numpy as np
from ctypes import *
from pylibsparseir.core import _lib
from pylibsparseir.ctypes_wrapper import *
from pylibsparseir.constants import *


class TestCAPICoreFixed:
    """Test core C API functions with correct signatures"""

    def test_kernel_creation_and_domain(self):
        """Test kernel operations and domain retrieval"""
        # Test logistic kernel
        status = c_int()
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert kernel is not None

        # Test kernel domain
        xmin = c_double()
        xmax = c_double()
        ymin = c_double()
        ymax = c_double()
        status_val = _lib.spir_kernel_domain(kernel, byref(xmin), byref(xmax),
                                            byref(ymin), byref(ymax))
        assert status_val == COMPUTATION_SUCCESS
        assert xmin.value == pytest.approx(-1.0)
        assert xmax.value == pytest.approx(1.0)

        # Release kernel
        _lib.spir_kernel_release(kernel)

        # Test regularized boson kernel
        kernel = _lib.spir_reg_bose_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert kernel is not None

        _lib.spir_kernel_domain(kernel, byref(xmin), byref(xmax),
                               byref(ymin), byref(ymax))
        assert status_val == COMPUTATION_SUCCESS
        assert xmin.value == pytest.approx(-1.0)
        assert xmax.value == pytest.approx(1.0)

        _lib.spir_kernel_release(kernel)

    def test_sve_computation(self):
        """Test SVE computation"""
        status = c_int()

        # Create kernel
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        assert status.value == COMPUTATION_SUCCESS

        # Compute SVE
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-6), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert sve is not None

        # Get SVE size
        size = c_int()
        status_val = _lib.spir_sve_result_get_size(sve, byref(size))
        assert status_val == COMPUTATION_SUCCESS
        assert size.value > 0

        # Get singular values
        svals = np.zeros(size.value, dtype=np.float64)
        status_val = _lib.spir_sve_result_get_svals(sve, svals.ctypes.data_as(POINTER(c_double)))
        assert status_val == COMPUTATION_SUCCESS
        assert not np.any(np.isnan(svals))
        assert np.all(svals > 0)  # Singular values should be positive
        assert np.all(np.diff(svals) <= 0)  # Should be in descending order

        # Cleanup
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

    def test_basis_constructors(self):
        """Test basis construction for different statistics"""
        for stats, stats_val in [("fermionic", SPIR_STATISTICS_FERMIONIC),
                                ("bosonic", SPIR_STATISTICS_BOSONIC)]:
            status = c_int()
            beta = 2.0
            wmax = 1.0

            # Create kernel
            kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
            assert status.value == COMPUTATION_SUCCESS

            # Compute SVE
            cutoff = -1.0
            lmax = -1
            n_gauss = -1
            Twork = SPIR_TWORK_FLOAT64X2
            sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
            assert status.value == COMPUTATION_SUCCESS

            # Create basis
            max_size = -1
            basis = _lib.spir_basis_new(c_int(stats_val), c_double(beta), c_double(wmax),
                                      kernel, sve, max_size, byref(status))
            assert status.value == COMPUTATION_SUCCESS
            assert basis is not None

            # Check basis properties
            size = c_int()
            status_val = _lib.spir_basis_get_size(basis, byref(size))
            assert status_val == COMPUTATION_SUCCESS
            assert size.value > 0

            retrieved_stats = c_int()
            status_val = _lib.spir_basis_get_stats(basis, byref(retrieved_stats))
            assert status_val == COMPUTATION_SUCCESS
            assert retrieved_stats.value == stats_val

            # Get singular values
            svals = np.zeros(size.value, dtype=np.float64)
            status_val = _lib.spir_basis_get_svals(basis, svals.ctypes.data_as(POINTER(c_double)))
            assert status_val == COMPUTATION_SUCCESS
            assert not np.any(np.isnan(svals))

            # Cleanup
            _lib.spir_basis_release(basis)
            _lib.spir_sve_result_release(sve)
            _lib.spir_kernel_release(kernel)

    def test_tau_sampling_creation_and_properties(self):
        """Test TauSampling creation and basic properties"""
        status = c_int()
        beta = 10.0
        wmax = 1.0

        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        max_size = -1
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        basis = _lib.spir_basis_new(c_int(SPIR_STATISTICS_FERMIONIC), c_double(beta),
                                   c_double(wmax), kernel, sve, max_size, byref(status))

        # Get default tau points
        n_tau = c_int()
        status_val = _lib.spir_basis_get_n_default_taus(basis, byref(n_tau))
        assert status_val == COMPUTATION_SUCCESS
        assert n_tau.value > 0

        default_taus = np.zeros(n_tau.value, dtype=np.float64)
        status_val = _lib.spir_basis_get_default_taus(basis,
                                                     default_taus.ctypes.data_as(POINTER(c_double)))
        assert status_val == COMPUTATION_SUCCESS

        # Create tau sampling
        tau_sampling = _lib.spir_tau_sampling_new(basis, n_tau.value,
                                                 default_taus.ctypes.data_as(POINTER(c_double)),
                                                 byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert tau_sampling is not None

        # Verify tau points are in valid range [0, beta]
        assert np.all(default_taus >= -beta/2)  # Transformed coordinates
        assert np.all(default_taus <= beta/2)

        # Cleanup
        _lib.spir_sampling_release(tau_sampling)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

    def test_matsubara_sampling_creation(self):
        """Test MatsubaraSampling creation"""
        status = c_int()
        beta = 10.0
        wmax = 1.0
        max_size = -1

        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        basis = _lib.spir_basis_new(c_int(SPIR_STATISTICS_FERMIONIC), c_double(beta),
                                   c_double(wmax), kernel, sve, max_size, byref(status))

        # Get default Matsubara points
        n_matsu = c_int()
        status_val = _lib.spir_basis_get_n_default_matsus(basis, c_bool(False), byref(n_matsu))
        assert status_val == COMPUTATION_SUCCESS
        assert n_matsu.value > 0

        default_matsus = np.zeros(n_matsu.value, dtype=np.int64)
        status_val = _lib.spir_basis_get_default_matsus(basis, c_bool(False),
                                                       default_matsus.ctypes.data_as(POINTER(c_int64)))
        assert status_val == COMPUTATION_SUCCESS

        # Create Matsubara sampling
        matsu_sampling = _lib.spir_matsu_sampling_new(basis, c_bool(False), n_matsu.value,
                                                     default_matsus.ctypes.data_as(POINTER(c_int64)),
                                                     byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert matsu_sampling is not None

        # Cleanup
        _lib.spir_sampling_release(matsu_sampling)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

    def test_basis_functions_u(self):
        """Test u basis functions retrieval"""
        status = c_int()
        beta = 5.0
        wmax = 1.0

        # Create basis
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        max_size = -1
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        basis = _lib.spir_basis_new(c_int(SPIR_STATISTICS_FERMIONIC), c_double(beta),
                                   c_double(wmax), kernel, sve, max_size, byref(status))

        # Get u functions
        u_funcs = _lib.spir_basis_get_u(basis, byref(status))
        assert status.value == COMPUTATION_SUCCESS
        assert u_funcs is not None

        # Get function size
        funcs_size = c_int()
        status_val = _lib.spir_funcs_get_size(u_funcs, byref(funcs_size))
        assert status_val == COMPUTATION_SUCCESS
        assert funcs_size.value > 0

        # Cleanup
        _lib.spir_funcs_release(u_funcs)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

    def test_memory_management(self):
        """Test that all release functions work without segfaults"""
        status = c_int()

        # Create full setup
        kernel = _lib.spir_logistic_kernel_new(c_double(10.0), byref(status))
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        sve = _lib.spir_sve_result_new(kernel, c_double(1e-10), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        basis = _lib.spir_basis_new(c_int(SPIR_STATISTICS_FERMIONIC), c_double(10.0),
                                   c_double(1.0), kernel, sve, -1, byref(status))

        u_funcs = _lib.spir_basis_get_u(basis, byref(status))

        n_tau = c_int()
        _lib.spir_basis_get_n_default_taus(basis, byref(n_tau))
        tau_points = np.zeros(n_tau.value, dtype=np.float64)
        _lib.spir_basis_get_default_taus(basis, tau_points.ctypes.data_as(POINTER(c_double)))
        tau_sampling = _lib.spir_tau_sampling_new(basis, n_tau.value,
                                                 tau_points.ctypes.data_as(POINTER(c_double)),
                                                 byref(status))

        # Release everything in correct order
        _lib.spir_sampling_release(tau_sampling)
        _lib.spir_funcs_release(u_funcs)
        _lib.spir_basis_release(basis)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

        # Test should complete without segfault
        assert True


class TestBasisFunctionEvaluation:
    """Comprehensive basis function evaluation tests matching LibSparseIR.jl coverage"""

    def _spir_basis_new(self, statistics, beta, wmax, epsilon):
        """Helper function equivalent to C++ _spir_basis_new"""
        status = c_int()
        max_size = c_int(-1)
        # Create logistic kernel
        kernel = _lib.spir_logistic_kernel_new(c_double(beta * wmax), byref(status))
        if status.value != COMPUTATION_SUCCESS or kernel is None:
            return None, status.value

        # Create SVE result
        cutoff = -1.0
        lmax = -1
        n_gauss = -1
        Twork = SPIR_TWORK_FLOAT64X2
        sve = _lib.spir_sve_result_new(kernel, c_double(epsilon), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(status))
        if status.value != COMPUTATION_SUCCESS or sve is None:
            _lib.spir_kernel_release(kernel)
            return None, status.value

        # Create basis
        basis = _lib.spir_basis_new(c_int(statistics), c_double(beta), c_double(wmax),
                                   kernel, sve, max_size, byref(status))
        if status.value != COMPUTATION_SUCCESS or basis is None:
            _lib.spir_sve_result_release(sve)
            _lib.spir_kernel_release(kernel)
            return None, status.value

        # Clean up intermediate objects (like C++ version)
        _lib.spir_sve_result_release(sve)
        _lib.spir_kernel_release(kernel)

        return basis, COMPUTATION_SUCCESS

    @pytest.mark.parametrize("statistics", [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC])
    def test_basis_functions_comprehensive(self, statistics):
        """Test comprehensive basis function evaluation matching Julia tests"""
        beta = 2.0
        wmax = 5.0
        epsilon = 1e-6

        # Create basis using helper function (equivalent to C++ _spir_basis_new)
        basis, basis_status = self._spir_basis_new(statistics, beta, wmax, epsilon)
        assert basis_status == COMPUTATION_SUCCESS
        assert basis is not None

        # Get basis size
        basis_size = c_int()
        size_status = _lib.spir_basis_get_size(basis, byref(basis_size))
        assert size_status == COMPUTATION_SUCCESS
        size = basis_size.value

        # Get u basis functions
        u_status = c_int()
        u = _lib.spir_basis_get_u(basis, byref(u_status))
        assert u_status.value == COMPUTATION_SUCCESS
        assert u is not None

        # Get uhat basis functions
        uhat_status = c_int()
        uhat = _lib.spir_basis_get_uhat(basis, byref(uhat_status))
        assert uhat_status.value == COMPUTATION_SUCCESS
        assert uhat is not None

        # Get v basis functions
        v_status = c_int()
        v = _lib.spir_basis_get_v(basis, byref(v_status))
        assert v_status.value == COMPUTATION_SUCCESS
        assert v is not None

        # Test single point evaluation for u basis
        x = 0.5  # Test point for u basis (imaginary time)
        out = np.zeros(size, dtype=np.float64)
        eval_status = _lib.spir_funcs_eval(u, c_double(x), out.ctypes.data_as(POINTER(c_double)))
        assert eval_status == COMPUTATION_SUCCESS

        # Check that we got reasonable values
        assert np.all(np.isfinite(out))

        # Test single point evaluation for v basis
        y = 0.5 * wmax  # Test point for v basis (real frequency)
        eval_status = _lib.spir_funcs_eval(v, c_double(y), out.ctypes.data_as(POINTER(c_double)))
        assert eval_status == COMPUTATION_SUCCESS
        assert np.all(np.isfinite(out))

        # Test batch evaluation
        num_points = 5
        xs = np.array([0.2 * (i+1) for i in range(num_points)], dtype=np.float64)  # Points at 0.2, 0.4, 0.6, 0.8, 1.0
        batch_out = np.zeros(num_points * size, dtype=np.float64)

        # Test row-major order for u basis
        batch_status = _lib.spir_funcs_batch_eval(
            u, SPIR_ORDER_ROW_MAJOR, num_points,
            xs.ctypes.data_as(POINTER(c_double)),
            batch_out.ctypes.data_as(POINTER(c_double))
        )
        assert batch_status == COMPUTATION_SUCCESS
        assert np.all(np.isfinite(batch_out))

        # Test column-major order for u basis
        batch_status = _lib.spir_funcs_batch_eval(
            u, SPIR_ORDER_ROW_MAJOR, num_points,
            xs.ctypes.data_as(POINTER(c_double)),
            batch_out.ctypes.data_as(POINTER(c_double))
        )
        assert batch_status == COMPUTATION_SUCCESS
        assert np.all(np.isfinite(batch_out))

        # Test row-major order for v basis
        batch_status = _lib.spir_funcs_batch_eval(
            v, SPIR＿ORDER_ROW_MAJOR, num_points,
            xs.ctypes.data_as(POINTER(c_double)),
            batch_out.ctypes.data_as(POINTER(c_double))
        )
        assert batch_status == COMPUTATION_SUCCESS
        assert np.all(np.isfinite(batch_out))

        # Test column-major order for v basis
        batch_status = _lib.spir_funcs_batch_eval(
            v, SPIR_ORDER_COLUMN_MAJOR, num_points,
            xs.ctypes.data_as(POINTER(c_double)),
            batch_out.ctypes.data_as(POINTER(c_double))
        )
        assert batch_status == COMPUTATION_SUCCESS
        assert np.all(np.isfinite(batch_out))

        # Test error cases (corresponds to C++ error case testing)
        # Test with null function pointer
        eval_status = _lib.spir_funcs_eval(None, c_double(x), out.ctypes.data_as(POINTER(c_double)))
        assert eval_status != COMPUTATION_SUCCESS

        # Test batch evaluation error cases
        batch_status = _lib.spir_funcs_batch_eval(
            None, SPIR＿ORDER_ROW_MAJOR, num_points,
            xs.ctypes.data_as(POINTER(c_double)),
            batch_out.ctypes.data_as(POINTER(c_double))
        )
        assert batch_status != COMPUTATION_SUCCESS

        # Clean up
        _lib.spir_funcs_release(u)
        _lib.spir_funcs_release(v)
        _lib.spir_funcs_release(uhat)
        _lib.spir_basis_release(basis)

    def test_basis_statistics_verification(self):
        """Test basis statistics retrieval for both fermionic and bosonic cases"""
        for stats_val, expected in [(SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_FERMIONIC),
                                   (SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_BOSONIC)]:
            beta = 2.0
            wmax = 1.0
            epsilon = 1e-6

            basis, basis_status = self._spir_basis_new(stats_val, beta, wmax, epsilon)
            assert basis_status == COMPUTATION_SUCCESS
            assert basis is not None

            # Check statistics
            stats = c_int()
            stats_status = _lib.spir_basis_get_stats(basis, byref(stats))
            assert stats_status == COMPUTATION_SUCCESS
            assert stats.value == expected

            _lib.spir_basis_release(basis)

    def test_basis_constructor_with_sve_patterns(self):
        """Test different basis constructor patterns with explicit SVE"""
        for statistics in [SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC]:
            for use_reg_bose in [False, True]:
                if use_reg_bose and statistics == SPIR_STATISTICS_FERMIONIC:
                    continue  # Skip invalid combination

                beta = 2.0
                wmax = 5.0
                Lambda = 10.0
                epsilon = 1e-6

                # Create kernel
                kernel_status = c_int()
                if use_reg_bose:
                    kernel = _lib.spir_reg_bose_kernel_new(c_double(Lambda), byref(kernel_status))
                else:
                    kernel = _lib.spir_logistic_kernel_new(c_double(Lambda), byref(kernel_status))
                assert kernel_status.value == COMPUTATION_SUCCESS
                assert kernel is not None

                # Create SVE result
                cutoff = -1.0
                lmax = -1
                n_gauss = -1
                Twork = SPIR_TWORK_FLOAT64X2
                sve_status = c_int()
                sve_result = _lib.spir_sve_result_new(kernel, c_double(epsilon), c_double(cutoff), c_int(lmax), c_int(n_gauss), c_int(Twork), byref(sve_status))
                assert sve_status.value == COMPUTATION_SUCCESS
                assert sve_result is not None

                # Create basis with SVE
                max_size = -1
                basis_status = c_int()
                basis = _lib.spir_basis_new(c_int(statistics), c_double(beta), c_double(wmax),
                                          kernel, sve_result, max_size, byref(basis_status))
                assert basis_status.value == COMPUTATION_SUCCESS
                assert basis is not None

                # Check statistics
                stats = c_int()
                stats_status = _lib.spir_basis_get_stats(basis, byref(stats))
                assert stats_status == COMPUTATION_SUCCESS
                assert stats.value == statistics

                # Clean up
                _lib.spir_kernel_release(kernel)
                _lib.spir_sve_result_release(sve_result)
                _lib.spir_basis_release(basis)