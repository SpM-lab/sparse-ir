"""
High-level Python classes for sparse sampling
"""

import numpy as np
from ctypes import POINTER, c_double, c_int, byref
from pylibsparseir.core import c_double_complex, tau_sampling_new, tau_sampling_new_with_matrix, matsubara_sampling_new, matsubara_sampling_new_with_matrix, _lib
from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR
from . import augment

class TauSampling:
    """Sparse sampling in imaginary time."""

    def __init__(self, basis, sampling_points=None, use_positive_taus=False):
        """
        Initialize tau sampling.

        Parameters:
        -----------
        basis : FiniteTempBasis
            Finite temperature basis
        sampling_points : array_like, optional
            Tau sampling points. If None, use default.
        use_positive_taus : bool, optional
            If `use_positive_taus=False`, the sampling points are within
            the range [-β/2, β/2] and the distribution is symmetric.
            If `use_positive_taus=True`, the sampling points are
            folded to the positive tau domain [0, β), which is
            the default behavior of sparseir 1.x.x.
        """
        self.basis = basis

        if sampling_points is None:
            self.sampling_points = basis.default_tau_sampling_points()
            if use_positive_taus:
                self.sampling_points = np.mod(self.sampling_points, basis.beta)
        else:
            self.sampling_points = np.asarray(sampling_points, dtype=np.float64)

        self.sampling_points = np.sort(self.sampling_points)
        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            matrix = basis.u(self.sampling_points).T
            self._ptr = tau_sampling_new_with_matrix(basis, basis.statistics, self.sampling_points, matrix)
        else:
            # Create sampling object
            self._ptr = tau_sampling_new(basis._ptr, self.sampling_points)

    @property
    def tau(self):
        """Tau sampling points."""
        return self.sampling_points

    def evaluate(self, al, axis=0):
        """
        Transform basis coefficients to sampling points.

        Parameters:
        -----------
        al : array_like
            Basis coefficients
        axis : int, optional
            Axis along which to transform

        Returns:
        --------
        ndarray
            Values at sampling points
        """
        al = np.ascontiguousarray(al)
        output_dims = list(al.shape)
        ndim = len(output_dims)
        input_dims = np.asarray(al.shape, dtype=np.int32)
        output_dims[axis] = len(self.sampling_points)
        if al.dtype.kind == "f":
            output = np.zeros(output_dims, dtype=np.float64)

            status = _lib.spir_sampling_eval_dd(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                al.ctypes.data_as(POINTER(c_double)),
                output.ctypes.data_as(POINTER(c_double))
            )
        elif al.dtype.kind == "c":
            output = np.zeros(output_dims, dtype=c_double_complex)

            status = _lib.spir_sampling_eval_zz(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                al.ctypes.data_as(POINTER(c_double_complex)),
                output.ctypes.data_as(POINTER(c_double_complex))
            )
            output = output['real'] + 1j * output['imag']
        else:
            raise ValueError(f"Unsupported dtype: {al.dtype}")

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to evaluate sampling: {status}")

        return output

    def fit(self, ax, axis=0):
        """
        Fit basis coefficients from sampling point values.
        """
        ax = np.ascontiguousarray(ax)
        ndim = len(ax.shape)
        input_dims = np.asarray(ax.shape, dtype=np.int32)
        output_dims = list(ax.shape)
        output_dims[axis] = self.basis.size
        if ax.dtype.kind == "f":
            output = np.zeros(output_dims, dtype=np.float64)
            status = _lib.spir_sampling_fit_dd(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
            ndim,
            input_dims.ctypes.data_as(POINTER(c_int)),
            axis,
            ax.ctypes.data_as(POINTER(c_double)),
            output.ctypes.data_as(POINTER(c_double))
        )
        elif ax.dtype.kind == "c":
            output = np.zeros(output_dims, dtype=c_double_complex)
            status = _lib.spir_sampling_fit_zz(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                ax.ctypes.data_as(POINTER(c_double_complex)),
                output.ctypes.data_as(POINTER(c_double_complex))
            )
            output = output['real'] + 1j * output['imag']
        else:
            raise ValueError(f"Unsupported dtype: {ax.dtype}")

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to fit sampling: {status}")

        return output

    @property
    def cond(self):
        """Condition number of the sampling matrix."""
        cond = c_double()
        status = _lib.spir_sampling_get_cond_num(self._ptr, byref(cond))
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to get condition number: {status}")
        return cond.value

    def __repr__(self):
        return f"TauSampling(n_points={len(self.sampling_points)})"


class MatsubaraSampling:
    """Sparse sampling in Matsubara frequencies."""

    def __init__(self, basis, sampling_points=None, positive_only=False):
        """
        Initialize Matsubara sampling.

        Parameters:
        -----------
        basis : FiniteTempBasis
            Finite temperature basis
        sampling_points : array_like, optional
            Matsubara frequency indices. If None, use default.
        positive_only : bool, optional
            If True, use only positive frequencies
        """
        self.basis = basis
        self.positive_only = positive_only

        if sampling_points is None:
            self.sampling_points = basis.default_matsubara_sampling_points(positive_only=positive_only)
        else:
            self.sampling_points = np.asarray(sampling_points, dtype=np.int64)

        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            matrix = basis.uhat(self.sampling_points).T
            matrix = np.ascontiguousarray(matrix, dtype=np.complex128)

            self._ptr = matsubara_sampling_new_with_matrix(
                basis.statistics,
                basis.size,
                positive_only,
                self.sampling_points,
                matrix
            )
        else:
            # Create sampling object
            self._ptr = matsubara_sampling_new(basis._ptr, positive_only, self.sampling_points)

    @property
    def wn(self):
        """Matsubara frequency indices."""
        return self.sampling_points

    def evaluate(self, al, axis=0):
        """
        Transform basis coefficients to sampling points.

        Parameters:
        -----------
        al : array_like
            Basis coefficients
        axis : int, optional
            Axis along which to transform

        Returns:
        --------
        ndarray
            Values at Matsubara frequencies (complex)
        """
        # For better numerical stability, we need to make the input array contiguous.
        al = np.ascontiguousarray(al)
        output_dims = list(al.shape)
        ndim = len(output_dims)
        input_dims = np.asarray(al.shape, dtype=np.int32)
        output_dims[axis] = len(self.sampling_points)
        output_cdouble_complex = np.zeros(output_dims, dtype=c_double_complex)
        if al.dtype.kind == "f":
            status = _lib.spir_sampling_eval_dz(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                al.ctypes.data_as(POINTER(c_double)),
                output_cdouble_complex.ctypes.data_as(POINTER(c_double_complex))
            )
            output = output_cdouble_complex['real'] + 1j * output_cdouble_complex['imag']
        elif al.dtype.kind == "c":
            status = _lib.spir_sampling_eval_zz(
                self._ptr,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                al.ctypes.data_as(POINTER(c_double_complex)),
                output_cdouble_complex.ctypes.data_as(POINTER(c_double_complex))
            )
            output = output_cdouble_complex['real'] + 1j * output_cdouble_complex['imag']
        else:
            raise ValueError(f"Unsupported dtype: {al.dtype}")

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to evaluate sampling: {status}")

        return output

    def fit(self, ax, axis=0):
        """
        Fit basis coefficients from Matsubara frequency values.
        """
        ax = np.ascontiguousarray(ax)
        ndim = len(ax.shape)
        input_dims = np.asarray(ax.shape, dtype=np.int32)
        output_dims = list(ax.shape)
        output_dims[axis] = self.basis.size
        output = np.zeros(output_dims, dtype=c_double_complex)

        status = _lib.spir_sampling_fit_zz(
            self._ptr,
            SPIR_ORDER_ROW_MAJOR,
            ndim,
            input_dims.ctypes.data_as(POINTER(c_int)),
            axis,
            ax.ctypes.data_as(POINTER(c_double_complex)),
            output.ctypes.data_as(POINTER(c_double_complex))
        )
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to fit sampling: {status}")
        return output['real']

    @property
    def cond(self):
        """Condition number of the sampling matrix."""
        cond = c_double()
        status = _lib.spir_sampling_get_cond_num(self._ptr, byref(cond))
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to get condition number: {status}")
        return cond.value

    def __repr__(self):
        return f"MatsubaraSampling(n_points={len(self.sampling_points)}, positive_only={self.positive_only})"