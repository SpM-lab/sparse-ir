# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
High-level Python classes for sparse sampling
"""

import numpy as np
from ctypes import POINTER, c_double, c_int, byref, c_bool, c_int64
from pylibsparseir.core import (
    c_double_complex,
    get_default_blas_backend,
    matsubara_sampling_new,
    tau_sampling_new,
    _lib,
    _statistics_to_c,
)
from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR
from . import augment

class TauSampling:
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.
    """

    def __init__(self, basis, sampling_points=None, use_positive_taus=True):
        """
        Initialize tau sampling.

        Parameters:
        -----------
        basis : FiniteTempBasis
            Finite temperature basis
        sampling_points : array_like, optional
            Tau sampling points. If None, use default.
        use_positive_taus : bool, optional
            If `use_positive_taus=True`, the sampling points are
            folded to the positive tau domain [0, β) [default].
            If `use_positive_taus=False`, the sampling points are within
            the range [-β/2, β/2] and the distribution is symmetric.
        """
        self.basis = basis

        if sampling_points is None:
            self.sampling_points = basis.default_tau_sampling_points(
                use_positive_taus=use_positive_taus
            )
        else:
            self.sampling_points = np.asarray(sampling_points, dtype=np.float64)

        self.sampling_points = np.sort(self.sampling_points)
        self._backend = get_default_blas_backend()
        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            # matrix: (n_points, n_funcs)
            matrix = np.ascontiguousarray(basis.u(self.sampling_points).T)
            status = c_int()
            sampling = _lib.spir_tau_sampling_new_with_matrix(
                SPIR_ORDER_ROW_MAJOR,
                _statistics_to_c(basis.statistics),
                basis.size,
                self.sampling_points.size,
                self.sampling_points.ctypes.data_as(POINTER(c_double)),
                matrix.ctypes.data_as(POINTER(c_double)),
                byref(status)
            )
            if status.value != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to create tau sampling: {status.value}")
            self._ptr = sampling
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
                self._backend,
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
                self._backend,
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
                self._backend,
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
                self._backend,
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
    """Sparse sampling in Matsubara frequencies.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary frequencies.

    By setting ``positive_only=True``, one assumes that functions to be fitted
    are symmetric in Matsubara frequency, i.e.::

        Ghat(iv) == Ghat(-iv).conj()

    or equivalently, that they are purely real in imaginary time.  In this
    case, sparse sampling is performed over non-negative frequencies only,
    cutting away half of the necessary sampling space.
    """

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

        self._backend = get_default_blas_backend()
        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            matrix = basis.uhat(self.sampling_points).T
            matrix = np.ascontiguousarray(matrix, dtype=np.complex128)

            status = c_int()
            sampling = _lib.spir_matsu_sampling_new_with_matrix(
                SPIR_ORDER_ROW_MAJOR,                           # order
                _statistics_to_c(basis.statistics),                   # statistics
                c_int(basis.size),                              # basis_size
                c_bool(positive_only),                          # positive_only
                c_int(len(self.sampling_points)),                    # num_points
                self.sampling_points.ctypes.data_as(POINTER(c_int64)), # points
                matrix.ctypes.data_as(POINTER(c_double_complex)), # matrix
                byref(status)                                   # status
            )
            if status.value != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to create matsubara sampling: {status.value}")
            self._ptr = sampling
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
                self._backend,
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
                self._backend,
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
            self._backend,
            SPIR_ORDER_ROW_MAJOR,
            ndim,
            input_dims.ctypes.data_as(POINTER(c_int)),
            axis,
            ax.ctypes.data_as(POINTER(c_double_complex)),
            output.ctypes.data_as(POINTER(c_double_complex))
        )
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to fit sampling: {status}")
        return output['real'] + 1j * output['imag']

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
