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
from . import _util


def _zeta(statistics):
    """Reduced-frequency offset: 1 for fermions (odd n), 0 for bosons (even n)."""
    if statistics == 'F':
        return 1
    if statistics == 'B':
        return 0
    raise ValueError(f"invalid statistics {statistics!r}, expected 'F' or 'B'")


def _prepare_input(a, axis, expected, what):
    """Validate an input array's axis and length before it crosses to C.

    Returns ``(array, axis, ndim)`` with ``axis`` resolved to a non-negative
    dimension index (the C API takes a non-negative target dimension).
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError(f"{what} must be at least one-dimensional")
    axis = _util.normalize_axis(axis, a.ndim)
    if a.shape[axis] != expected:
        raise ValueError(
            f"{what} has length {a.shape[axis]} along axis {axis}, "
            f"expected {expected}")
    return a, axis, a.ndim


class TauSampling:
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.

    Note:
        Real-valued input (any of ``bool``, integer, ``float32``, ``float64``)
        is normalized to ``float64`` and complex input to ``complex128`` before
        it crosses the C boundary; narrow types therefore agree with the
        ``float64``/``complex128`` result to their own input precision rather
        than producing garbage.
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
            points = basis.default_tau_sampling_points(
                use_positive_taus=use_positive_taus
            )
        else:
            points = sampling_points
        points = _util.as_boundary_real(points, "sampling_points")
        if points.ndim != 1:
            raise ValueError(
                f"sampling_points must be one-dimensional, got shape {points.shape}")
        if points.size == 0:
            raise ValueError("sampling_points must not be empty")
        # np.sort returns a fresh C-contiguous array; the pointer below is
        # taken from this object, not from the caller's array.
        self.sampling_points = np.ascontiguousarray(np.sort(points),
                                                    dtype=np.float64)

        self._backend = get_default_blas_backend()
        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            # matrix: (n_points, n_funcs)
            matrix = np.asarray(basis.u(self.sampling_points).T)
            if matrix.size and not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "tau sampling matrix is not finite: at least one "
                    "augmentation of this basis is undefined in imaginary "
                    "time (MatsubaraConst is NaN in tau), so tau sampling "
                    "cannot be constructed for it")
            matrix = _util.as_boundary_real(matrix, "tau sampling matrix")
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
            if not sampling:
                raise RuntimeError("Failed to create tau sampling: null handle")
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
            Values at sampling points. ``float64`` for real input,
            ``complex128`` for complex input.
        """
        al, axis, ndim = _prepare_input(al, axis, self.basis.size,
                                        "basis coefficients")
        output_dims = list(al.shape)
        output_dims[axis] = len(self.sampling_points)
        input_dims = np.ascontiguousarray(al.shape, dtype=np.int32)

        if al.dtype.kind == "c":
            al = _util.as_boundary_complex(al, "basis coefficients")
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
            result = output['real'] + 1j * output['imag']
        else:
            al = _util.as_boundary_real(al, "basis coefficients")
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
            result = output

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to evaluate sampling: {status}")

        return result

    def fit(self, ax, axis=0):
        """
        Fit basis coefficients from sampling point values.

        Returns ``float64`` for real input and ``complex128`` for complex
        input.
        """
        ax, axis, ndim = _prepare_input(ax, axis, len(self.sampling_points),
                                        "sampling point values")
        output_dims = list(ax.shape)
        output_dims[axis] = self.basis.size
        input_dims = np.ascontiguousarray(ax.shape, dtype=np.int32)

        if ax.dtype.kind == "c":
            ax = _util.as_boundary_complex(ax, "sampling point values")
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
            result = output['real'] + 1j * output['imag']
        else:
            ax = _util.as_boundary_real(ax, "sampling point values")
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
            result = output

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to fit sampling: {status}")

        return result

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

    Note:
        ``sampling_points`` are *reduced* Matsubara indices: odd integers for
        a fermionic and even integers for a bosonic basis.  A non-integral or
        wrong-parity index raises :class:`ValueError`; it is never truncated
        or adjusted.
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
        self.positive_only = bool(positive_only)
        zeta = _zeta(basis.statistics)

        if sampling_points is None:
            points = basis.default_matsubara_sampling_points(
                positive_only=self.positive_only)
        else:
            points = sampling_points
        points = _util.as_boundary_matsubara(points, "sampling_points",
                                             zeta=zeta)
        if points.ndim != 1:
            raise ValueError(
                f"sampling_points must be one-dimensional, got shape {points.shape}")
        if points.size == 0:
            raise ValueError("sampling_points must not be empty")
        self.sampling_points = points

        self._backend = get_default_blas_backend()
        if isinstance(basis, augment.AugmentedBasis):
            # Create sampling object
            matrix = _util.as_boundary_complex(
                basis.uhat(self.sampling_points).T,
                "Matsubara sampling matrix")

            status = c_int()
            sampling = _lib.spir_matsu_sampling_new_with_matrix(
                SPIR_ORDER_ROW_MAJOR,                           # order
                _statistics_to_c(basis.statistics),                   # statistics
                c_int(basis.size),                              # basis_size
                c_bool(self.positive_only),                     # positive_only
                c_int(len(self.sampling_points)),                    # num_points
                self.sampling_points.ctypes.data_as(POINTER(c_int64)), # points
                matrix.ctypes.data_as(POINTER(c_double_complex)), # matrix
                byref(status)                                   # status
            )
            if status.value != COMPUTATION_SUCCESS:
                raise RuntimeError(f"Failed to create matsubara sampling: {status.value}")
            if not sampling:
                raise RuntimeError(
                    "Failed to create matsubara sampling: null handle")
            self._ptr = sampling
        else:
            # Create sampling object
            self._ptr = matsubara_sampling_new(basis._ptr, self.positive_only,
                                               self.sampling_points)

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
            Values at Matsubara frequencies (always ``complex128``)
        """
        al, axis, ndim = _prepare_input(al, axis, self.basis.size,
                                        "basis coefficients")
        output_dims = list(al.shape)
        output_dims[axis] = len(self.sampling_points)
        input_dims = np.ascontiguousarray(al.shape, dtype=np.int32)
        output = np.zeros(output_dims, dtype=c_double_complex)

        if al.dtype.kind == "c":
            al = _util.as_boundary_complex(al, "basis coefficients")
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
        else:
            al = _util.as_boundary_real(al, "basis coefficients")
            status = _lib.spir_sampling_eval_dz(
                self._ptr,
                self._backend,
                SPIR_ORDER_ROW_MAJOR,
                ndim,
                input_dims.ctypes.data_as(POINTER(c_int)),
                axis,
                al.ctypes.data_as(POINTER(c_double)),
                output.ctypes.data_as(POINTER(c_double_complex))
            )

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to evaluate sampling: {status}")

        return output['real'] + 1j * output['imag']

    def fit(self, ax, axis=0):
        """
        Fit basis coefficients from Matsubara frequency values.

        Returns ``complex128``; the underlying C entry point
        (``spir_sampling_fit_zz``) only exists in the complex flavour, so
        real-valued input is widened to ``complex128`` here.  Passing the raw
        buffer of a real array through a complex pointer would read twice as
        many bytes as were allocated.
        """
        ax, axis, ndim = _prepare_input(ax, axis, len(self.sampling_points),
                                        "Matsubara frequency values")
        ax = _util.as_boundary_complex(ax, "Matsubara frequency values")
        output_dims = list(ax.shape)
        output_dims[axis] = self.basis.size
        input_dims = np.ascontiguousarray(ax.shape, dtype=np.int32)
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
