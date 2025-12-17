# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Discrete Lehmann Representation (DLR) functionality for SparseIR.

This module implements DLR basis with poles at IR extrema, providing
an alternative representation that can be more efficient for certain calculations.
"""

import ctypes
import numpy as np
from .abstract import AbstractBasis
from pylibsparseir.core import basis_get_default_omega_sampling_points
from pylibsparseir.core import _lib, COMPUTATION_SUCCESS, get_default_blas_backend
from pylibsparseir.constants import SPIR_ORDER_ROW_MAJOR

class DiscreteLehmannRepresentation(AbstractBasis):
    """Discrete Lehmann representation (DLR), with poles being extrema of IR.

    This class implements a variant of the discrete Lehmann representation
    (`DLR`_).  Instead of a truncated singular value expansion of the analytic
    continuation kernel ``K`` like the IR, the discrete Lehmann representation
    is based on a "sketching" of ``K``.  The resulting basis is a
    linear combination of discrete set of poles on the real-frequency axis,
    continued to the imaginary-frequency axis::

        G(iv) == sum(a[i] / (iv - w[i]) for i in range(L))

    Warning:
        The poles on the real-frequency axis selected for the DLR are based
        on a rank-revealing decomposition, which offers accuracy guarantees.
        Here, we instead select the pole locations based on the zeros of the IR
        basis functions on the real axis, which is a heuristic.  We do not
        expect that difference to matter, but please don't blame the DLR
        authors if we were wrong :-)

    .. _DLR: https://doi.org/10.1103/PhysRevB.105.235115
    """

    def __init__(self, basis: AbstractBasis, poles=None):
        status = ctypes.c_int()
        if poles is None:
            poles = basis_get_default_omega_sampling_points(basis._ptr)
        self._basis = basis
        self._poles = np.ascontiguousarray(poles)
        self._backend = get_default_blas_backend()
        self._ptr = _lib.spir_dlr_new_with_poles(basis._ptr, len(poles), poles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), status)
        if status.value != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to create DLR basis: {status.value}")

    @property
    def u(self): return self._basis._u

    @property
    def uhat(self): return self._basis._uhat

    @property
    def statistics(self):
        return self._basis.statistics

    @property
    def sampling_points(self):
        return self._poles

    @property
    def shape(self): return self.size,

    @property
    def size(self): return len(self._poles)

    @property
    def basis(self) -> AbstractBasis:
        """ Underlying basis """
        return self._basis

    @property
    def lambda_(self):
        return self._basis.lambda_

    @property
    def beta(self):
        return self._basis.beta

    @property
    def wmax(self):
        return self._basis.wmax

    @property
    def significance(self):
        return np.ones(self.shape)

    @property
    def accuracy(self):
        return self._basis.accuracy

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """From IR to DLR

        Convert expansion coefficients from IR basis to DLR basis.

        Parameters
        ----------
        gl : array_like
            Expansion coefficients in IR
        axis : int, optional
            Axis along which to convert

        Returns
        -------
        array_like
            Expansion coefficients in DLR
        """
        gl = np.ascontiguousarray(gl)
        if gl.shape[axis] != self.basis.size:
            raise ValueError(f"Input array has wrong size along dimension {axis}")

        output_dims = list(gl.shape)
        output_dims[axis] = self.size
        output = np.zeros(output_dims, dtype=gl.dtype)

        ndim = len(gl.shape)
        input_dims = np.asarray(gl.shape, dtype=np.int32)
        target_dim = axis
        order = SPIR_ORDER_ROW_MAJOR

        if gl.dtype.kind == 'f':
            ret = _lib.spir_ir2dlr_dd(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                target_dim,
                gl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        elif gl.dtype.kind == 'c':
            ret = _lib.spir_ir2dlr_zz(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                target_dim,
                # TODO: use complex data
                gl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        else:
            raise ValueError(f"Unsupported dtype: {gl.dtype}")
        if ret != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to convert IR to DLR: {ret}")
        return output

    def to_IR(self, g_dlr: np.ndarray, axis=0) -> np.ndarray:
        """From DLR to IR

        Convert expansion coefficients from DLR basis to IR basis.

        Parameters
        ----------
        g_dlr : array_like
            Expansion coefficients in DLR
        axis : int, optional
            Axis along which to convert

        Returns
        -------
        array_like
            Expansion coefficients in IR
        """
        g_dlr = np.ascontiguousarray(g_dlr)
        if g_dlr.shape[axis] != self.size:
            raise ValueError(f"Input array has wrong size along dimension {axis}")
        output_dims = np.asarray(g_dlr.shape, dtype=np.int32)
        output_dims[axis] = self.basis.size
        output = np.zeros(output_dims, dtype=g_dlr.dtype)
        ndim = len(g_dlr.shape)
        input_dims = np.asarray(g_dlr.shape, dtype=np.int32)
        target_dim = axis
        order = SPIR_ORDER_ROW_MAJOR

        if g_dlr.dtype.kind == 'f':
            ret = _lib.spir_dlr2ir_dd(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                target_dim,
                g_dlr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        elif g_dlr.dtype.kind == 'c':
            ret = _lib.spir_dlr2ir_zz(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                target_dim,
                # TODO: use complex data
                g_dlr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        else:
            raise ValueError(f"Unsupported dtype: {g_dlr.dtype}")
        if ret != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to convert DLR to IR: {ret}")
        return output

    def default_tau_sampling_points(self, **kwargs):
        return self._basis.default_tau_sampling_points(**kwargs)

    def default_matsubara_sampling_points(self, **kwargs):
        return self._basis.default_matsubara_sampling_points(**kwargs)

    @property
    def is_well_conditioned(self):
        return False
