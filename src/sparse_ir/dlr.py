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
from pylibsparseir.core import (
    _lib,
    COMPUTATION_SUCCESS,
    get_default_blas_backend,
    c_double_complex,
    basis_get_u,
    basis_get_uhat,
)
from pylibsparseir.constants import SPIR_ORDER_ROW_MAJOR
from . import _util
from .poly import (
    FunctionSet,
    FunctionSetFT,
    PiecewiseLegendrePolyVector,
    PiecewiseLegendrePolyFTVector,
)

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
        # Normalize first, then take the pointer from the *normalized* object.
        # Taking it from the caller's array instead silently hands C the
        # buffer of a non-contiguous or non-float64 array.
        poles = _util.as_boundary_real(poles, "poles")
        if poles.ndim != 1:
            raise ValueError(
                f"poles must be one-dimensional, got shape {poles.shape}")
        if poles.size == 0:
            raise ValueError("poles must not be empty")
        self._basis = basis
        self._poles = poles
        self._u = None
        self._uhat = None
        self._backend = get_default_blas_backend()
        self._ptr = _lib.spir_dlr_new_with_poles(
            basis._ptr,
            poles.size,
            poles.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            status,
        )
        if status.value != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to create DLR basis: {status.value}")
        if not self._ptr:
            raise RuntimeError("Failed to create DLR basis: null handle")

    @property
    def u(self):
        r"""DLR basis functions on the imaginary-time axis.

        These are the *DLR* basis functions, i.e. ``u[i](tau)`` is the
        imaginary-time kernel evaluated at the ``i``-th pole, so that::

            gtau == g_dlr @ dlr.u(tau)

        holds for DLR coefficients ``g_dlr``.  They are **not** the basis
        functions of the underlying IR basis.
        """
        if self._u is None:
            beta = self._basis.beta
            self._u = PiecewiseLegendrePolyVector(
                FunctionSet(basis_get_u(self._ptr)),
                -beta, beta, beta, default_overlap_range=(0, beta))
        return self._u

    @property
    def uhat(self):
        r"""DLR basis functions on the reduced Matsubara frequency axis.

        ``uhat[i](n)`` is the Fourier transform of :py:attr:`u`, so that::

            giv == g_dlr @ dlr.uhat(n)

        holds for DLR coefficients ``g_dlr``.  They are **not** the Matsubara
        basis functions of the underlying IR basis.
        """
        if self._uhat is None:
            self._uhat = PiecewiseLegendrePolyFTVector(
                FunctionSetFT(basis_get_uhat(self._ptr)))
        return self._uhat

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
        gl = np.asarray(gl)
        if gl.ndim == 0:
            raise ValueError("IR coefficients must be at least one-dimensional")
        axis = _util.normalize_axis(axis, gl.ndim)
        if gl.shape[axis] != self.basis.size:
            raise ValueError(
                f"IR coefficients have length {gl.shape[axis]} along axis "
                f"{axis}, expected {self.basis.size}")

        output_dims = list(gl.shape)
        output_dims[axis] = self.size

        ndim = gl.ndim
        input_dims = np.ascontiguousarray(gl.shape, dtype=np.int32)
        target_dim = axis
        order = SPIR_ORDER_ROW_MAJOR

        if gl.dtype.kind != 'c':
            gl = _util.as_boundary_real(gl, "IR coefficients")
            output = np.zeros(output_dims, dtype=np.float64)
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
        else:
            gl = _util.as_boundary_complex(gl, "IR coefficients")
            output_c = np.zeros(output_dims, dtype=c_double_complex)
            ret = _lib.spir_ir2dlr_zz(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                target_dim,
                gl.ctypes.data_as(ctypes.POINTER(c_double_complex)),
                output_c.ctypes.data_as(ctypes.POINTER(c_double_complex)),
            )
            output = output_c['real'] + 1j * output_c['imag']
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
        g_dlr = np.asarray(g_dlr)
        if g_dlr.ndim == 0:
            raise ValueError("DLR coefficients must be at least one-dimensional")
        axis = _util.normalize_axis(axis, g_dlr.ndim)
        if g_dlr.shape[axis] != self.size:
            raise ValueError(
                f"DLR coefficients have length {g_dlr.shape[axis]} along axis "
                f"{axis}, expected {self.size}")
        output_dims = list(g_dlr.shape)
        output_dims[axis] = self.basis.size
        ndim = g_dlr.ndim
        input_dims = np.ascontiguousarray(g_dlr.shape, dtype=np.int32)
        target_dim = axis
        order = SPIR_ORDER_ROW_MAJOR

        if g_dlr.dtype.kind != 'c':
            g_dlr = _util.as_boundary_real(g_dlr, "DLR coefficients")
            output = np.zeros(output_dims, dtype=np.float64)
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
        else:
            g_dlr = _util.as_boundary_complex(g_dlr, "DLR coefficients")
            output_c = np.zeros(output_dims, dtype=c_double_complex)
            ret = _lib.spir_dlr2ir_zz(
                self._ptr,
                self._backend,
                order,
                ndim,
                input_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                target_dim,
                g_dlr.ctypes.data_as(ctypes.POINTER(c_double_complex)),
                output_c.ctypes.data_as(ctypes.POINTER(c_double_complex)),
            )
            output = output_c['real'] + 1j * output_c['imag']
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
