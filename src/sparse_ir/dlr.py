# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Discrete Lehmann Representation (DLR) functionality for SparseIR.

This module implements DLR basis with poles at the roots of the first
discarded real-frequency IR basis function V_L, providing an alternative
representation that can be more efficient for certain calculations.
"""

import ctypes
import numpy as np
from .abstract import AbstractBasis
from .basis import FiniteTempBasis
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
    r"""Discrete Lehmann representation (DLR), with poles at the roots of V_L.

    This class implements a variant of the discrete Lehmann representation
    (`DLR`_).  Instead of a truncated singular value expansion of the analytic
    continuation kernel ``K`` like the IR, the discrete Lehmann representation
    is based on a "sketching" of ``K``.  The resulting basis is a
    linear combination of discrete set of poles :math:`\bar\omega_p` on the
    real-frequency axis: for both statistics,

    .. math::

        \rho(\omega) = \sum_p c_p \delta(\omega - \bar\omega_p), \qquad
        G(\tau) = \sum_p c_p u_p(\tau), \qquad
        u_p(\tau) = -K(\tau, \bar\omega_p)
        = -\frac{e^{-\tau\bar\omega_p}}{1 + e^{-\beta\bar\omega_p}},

    with :math:`\rho = A` for fermions and :math:`\rho = A/\tanh(\beta\omega/2)`
    for bosons (see :class:`~sparse_ir.FiniteTempBasis`), continued to the
    imaginary-frequency axis as
    :math:`G(\mathrm{i}\nu) = \sum_p c_p \hat u_p(\mathrm{i}\nu)` with

    .. math::

        \hat u_p(\mathrm{i}\nu) = \frac{1}{\mathrm{i}\nu - \bar\omega_p}
        \ \text{(fermions)}, \qquad
        \hat u_p(\mathrm{i}\nu)
        = \frac{\tanh(\beta\bar\omega_p/2)}{\mathrm{i}\nu - \bar\omega_p}
        \ \text{(bosons)}.

    For bosons, the spectral function is thus
    :math:`A(\omega) = \sum_p c_p \tanh(\beta\bar\omega_p/2)\,
    \delta(\omega - \bar\omega_p)`.  The DLR coefficients :math:`c_p` and the
    IR coefficients are related by
    :math:`G_l = -S_l \sum_p V_l(\bar\omega_p)\, c_p` (:py:meth:`to_IR`).

    Warning:
        The poles on the real-frequency axis selected for the DLR are based
        on a rank-revealing decomposition, which offers accuracy guarantees.
        Here, we instead select the pole locations based on the roots of V_L,
        the first IR basis function on the real axis beyond the basis, which
        is a heuristic.  We do not expect that difference to matter, but
        please don't blame the DLR authors if we were wrong :-)

    .. _DLR: https://doi.org/10.1103/PhysRevB.105.235115
    """

    def __init__(self, basis: AbstractBasis, poles=None):
        """
        Parameters
        ----------
        basis : FiniteTempBasis
            IR basis on which the DLR is built
        poles : array_like, optional
            Pole positions in ``[-wmax, wmax]``.  If None, use
            ``basis.default_omega_sampling_points()``, the L roots of V_L.
        """
        if not isinstance(basis, FiniteTempBasis):
            raise TypeError("DiscreteLehmannRepresentation is built on a "
                            f"FiniteTempBasis, got {type(basis).__name__}")
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
        # The C library panics on a pole outside the frequency window
        # (SpM-lab/sparse-ir-rs#266); reject it here with the value.
        _util.check_domain(poles, -basis.wmax, basis.wmax, "poles")
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

        These are the *DLR* basis functions, i.e. ``u[p](tau)`` is minus the
        logistic kernel at the ``p``-th pole,
        :math:`u_p(\tau) = -K(\tau, \bar\omega_p) =
        -e^{-\tau\bar\omega_p}/(1 + e^{-\beta\bar\omega_p})` for both
        statistics, so that::

            gtau == g_dlr @ dlr.u(tau)

        holds for DLR coefficients ``g_dlr``.  They are **not** the basis
        functions of the underlying IR basis.

        ``tau`` may lie anywhere in ``[-beta, beta]``, with the extension and
        endpoint rule of :py:attr:`FiniteTempBasis.u
        <sparse_ir.FiniteTempBasis.u>`: :math:`u_p(\tau) = (-1)^\zeta
        u_p(\tau + \beta)` for negative times, ``+0.0`` is 0⁺, ``beta`` is
        β⁻, ``-0.0`` is 0⁻ and ``-beta`` is (-β)⁺.

        They are not piecewise polynomials: ``deriv`` and ``overlap`` are not
        supported by the C library for them and raise ``RuntimeError``.
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

        ``uhat[p](n)`` is the Fourier transform of :py:attr:`u` at
        ν = nπ/β: :math:`\hat u_p(\mathrm{i}\nu) = 1/(\mathrm{i}\nu -
        \bar\omega_p)` for fermions and :math:`\tanh(\beta\bar\omega_p/2)/
        (\mathrm{i}\nu - \bar\omega_p)` for bosons, so that::

            giv == g_dlr @ dlr.uhat(n)

        holds for DLR coefficients ``g_dlr``.  They are **not** the Matsubara
        basis functions of the underlying IR basis.
        """
        if self._uhat is None:
            self._uhat = PiecewiseLegendrePolyFTVector(
                FunctionSetFT(basis_get_uhat(self._ptr),
                              zeta=1 if self.statistics == 'F' else 0))
        return self._uhat

    @property
    def statistics(self):
        """Quantum statistic of the underlying basis ('F' or 'B')"""
        return self._basis.statistics

    @property
    def sampling_points(self):
        """The poles of the DLR on the real-frequency axis.

        By default the roots of V_L, ``basis.default_omega_sampling_points()``.
        """
        return self._poles

    @property
    def shape(self): return self.size,

    @property
    def size(self):
        """Number of poles"""
        return len(self._poles)

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
        """All ones, since the DLR basis functions are not ordered by significance"""
        return np.ones(self.shape)

    @property
    def accuracy(self):
        """Accuracy of the underlying IR basis (FiniteTempBasis.accuracy)"""
        return self._basis.accuracy

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """From IR to DLR

        Convert expansion coefficients from IR basis to DLR basis: the
        inverse of :py:meth:`to_IR`, which finds the c_p with
        G_l = -S_l Σ_p V_l(ω̄_p) c_p.

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

        Convert expansion coefficients from DLR basis to IR basis:
        G_l = -S_l Σ_p V_l(ω̄_p) c_p for the DLR coefficients c_p.

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
        """Default tau sampling points of the underlying IR basis"""
        return self._basis.default_tau_sampling_points(**kwargs)

    def default_matsubara_sampling_points(self, **kwargs):
        """Default Matsubara sampling points of the underlying IR basis"""
        return self._basis.default_matsubara_sampling_points(**kwargs)

    @property
    def is_well_conditioned(self):
        """False, since sampling in the DLR basis is not expected to be well-conditioned"""
        return False
