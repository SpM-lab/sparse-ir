# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Kernel classes for SparseIR.

This module provides Python wrappers for kernel objects from the C library.
"""

import ctypes
import warnings
from ctypes import c_int, c_double, byref
import numpy as np

from pylibsparseir.core import _lib
from pylibsparseir.core import logistic_kernel_new, reg_bose_kernel_new
from pylibsparseir.constants import COMPUTATION_SUCCESS
from .abstract import AbstractKernel


def kernel_domain(kernel: AbstractKernel):
    """Get the domain boundaries of a kernel.

    Returns ``(xmin, xmax, ymin, ymax)`` of the dimensionless kernel
    K(x, y): ``(-1, 1, -1, 1)`` for both kernels.
    """
    xmin = c_double()
    xmax = c_double()
    ymin = c_double()
    ymax = c_double()

    status = _lib.spir_kernel_get_domain(
        kernel._ptr, byref(xmin), byref(xmax), byref(ymin), byref(ymax)
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get kernel domain: {status}")

    return xmin.value, xmax.value, ymin.value, ymax.value

def _check_lambda(lambda_):
    """Return ``lambda_`` as float; the kernel cutoff must be positive and finite."""
    lambda_ = float(lambda_)
    if not (np.isfinite(lambda_) and lambda_ > 0):
        raise ValueError(
            f"kernel cutoff lambda_ must be positive and finite, got {lambda_!r}")
    return lambda_


class LogisticKernel(AbstractKernel):
    r"""Fermionic/bosonic analytical continuation kernel.

    The logistic kernel is the default kernel for both statistics.  In
    physical units, for imaginary time τ ∈ [0, β] and real frequency
    ω ∈ [-ωmax, ωmax], it reads

    .. math::  K(\tau, \omega) = \frac{e^{-\tau\omega}}{1 + e^{-\beta\omega}}.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = ω/ωmax``,
    the integral kernel is a function on ``[-1, 1] x [-1, 1]``:

    .. math::  K(x, y) = \frac{\exp(-\Lambda y(x + 1)/2)}{1 + \exp(-\Lambda y)}

    with Λ = β ωmax; both forms take the same values.  A fermionic Green's
    function with spectral function :math:`A(\omega)` is
    :math:`G(\tau) = -\int d\omega\, K(\tau, \omega) A(\omega)`.
    One can model the τ dependence of a bosonic correlation function with the
    same kernel as follows:

    .. math::

        G(\tau) = -\int d\omega\,
            \frac{e^{-\tau\omega}}{1 - e^{-\beta\omega}} A(\omega)
            = -\int d\omega\, K(\tau, \omega)
            \frac{A(\omega)}{\tanh(\beta\omega/2)},

    i.e., a rescaling of the spectral function with the weight function:

    .. math::  w(\omega) = \frac1{\tanh(\beta\omega/2)} = \frac1{\tanh(\Lambda y/2)}.

    Parameters
    ----------
    lambda\_ : float
        Kernel cutoff Λ = β * ωmax
    """

    def __init__(self, lambda_):
        """Initialize logistic kernel with cutoff ``lambda_``."""
        self._lambda = _check_lambda(lambda_)
        self._ptr = logistic_kernel_new(self._lambda)

    @property
    def lambda_(self):
        """Kernel cutoff."""
        return self._lambda

    def __del__(self):
        """Clean up kernel resources."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_kernel_release(self._ptr)


class RegularizedBoseKernel(AbstractKernel):
    r"""Regularized bosonic analytical continuation kernel.

    .. warning::

        Deprecated: use :class:`LogisticKernel`, the default kernel for both
        statistics.  ``RegularizedBoseKernel`` will be removed in a future
        release.  For ``wmax != 1``, libsparseir releases without the fix of
        SpM-lab/sparse-ir-rs#273 scale the singular values of its bases by
        ``wmax**-1`` instead of ``wmax**+1``.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = ω/ωmax``, the bosonic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``:

    .. math::

        K(x, y) = \frac{y \exp(-\Lambda y(x + 1)/2)}{1 - \exp(-\Lambda y)}

    In physical units it is :math:`K(\tau, \omega) = \omega_\mathrm{max}
    K(x, y) = \omega e^{-\tau\omega} / (1 - e^{-\beta\omega})`, which acts on
    :math:`A(\omega)/\omega` for the spectral function :math:`A(\omega)`
    (N. Chikano et al., Computer Physics Communications 240, 181 (2019),
    Eqs. (1)-(3)).
    Care has to be taken in evaluating this expression around ``y == 0``.

    Parameters
    ----------
    lambda\_ : float
        Kernel cutoff Λ = β * ωmax
    """

    def __init__(self, lambda_):
        """Initialize regularized bosonic kernel with cutoff ``lambda_``."""
        warnings.warn(
            "RegularizedBoseKernel is deprecated and will be removed in a "
            "future release; use LogisticKernel, the default kernel for both "
            "statistics (https://github.com/SpM-lab/sparse-ir-rs/issues/273)",
            DeprecationWarning, stacklevel=2)
        self._lambda = _check_lambda(lambda_)
        self._ptr = reg_bose_kernel_new(self._lambda)

    @property
    def lambda_(self):
        """Kernel cutoff."""
        return self._lambda

    def __del__(self):
        """Clean up kernel resources."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_kernel_release(self._ptr)