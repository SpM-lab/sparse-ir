# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Kernel classes for SparseIR.

This module provides Python wrappers for kernel objects from the C library.
"""

import ctypes
from ctypes import c_int, c_double, byref
import numpy as np

from pylibsparseir.core import _lib
from pylibsparseir.core import logistic_kernel_new, reg_bose_kernel_new
from pylibsparseir.constants import COMPUTATION_SUCCESS
from .abstract import AbstractKernel


def kernel_domain(kernel: AbstractKernel):
    """Get the domain boundaries of a kernel."""
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

class LogisticKernel(AbstractKernel):
    r"""Fermionic/bosonic analytical continuation kernel.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``,
    the integral kernel is a function on ``[-1, 1] x [-1, 1]``:

    .. math::  K(x, y) = \frac{\exp(-\Lambda y(x + 1)/2)}{1 + \exp(-\Lambda y)}

    LogisticKernel is a fermionic analytic continuation kernel.
    Nevertheless, one can model the τ dependence of
    a bosonic correlation function as follows:

    .. math::

        \int \frac{\exp(-\Lambda y(x + 1)/2)}{1 - \exp(-\Lambda y)} \rho(y) dy
            = \int K(x, y) \frac{\rho'(y)}{\tanh(\Lambda y/2)} dy

    i.e., a rescaling of the spectral function with the weight function:

    .. math::  w(y) = \frac1{\tanh(\Lambda y/2)}.

    Parameters
    ----------
    lambda_ : float
        Kernel cutoff Λ = β * ωmax
    """

    def __init__(self, lambda_):
        """Initialize logistic kernel with cutoff lambda."""
        self._lambda = float(lambda_)
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

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``, the fermionic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``:

    .. math::

        K(x, y) = \frac{y \exp(-\Lambda y(x + 1)/2)}{\exp(-\Lambda y) - 1}

    Care has to be taken in evaluating this expression around ``y == 0``.

    Parameters
    ----------
    lambda_ : float
        Kernel cutoff Λ = β * ωmax
    """

    def __init__(self, lambda_):
        """Initialize regularized bosonic kernel with cutoff lambda."""
        self._lambda = float(lambda_)
        self._ptr = reg_bose_kernel_new(self._lambda)

    @property
    def lambda_(self):
        """Kernel cutoff."""
        return self._lambda

    def __del__(self):
        """Clean up kernel resources."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_kernel_release(self._ptr)