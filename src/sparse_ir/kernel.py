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

    status = _lib.spir_kernel_domain(
        kernel._ptr, byref(xmin), byref(xmax), byref(ymin), byref(ymax)
    )
    if status != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Failed to get kernel domain: {status}")

    return xmin.value, xmax.value, ymin.value, ymax.value

class LogisticKernel(AbstractKernel):
    """
    Fermionic/logistic imaginary-time kernel.

    This kernel treats a fermionic spectral function at finite temperature.
    The definition is:

        K(τ, ω) = exp(-τ ω) / (1 + exp(-β ω))

    with τ ∈ [0, β] and ω ∈ [-ωmax, ωmax].

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
    """
    Bosonic imaginary-time kernel.

    This kernel treats a bosonic spectral function at finite temperature.
    The definition is:

        K(τ, ω) = ω exp(-τ ω) / (1 - exp(-β ω))

    with τ ∈ [0, β] and ω ∈ [-ωmax, ωmax]. The kernel is regularized
    at ω = 0 to avoid the singularity.

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