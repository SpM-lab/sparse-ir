"""
SVE (Singular Value Expansion) functionality for SparseIR.

This module provides Python wrappers for SVE computation and results.
"""

import ctypes
from ctypes import c_int, byref

from pylibsparseir.core import _lib, sve_result_new, sve_result_get_svals, sve_result_get_size
from pylibsparseir.constants import COMPUTATION_SUCCESS, SPIR_ORDER_ROW_MAJOR
from .abstract import AbstractKernel
from .kernel import LogisticKernel, RegularizedBoseKernel

class SVEResult:
    """
    Result of a singular value expansion (SVE).

    Contains the singular values and basis functions resulting from
    the SVE of an integral kernel.
    """

    def __init__(self, kernel: AbstractKernel, epsilon: float):
        """
        Compute SVE of the given kernel.

        Parameters
        ----------
        kernel : LogisticKernel or RegularizedBoseKernel
            Kernel to compute SVE for
        epsilon : float
            Desired accuracy of the expansion
        """
        if not isinstance(kernel, (LogisticKernel, RegularizedBoseKernel)):
            raise TypeError("kernel must be LogisticKernel or RegularizedBoseKernel")

        self._kernel = kernel  # Store kernel for later use
        self._epsilon = epsilon

        self._ptr = sve_result_new(kernel._ptr, epsilon)

    def __len__(self):
        return sve_result_get_size(self._ptr)

    @property
    def s(self):
        return sve_result_get_svals(self._ptr)

    def __del__(self):
        """Clean up SVE resources."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_sve_result_release(self._ptr)


def compute(kernel, epsilon):
    """Perform truncated singular value expansion of a kernel.

    Perform a truncated singular value expansion (SVE) of an integral
    kernel ``K : [xmin, xmax] x [ymin, ymax] -> R``::

        K(x, y) == sum(s[l] * u[l](x) * v[l](y) for l in (0, 1, 2, ...)),

    where ``s[l]`` are the singular values, which are ordered in non-increasing
    fashion, ``u[l](x)`` are the left singular functions, which form an
    orthonormal system on ``[xmin, xmax]``, and ``v[l](y)`` are the right
    singular functions, which form an orthonormal system on ``[ymin, ymax]``.

    The SVE is mapped onto the singular value decomposition (SVD) of a matrix
    by expanding the kernel in piecewise Legendre polynomials (by default by
    using a collocation).

    Arguments:
        kernel (kernel.AbstractKernel):
            Integral kernel to take SVE from
        epsilon (float):
            Accuracy target for the basis: attempt to have singular values down
            to a relative magnitude of ``epsilon``, and have each singular value
            and singular vector be accurate to ``epsilon``.

    Returns:
        An ``SVEResult`` containing the truncated singular value expansion.
    """
    return SVEResult(kernel, epsilon)