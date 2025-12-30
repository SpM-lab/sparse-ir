# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger,
# Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
SVE (Singular Value Expansion) functionality for SparseIR.

This module provides Python wrappers for SVE computation and results.
"""
import numpy as np

from pylibsparseir.constants import (
    SPIR_TWORK_FLOAT64,
    SPIR_TWORK_FLOAT64X2,
)
from pylibsparseir.core import (
    _lib,
    sve_result_new,
    sve_result_get_svals,
    sve_result_get_size,
)
from .abstract import AbstractKernel
from .kernel import LogisticKernel, RegularizedBoseKernel


def _resolve_work_dtype(work_dtype):
    if work_dtype is None:
        return None

    if isinstance(work_dtype, str):
        work_dtype = work_dtype.lower()
        if work_dtype in {"float64", "double"}:
            return SPIR_TWORK_FLOAT64
        if work_dtype in {"float64x2", "ddouble"}:
            return SPIR_TWORK_FLOAT64X2
        raise TypeError(f"unexpected work_dtype string {work_dtype!r}")

    dtype = np.dtype(work_dtype)
    if dtype == np.float64:
        return SPIR_TWORK_FLOAT64
    if dtype.itemsize > np.dtype(np.float64).itemsize:
        return SPIR_TWORK_FLOAT64X2
    raise TypeError(f"unsupported work_dtype {dtype}")


class SVEResult:
    """
    Result of a singular value expansion (SVE).

    Contains the singular values and basis functions resulting from
    the SVE of an integral kernel.
    """

    def __init__(
        self,
        kernel: AbstractKernel,
        eps: float,
        cutoff: float = -1,
        n_sv: int = -1,
        work_dtype=None,
    ):
        """
        Compute SVE of the given kernel.

        Parameters
        ----------
        kernel : LogisticKernel or RegularizedBoseKernel
            Kernel to compute SVE for
        eps : float
            Desired accuracy of the expansion
        cutoff : float
            Relative cutoff for the singular values.
        n_sv : int
            Maximum basis size. If given, only at most the ``n_sv`` most
            significant singular values and associated singular functions are
            returned.
        """
        if not isinstance(kernel, (LogisticKernel, RegularizedBoseKernel)):
            raise TypeError(
                "kernel must be LogisticKernel or RegularizedBoseKernel"
            )

        self._kernel = kernel  # Store kernel for later use
        self._eps = eps
        self._cutoff = cutoff
        self._n_sv = n_sv

        twork = _resolve_work_dtype(work_dtype)
        self._ptr = sve_result_new(
            kernel._ptr,
            eps,
            cutoff=cutoff,
            lmax=n_sv,
            Twork=twork,
        )

    def __len__(self):
        return sve_result_get_size(self._ptr)

    @property
    def s(self):
        return sve_result_get_svals(self._ptr)

    def __del__(self):
        """Clean up SVE resources."""
        if hasattr(self, '_ptr') and self._ptr:
            _lib.spir_sve_result_release(self._ptr)


def compute(
    kernel,
    eps=np.finfo(np.float64).eps,
    n_sv=-1,
    work_dtype=None,
):
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
        K (kernel.AbstractKernel):
            Integral kernel to take SVE from
        eps (float):
            Relative truncation threshold for the singular values,
            defaulting to the machine epsilon (2.2e-16)
        n_sv (int):
            Maximum basis size. If given, only at most the ``n_sv`` most
            significant singular values and associated singular functions are
            returned. Defaults to -1, which means all singular values are
            returned.
        work_dtype (dtype-like or str, optional):
            Working data type used during the SVE / SVD computation. Accepts
            ``numpy.float64``, ``float``, or strings such as ``"float64"`` or
            ``"float64x2"``. Defaults to ``float64x2`` for maximal precision.

    Returns:
        An ``SVEResult`` containing the truncated singular value expansion.
    """

    if eps is None:
        eps = np.finfo(np.float64).eps
    return SVEResult(
        kernel,
        eps=eps,
        cutoff=-1,
        n_sv=n_sv,
        work_dtype=work_dtype,
    )


# Backward compatibility
compute_sve = compute
