# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger,
# Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
SVE (Singular Value Expansion) functionality for SparseIR.

This module provides Python wrappers for SVE computation and results.
"""
import numbers

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
    r"""
    Result of a singular value expansion (SVE).

    Holds the SVE of the dimensionless kernel on ``[-1, 1] x [-1, 1]``,
    :math:`K(x, y) = \sum_l s_l u_l(x) v_l(y)`, and exposes only its
    singular values :py:attr:`s` (the dimensionless :math:`s_l`) and their
    number, ``len(sve)``; the singular functions are not exposed.  A
    :class:`~sparse_ir.FiniteTempBasis` built from it holds them in physical
    units, with :math:`S_l = \sqrt{\beta\omega_\mathrm{max}/2}\, s_l` for
    :class:`~sparse_ir.LogisticKernel`.

    The result is not truncated at ``eps``: it holds every singular value
    resolved in the working precision, so ``len(sve)`` does not depend on
    ``eps``.  The truncation to S_l/S_0 >= eps happens in FiniteTempBasis.
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
            Kernel to compute SVE for; other kernels raise TypeError.
        eps : float
            Accuracy target passed to libsparseir.  It does not truncate the
            result (see above).
        cutoff : float
            Ignored: pylibsparseir does not pass it to libsparseir.
        n_sv : int
            Passed to libsparseir as ``lmax``, which libsparseir currently
            ignores.  To limit the basis size, use the ``max_size`` argument
            of FiniteTempBasis.
        work_dtype : dtype-like or str, optional
            Working precision: ``float64`` (``numpy.float64``, ``float``,
            ``"float64"``, ``"double"``) or ``float64x2`` (``"float64x2"``,
            ``"ddouble"``, or a NumPy dtype wider than 8 bytes).  Defaults to
            ``float64x2``.  It sets how many singular values are resolved.
        """
        if not isinstance(kernel, (LogisticKernel, RegularizedBoseKernel)):
            raise TypeError(
                "kernel must be LogisticKernel or RegularizedBoseKernel"
            )
        if not isinstance(eps, numbers.Real):
            raise TypeError(f"accuracy eps must be a real number, got {eps!r}")
        if not (np.isfinite(eps) and eps > 0):
            raise ValueError(
                f"accuracy eps must be positive and finite, got {eps!r}")

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
        """Dimensionless singular values s_l of the kernel, non-increasing"""
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
    """Perform singular value expansion of a kernel.

    Perform a singular value expansion (SVE) of the dimensionless integral
    kernel ``K : [-1, 1] x [-1, 1] -> R``::

        K(x, y) == sum(s[l] * u[l](x) * v[l](y) for l in (0, 1, 2, ...)),

    where ``s[l]`` are the singular values, which are ordered in non-increasing
    fashion, ``u[l](x)`` are the left singular functions, which form an
    orthonormal system on ``[-1, 1]``, and ``v[l](y)`` are the right
    singular functions, which form an orthonormal system on ``[-1, 1]``.
    The returned :class:`SVEResult` exposes only the ``s[l]``.

    The SVE is mapped onto the singular value decomposition (SVD) of a matrix
    by expanding the kernel in piecewise Legendre polynomials (by default by
    using a collocation).

    Arguments:
        kernel (LogisticKernel or RegularizedBoseKernel):
            Integral kernel to take SVE from
        eps (float):
            Accuracy target, defaulting to the machine epsilon (2.2e-16).
            It does not truncate the result (see :class:`SVEResult`).
        n_sv (int):
            Currently has no effect (see :class:`SVEResult`).  Defaults to -1.
        work_dtype (dtype-like or str, optional):
            Working data type used during the SVE / SVD computation. Accepts
            ``numpy.float64``, ``float``, or strings such as ``"float64"`` or
            ``"float64x2"``. Defaults to ``float64x2`` for maximal precision.

    Returns:
        An ``SVEResult`` containing the singular value expansion.
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
