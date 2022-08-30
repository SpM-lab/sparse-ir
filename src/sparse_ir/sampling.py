# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn

from . import _util


class AbstractSampling:
    """Base class for sparse sampling.

    Encodes the "basis transformation" of a propagator from the truncated IR
    basis coefficients ``G_ir[l]`` to time/frequency sampled on sparse points
    ``G(x[i])`` together with its inverse, a least squares fit::

             ________________                   ___________________
            |                |    evaluate     |                   |
            |     Basis      |---------------->|     Value on      |
            |  coefficients  |<----------------|  sampling points  |
            |________________|      fit        |___________________|

    """
    def evaluate(self, al, axis=None):
        """Evaluate the basis coefficients at the sparse sampling points"""
        return self.matrix.matmul(al, axis)

    def fit(self, ax, axis=None):
        """Fit basis coefficients from the sparse sampling points"""
        return self.matrix.lstsq(ax, axis)

    @property
    def cond(self):
        """Condition number of the fitting problem"""
        return self.matrix.cond

    @property
    def sampling_points(self):
        """Set of sampling points"""
        raise NotImplementedError()

    @property
    def matrix(self):
        """Evaluation matrix is decomposed form"""
        raise NotImplementedError()

    @property
    def basis(self):
        """Basis instance"""
        raise NotImplementedError()


class TauSampling(AbstractSampling):
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.
    """
    def __init__(self, basis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_tau_sampling_points()
        else:
            sampling_points = np.asarray(sampling_points)
            if sampling_points.ndim != 1:
                raise ValueError("sampling points must be vector")

        matrix = basis.u(sampling_points).T
        self._basis = basis
        self._sampling_points = sampling_points
        self._matrix = DecomposedMatrix(matrix)

        if basis.is_well_conditioned and self._matrix.cond > 1e8:
            warn(f"Sampling matrix is poorly conditioned "
                 f"(kappa = {self._matrix.cond:.2g})", ConditioningWarning)

    @property
    def basis(self): return self._basis

    @property
    def sampling_points(self): return self._sampling_points

    @property
    def matrix(self): return self._matrix

    @property
    def tau(self):
        """Sampling points in (reduced) imaginary time"""
        return self._sampling_points


class MatsubaraSampling(AbstractSampling):
    """Sparse sampling in Matsubara frequencies.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary frequencies.
    """
    def __init__(self, basis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_matsubara_sampling_points()
        else:
            sampling_points = _util.check_reduced_matsubara(sampling_points)
            if sampling_points.ndim != 1:
                raise ValueError("sampling points must be vector")

        matrix = basis.uhat(sampling_points).T
        self._basis = basis
        self._sampling_points = sampling_points
        self._matrix = DecomposedMatrix(matrix)

        if basis.is_well_conditioned and self._matrix.cond > 1e8:
            warn(f"Sampling matrix is poorly conditioned "
                 f"(kappa = {self._matrix.cond:.2g})", ConditioningWarning)

    @property
    def basis(self): return self._basis

    @property
    def sampling_points(self): return self._sampling_points

    @property
    def matrix(self): return self._matrix

    @property
    def wn(self):
        """Sampling points as (reduced) Matsubara frequencies"""
        return self._sampling_points


class DecomposedMatrix:
    """Matrix in SVD decomposed form for fast and accurate fitting.

    Stores a matrix ``A`` together with its thin SVD form::

        A == (u * s) @ vH.

    This allows for fast and accurate least squares fits using ``A.lstsq(x)``.
    """
    def __init__(self, a, svd_result=None):
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be of matrix form")
        if svd_result is None:
            u, s, vH = np.linalg.svd(a, full_matrices=False)
        else:
            u, s, vH = _util.check_svd_result(svd_result, a.shape)

        # Remove singular values which are exactly zero
        where = s.astype(bool)
        if not where.all():
            u, s, vH = u[:, where], s[where], vH[where]

        self._a = a
        self._uH = np.array(u.conj().T)
        self._s = s
        self._v = np.array(vH.conj().T)

    def __matmul__(self, x):
        """Matrix-matrix multiplication."""
        return self._a @ x

    def matmul(self, x, axis=None):
        """Compute ``A @ x`` (optionally along specified axis of x)"""
        return _matop_along_axis(self._a.__matmul__, x, axis)

    def _lstsq(self, x):
        r = self._uH @ x
        r = r / (self._s[:, None] if r.ndim > 1 else self._s)
        return self._v @ r

    def lstsq(self, x, axis=None):
        """Return ``y`` such that ``np.linalg.norm(A @ y - x)`` is minimal"""
        return _matop_along_axis(self._lstsq, x, axis)

    def __array__(self, dtype=""):
        """Convert to numpy array."""
        return self._a if dtype == "" else self._a.astype(dtype)

    @property
    def a(self):
        """Full matrix"""
        return self._a

    @property
    def u(self):
        """Left singular vectors, aranged column-wise"""
        return self._uH.conj().T

    @property
    def s(self):
        """Most significant, nonzero singular values"""
        return self._s

    @property
    def vH(self):
        """Right singular vectors, transposed"""
        return self._v.conj().T

    @property
    def cond(self):
        """Condition number of matrix"""
        return self._s[0] / self._s[-1]


class ConditioningWarning(RuntimeWarning):
    pass


def _matop_along_axis(op, x, axis=None):
    if axis is None:
        return op(x)

    x = np.asarray(x)
    target_axis = max(x.ndim - 2, 0)
    x = np.moveaxis(x, axis, target_axis)
    r = op(x)
    return np.moveaxis(r, target_axis, axis)
