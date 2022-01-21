# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn


class SamplingBase:
    """Base class for sparse sampling.

    Encodes the "basis transformation" of a propagator from the truncated IR
    basis coefficients ``G_ir[l]`` to time/frequency sampled on sparse points
    ``G(x[i])`` together with its inverse, a least squares fit::

             ________________                   ___________________
            |                |    evaluate     |                   |
            |     Basis      |---------------->|     Value on      |
            |  coefficients  |<----------------|  sampling points  |
            |________________|      fit        |___________________|


    Attributes:
        basis: IR Basis instance
        matrix: Evaluation matrix is decomposed form
        sampling_points: Set of sampling points
    """
    def __init__(self, basis, sampling_points):
        self.sampling_points = np.array(sampling_points)
        self.basis = basis
        self.matrix = DecomposedMatrix(
                        self.__class__.eval_matrix(basis, self.sampling_points))

        # Check conditioning
        self.cond = self.matrix.s[0] / self.matrix.s[-1]
        if self.cond > 1e8:
            warn("Sampling matrix is poorly conditioned (cond = %.2g)"
                 % self.cond, ConditioningWarning)

    def evaluate(self, al, axis=None):
        """Evaluate the basis coefficients at the sparse sampling points"""
        return self.matrix.matmul(al, axis)

    def fit(self, ax, axis=None):
        """Fit basis coefficients from the sparse sampling points"""
        return self.matrix.lstsq(ax, axis)

    @classmethod
    def eval_matrix(cls, basis, x):
        """Return evaluation matrix from coefficients to sampling points"""
        raise NotImplementedError()


class TauSampling(SamplingBase):
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.
    """
    def __init__(self, basis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_tau_sampling_points()
        super().__init__(basis, sampling_points)

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.u(x).T

    @property
    def tau(self):
        """Sampling points in (reduced) imaginary time"""
        return self.sampling_points


class MatsubaraSampling(SamplingBase):
    """Sparse sampling in Matsubara frequencies.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary frequencies.
    """
    def __init__(self, basis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_matsubara_sampling_points()
        super().__init__(basis, sampling_points)

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.uhat(x).T

    @property
    def wn(self):
        """Sampling points as (reduced) Matsubara frequencies"""
        return self.sampling_points


class DecomposedMatrix:
    """Matrix in SVD decomposed form for fast and accurate fitting.

    Stores a matrix ``A`` together with its thin SVD form::

        A == (u * s) @ vH.

    This allows for fast and accurate least squares fits using ``A.lstsq(x)``.
    """
    @classmethod
    def get_svd_result(cls, a, eps=None):
        """Construct decomposition from matrix"""
        u, s, vH = np.linalg.svd(a, full_matrices=False)
        where = s.astype(bool) if eps is None else s/s[0] <= eps
        if not where.all():
            return u[:, where], s[where], vH[where]
        else:
            return u, s, vH

    def __init__(self, a, svd_result=None):
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be of matrix form")
        if svd_result is None:
            u, s, vH = self.__class__.get_svd_result(a)
        else:
            u, s, vH = map(np.asarray, svd_result)

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
