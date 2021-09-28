# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn

from . import svd


class SamplingBase:
    """Base class for sparse sampling.

    Encodes the "basis transformation" of a propagator from the truncated IR
    basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
    `G(x[i])` together with its inverse, a least squares fit:

             ________________                   ___________________
            |                |    evaluate     |                   |
            |     Basis      |---------------->|     Value on      |
            |  coefficients  |<----------------|  sampling points  |
            |________________|      fit        |___________________|

    """
    def __init__(self, basis, x=None):
        if x is None:
            x = self.__class__.default_sampling_points(basis)
        else:
            x = np.asarray(x)

        self.basis = basis
        self.matrix = DecomposedMatrix(self.__class__.eval_matrix(basis, x))
        self.x = x

        # Check conditioning
        cond = self.matrix.s[0] / self.matrix.s[-1]
        if cond > 1e8:
            warn("Sampling matrix is poorly conditioned (cond = %.2g)" % cond,
                 ConditioningWarning)

    def evaluate(self, al, axis=None):
        """Evaluate the basis coefficients"""
        return self.matrix.matmul(al, axis)

    def fit(self, ax, axis=None):
        return self.matrix.lstsq(ax, axis)

    @classmethod
    def default_sampling_points(cls, basis):
        raise NotImplementedError()

    @classmethod
    def eval_matrix(cls, basis, x):
        raise NotImplementedError()


class TauSampling(SamplingBase):
    @classmethod
    def default_sampling_points(cls, basis):
        poly = basis.u[-1]
        maxima = poly.deriv().roots()
        left = .5 * (maxima[:1] + poly.xmin)
        right = .5 * (maxima[-1:] + poly.xmax)
        return np.concatenate([left, maxima, right])

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.u(x)


class MatsubaraSampling(SamplingBase):
    @classmethod
    def default_sampling_points(cls, basis):
        polyhat = basis.uhat[-1]
        return polyhat.extrema()

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.uhat(x)


class DecomposedMatrix:
    """Matrix in SVD decomposed form for fast and accurate fitting.

    Stores a matrix `A` together with its thin SVD form: `A == (u * s) @ vt`.
    This allows for fast and accurate least squares fits using `A.lstsq(x)`.
    """
    @classmethod
    def get_svd_result(cls, a, eps=None):
        """Construct decomposition from matrix"""
        u, s, v = svd.compute(a, strategy='accurate')
        where = s.astype(bool) if eps is None else s/s[0] <= eps
        if not where.all():
            return u[:, where], s[where], v.T[where]
        else:
            return u, s, v.T

    def __init__(self, a, svd_result=None):
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError("a must be of matrix form")
        if svd_result is None:
            u, s, vt = self.__class__.get_svd_result(a)
        else:
            u, s, vt = map(np.asarray, svd_result)

        self.a = a
        self.u = u
        self.s = s
        self.vt = vt

    def __matmul__(self, x):
        """Matrix-matrix multiplication."""
        return self.a @ x

    def matmul(self, x, axis=None):
        """Compute `A @ x` (optionally along specified axis of x)"""
        if axis is None:
            return self @ x

        x = np.asarray(x)
        target_axis = max(x.ndim - 2, 0)
        x = np.moveaxis(x, axis, target_axis)
        r = self @ x
        return np.moveaxis(r, target_axis, axis)

    def _lstsq(self, x):
        r = self.u.T @ x
        r = r / (self.s[:, None] if r.ndim > 1 else self.s)
        return self.vt.T @ r

    def lstsq(self, x, axis=None):
        """Return `y` such that `np.linalg.norm(A @ y - x)` is minimal"""
        if axis is None:
            return self._lstsq(x)

        x = np.asarray(x)
        target_axis = max(x.ndim - 2, 0)
        x = np.moveaxis(x, axis, target_axis)
        r = self._lstsq(x)
        return np.moveaxis(r, target_axis, axis)

    def __array__(self, dtype=None):
        """Convert to numpy array."""
        return self.a.astype(dtype)


class ConditioningWarning(RuntimeWarning):
    pass
