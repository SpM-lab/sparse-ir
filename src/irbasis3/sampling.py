# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn

def _double(w):
    # Take middle points
    mid_w = np.array(0.5*(w[:-1]+w[1:]), dtype=w.dtype)
    return np.unique(np.hstack((w, mid_w)))

def _oversampling(sampling_points):
    if issubclass(sampling_points.dtype.type, np.integer):
        if all(sampling_points%2 == 1):
            return 2*_double(sampling_points//2)+1
        elif all(sampling_points%2 == 0):
            return 2*_double(sampling_points//2)
        else:
            raise RuntimeError("Invalid input.")
    elif sampling_points.dtype == np.float64:
        return _double(sampling_points)
    else:
        raise RuntimeError("Invalid input.")

def _mul_with_1d_array(X, y, axis):
    """ Multiply numpy ndarray X with 1d array y along a given axis """
    ss = X.ndim * [None]
    ss[axis] = slice(None)
    return X * y[tuple(ss)]


class SamplingBase:
    """Base class for sparse sampling.

    Encodes the "basis transformation" of a propagator from the truncated IR
    basis coefficients `G_ir[l]` to time/frequency sampled on sparse points
    `G(x[i])` together with its inverse, a least squares fit::

             ________________                   ___________________
            |                |    evaluate     |                   |
            |     Basis      |---------------->|     Value on      |
            |  coefficients  |<----------------|  sampling points  |
            |________________|      fit        |___________________|

    Attributes:
    -----------
     - `basis` : IR Basis instance
     - `matrix` : Evaluation matrix is decomposed form
     - `sampling_points` : Set of sampling points
     - `log_oversampling` : Oversampling factor. The number of the resultant sampling points will be increased approximately by 2^log_oversampling
     - `warn_cond` : Warn if the condition number is large.
     - `regularizer` : None (disable), True (singular values), or a 1D array-like object of floats.
    """
    def __init__(self, basis, sampling_points=None, log_oversampling=0, warn_cond=True, regularizer=None):
        if sampling_points is None:
            sampling_points = self.__class__.default_sampling_points(basis)
        else:
            sampling_points = np.array(sampling_points)
        
        if log_oversampling > 0:
            for _ in range(log_oversampling):
                sampling_points = _oversampling(sampling_points)
        
        if regularizer is None:
            self.regularizer = np.ones(basis.size)
        elif regularizer is True:
            self.regularizer = basis.s
        else:
            self.regularizer = np.asarray(regularizer)

        self.basis = basis
        self.matrix = DecomposedMatrix(
                        self.__class__.eval_matrix(basis, sampling_points) * self.regularizer[None,:]
                        )
        self.sampling_points = sampling_points

        # Check conditioning
        # FIXME: take into account regularizer
        self.cond = self.matrix.s[0] / self.matrix.s[-1]
        if warn_cond and self.cond > 1e8:
            warn("Sampling matrix is poorly conditioned (cond = %.2g)"
                 % self.cond, ConditioningWarning)

    def evaluate(self, al, axis=None):
        """Evaluate the basis coefficients at the sparse sampling points"""
        if axis is None: axis = 0
        al = _mul_with_1d_array(al, 1/self.regularizer, axis=axis)
        return self.matrix.matmul(al, axis)

    def fit(self, ax, axis=None):
        """Fit basis coefficients from the sparse sampling points"""
        if axis is None: axis = 0
        res = self.matrix.lstsq(ax, axis)
        return _mul_with_1d_array(res, self.regularizer, axis=axis)

    @classmethod
    def default_sampling_points(cls, basis):
        """Return default sampling points"""
        raise NotImplementedError()

    @classmethod
    def eval_matrix(cls, basis, x):
        """Return evaluation matrix from coefficients to sampling points"""
        raise NotImplementedError()


class TauSampling(SamplingBase):
    """Sparse sampling in imaginary time.

    Allows the transformation between the IR basis and a set of sampling points
    in (scaled/unscaled) imaginary time.
    """
    @classmethod
    def default_sampling_points(cls, basis):
        poly = basis.u[-1]
        maxima = poly.deriv().roots()
        left = .5 * (maxima[:1] + poly.xmin)
        right = .5 * (maxima[-1:] + poly.xmax)
        return np.concatenate([left, maxima, right])

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

    Attributes:
    -----------
     - `basis` : IR Basis instance
     - `matrix` : Evaluation matrix is decomposed form
     - `sampling_points` : Set of sampling points
    """
    @classmethod
    def default_sampling_points(cls, basis, mitigate=True):
        # Use the (discrete) extrema of the corresponding highest-order basis
        # function in Matsubara.  This turns out to be close to optimal with
        # respect to conditioning for this size (within a few percent).
        polyhat = basis.uhat[-1]
        wn = polyhat.extrema()

        # While the condition number for sparse sampling in tau saturates at a
        # modest level, the conditioning in Matsubara steadily deteriorates due
        # to the fact that we are not free to set sampling points continuously.
        # At double precision, tau sampling is better conditioned than iwn
        # by a factor of ~4 (still OK). To battle this, we fence the largest
        # frequency with two carefully chosen oversampling points, which brings
        # the two sampling problems within a factor of 2.
        if mitigate:
            wn_outer = wn[[0, -1]]
            wn_diff = 2 * np.round(0.025 * wn_outer).astype(int)
            if wn.size >= 20:
                wn = np.hstack([wn, wn_outer - wn_diff])
            if wn.size >= 42:
                wn = np.hstack([wn, wn_outer + wn_diff])
            wn = np.unique(wn)

        return wn

    @classmethod
    def eval_matrix(cls, basis, x):
        return basis.uhat(x).T

    @property
    def wn(self):
        """Sampling points as (reduced) Matsubara frequencies"""
        return self.sampling_points


class DecomposedMatrix:
    """Matrix in SVD decomposed form for fast and accurate fitting.

    Stores a matrix `A` together with its thin SVD form: `A == (u * s) @ vt`.
    This allows for fast and accurate least squares fits using `A.lstsq(x)`.
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
        r = self.u.conj().T @ x
        r = r / (self.s[:, None] if r.ndim > 1 else self.s)
        return self.vt.conj().T @ r

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
