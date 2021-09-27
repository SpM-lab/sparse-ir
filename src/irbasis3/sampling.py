# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

from . import svd


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
        return np.tensordot(self.a, x, (-1,0))

    def matmul(self, x, axis=None):
        """Compute `A @ x` (optionally along specified axis of x)"""
        if axis is None:
            return self._lstsq(x)

        x = np.asarray(x)
        x = np.moveaxis(x, axis, 0)
        r = self @ x
        return np.moveaxis(r, 0, axis)

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
        x = np.rollaxis(x, axis, target_axis + 1)
        r = self._lstsq(x)
        return np.rollaxis(r, target_axis, axis)

    def __array__(self, dtype=None):
        """Convert to numpy array."""
        return self.a.astype(dtype)
