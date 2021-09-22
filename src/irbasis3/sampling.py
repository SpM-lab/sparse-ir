# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np


class DecomposedMatrix:
    @classmethod
    def from_matrix(cls, a, eps=None):
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        where = s.astype(bool) if eps is None else s/s[0] <= eps
        if not where.all():
            u = u[:, where]
            s = s[where]
            v = vt[where]
        return cls(u, s, vt)

    def __init__(self, u, s, vt):
        self.u = np.asarray(u)
        self.s = np.asarray(s)
        self.vt = np.asarray(vt)

    def __matmul__(self, x):
        r = self.vt @ x
        r = (self.s[:, None] if r.ndim > 1 else self.s) * r
        return self.u @ r

    def lstsq(self, x):
        r = self.u.T @ x
        r = r / (self.s[:, None] if r.ndim > 1 else self.s)
        return self.vt.T @ r

    def __array__(self, dtype=None):
        return (self.u * self.s) @ self.vt
