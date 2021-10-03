# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
"""
Drop-in replacement for the irbasis module.

This is designed to be a drop-in replacement for `irbasis`, where the basis
can be computed on-the-fly for arbitrary values of Lambda.  In other words,
you should be able to replace `irbasis` with `irbasis3.adapter` and
everything should hopefully still work.

Note however that on-the-fly computation typically has lower accuracy.  Thus,
by default we only populate the basis down to singular values of 1.5e-8.
You can change this by setting the `ACCURACY` parameter.
"""
import numpy as _np

from . import kernel as _kernel
from . import poly as _poly
from . import sve as _sve

try:
    import xprec
except ImportError:
    raise RuntimeError("xprec is mandatory for using the adapter module!")

ACCURACY = 1.0e-15


def load(statistics, Lambda, h5file=None):
    if statistics == "F":
        K = _kernel.KernelFFlat(Lambda)
    elif statistics == "B":
        K = _kernel.KernelBFlat(Lambda)
    else:
        raise ValueError("Unknown statistics")

    u, s, v = _sve.compute(K, eps=ACCURACY)
    return Basis(statistics, Lambda, u, s, v)


class Basis:
    def __init__(self, statistics, Lambda, u, s, v):
        self._statistics = statistics
        self._Lambda = Lambda
        self._u = u
        self._s = s
        self._v = v

    @property
    def Lambda(self):
        """Dimensionless parameter of IR basis"""
        return self._Lambda

    @property
    def statistics(self):
        """Statistics, either "F" for fermions or "B" for bosons"""
        return self._statistics

    def dim(self):
        """Return dimension of basis"""
        return self._s.size

    def sl(self, l=None):
        """Return the singular value for the l-th basis function"""
        return _select(self._s, l)

    def ulx(self, l, x):
        """Return value of basis function for x"""
        return _selectvals(self._u, l, x)

    def d_ulx(self, l, x, order, section=None):
        """Return (higher-order) derivatives of u_l(x)"""
        return _selectvals(self._u.deriv(order), l, x)

    def vly(self, l, y):
        """Return value of basis function for y"""
        return _selectvals(self._v, l, y)

    def d_vly(self, l, y, order):
        """Return (higher-order) derivatives of v_l(y)"""
        return _selectvals(self._v.deriv(order), l, y)

    def compute_unl(self, n, whichl=None):
        """Compute transformation matrix from IR to Matsubara frequencies"""
        n = _np.ravel(n)
        freq = {'F': 'odd', 'B': 'even'}[self._statistics]
        uhat = _select(self._u, whichl).hat(freq)
        return _np.squeeze(uhat(2 * n + uhat.zeta).T)

    def num_sections_x(self):
        "Number of sections of piecewise polynomial representation of u_l(x)"
        return self._u.nsegments

    @property
    def section_edges_x(self):
        """End points of sections for u_l(x)"""
        return self._u.knots

    def num_sections_y(self):
        "Number of sections of piecewise polynomial representation of v_l(y)"
        return self._v.nsegments

    @property
    def section_edges_y(self):
        """End points of sections for v_l(y)"""
        return self._v.knots

    def sampling_points_x(self, whichl):
        """Computes "optimal" sampling points in x space for given basis"""
        return _poly.sampling_points_x(self._u[whichl])

    def sampling_points_y(self, whichl):
        """Computes "optimal" sampling points in y space for given basis"""
        return _poly.sampling_points_x(self._v[whichl])

    def sampling_points_matsubara(self, whichl):
        """Computes sampling points in Matsubara domain for given basis"""
        zeta = 0 if self.statistics == 'B' else 1
        part = ['real', 'imag'][(whichl + zeta) % 2]
        return _poly.find_hat_extrema(self._u[whichl], part, zeta)


def sampling_points_x(b, whichl):
    """Computes "optimal" sampling points in x space for given basis"""
    return b.sampling_points_x(whichl)


def sampling_points_y(b, whichl):
    """Computes "optimal" sampling points in y space for given basis"""
    return b.sampling_points_y(whichl)


def sampling_points_matsubara(b, whichl):
    """Computes "optimal" sampling points in Matsubara domain for given basis"""
    return b.sampling_points_matsubara(whichl)


def _select(p, l):
    return p if l is None else p[l]


def _selectvals(p, l, x):
    return p(x) if l is None else p.value(l, x)
