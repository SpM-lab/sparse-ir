# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Drop-in replacement for the irbasis module.

This is designed to be a drop-in replacement for ``irbasis``, where the basis
can be computed on-the-fly for arbitrary values of Lambda.  In other words,
you should be able to replace ``irbasis`` with ``sparse_ir.adapter`` and
everything should hopefully still work.

Note however that on-the-fly computation typically has lower accuracy unless
xprec is available. Thus, by default we only populate the basis down to
singular values of ~1e-9 and emit a warning. You can squelch the warning by
setting `WARN_ACCURACY` to false.
"""
# Do not import additional public symbols into this namespace, always use
# underscores - this module should look as much as possible like `irbasis`!
import numpy as _np
from warnings import warn as _warn

from . import basis as _basis
from . import poly as _poly
from . import kernel as _kernel

try:
    import xprec as _xprec
except ImportError:
    ACCURACY = 1.0e-9
    WARN_ACCURACY = True
else:
    ACCURACY = 1.0e-15
    WARN_ACCURACY = False


def load(statistics, Lambda, h5file=None):
    if WARN_ACCURACY:
        _warn("xprec package is not found - expect degraded accuracy!\n"
              "To squelch this warning, set WARN_ACCURACY to False.")

    kernel_type = {"F": _kernel.LogisticKernel,
                   "B": _kernel.RegularizedBoseKernel}[statistics]
    basis = _basis.DimensionlessBasis(statistics, float(Lambda),
                                      kernel=kernel_type(Lambda))
    return Basis(statistics, Lambda, (basis.u, basis.s, basis.v))


class Basis:
    def __init__(self, statistics, Lambda, sve_result):
        u, s, v = sve_result
        self._statistics = statistics
        self._Lambda = Lambda
        self._u = u
        self._s = s
        self._v = v

        conv_radius = 40 * Lambda
        even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self._uhat = _poly.PiecewiseLegendreFT(u, even_odd, conv_radius)

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
        nn = 2 * n + self._uhat.zeta
        return _np.squeeze(_select(self._uhat, whichl)(nn).T)

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
        return sampling_points_x(self, whichl)

    def sampling_points_y(self, whichl):
        """Computes "optimal" sampling points in y space for given basis"""
        return sampling_points_y(self, whichl)

    def sampling_points_matsubara(self, whichl):
        """Computes sampling points in Matsubara domain for given basis"""
        return sampling_points_matsubara(self, whichl)



def _select(p, l):
    return p if l is None else p[l]


def _selectvals(p, l, x):
    return p(x) if l is None else p.value(l, x)


"""" CODE BELOW IS TAKEN FROM IRBAIS FOR COMPATIBLITITY"""
def _find_roots(ulx):
    """Find all roots in (-1, 1) using double exponential mesh + bisection"""
    Nx = 10000
    eps = 1e-14
    tvec = _np.linspace(-3, 3, Nx)  # 3 is a very safe option.
    xvec = _np.tanh(0.5 * _np.pi * _np.sinh(tvec))

    zeros = []
    for i in range(Nx - 1):
        if ulx(xvec[i]) * ulx(xvec[i + 1]) < 0:
            a = xvec[i + 1]
            b = xvec[i]
            u_a = ulx(a)
            while a - b > eps:
                half_point = 0.5 * (a + b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5 * (a + b))
    return _np.array(zeros)


def _start_guesses(n=1000):
    "Construct points on a logarithmically extended linear interval"
    x1 = _np.arange(n)
    x2 = _np.array(_np.exp(_np.linspace(_np.log(n), _np.log(1E+8), n)), dtype=int)
    x = _np.unique(_np.hstack((x1, x2)))
    return x


def _get_unl_real(basis_xy, x, l):
    "Return highest-order basis function on the Matsubara axis"
    unl = basis_xy.compute_unl(x, l)

    # Purely real functions
    zeta = 1 if basis_xy.statistics == 'F' else 0
    if l % 2 == zeta:
        assert _np.allclose(unl.imag, 0)
        return unl.real
    else:
        assert _np.allclose(unl.real, 0)
        return unl.imag


def _sampling_points(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = _np.asarray(fn)
    fn_abs = _np.abs(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    sign_flip_bounds = _np.hstack((0, sign_flip.nonzero()[0] + 1, fn.size))
    points = []
    for segment in map(slice, sign_flip_bounds[:-1], sign_flip_bounds[1:]):
        points.append(fn_abs[segment].argmax() + segment.start)
    return _np.asarray(points)


def _full_interval(sample, stat):
    if stat == 'F':
        return _np.hstack((-sample[::-1]-1, sample))
    else:
        # If we have a bosonic basis and even order (odd maximum), we have a
        # root at zero. We have to artifically add that zero back, otherwise
        # the condition number will blow up.
        if sample[0] == 0:
            sample = sample[1:]
        return _np.hstack((-sample[::-1], 0, sample))


def _get_mats_sampling(basis_xy, lmax=None):
    "Generate Matsubara sampling points from extrema of basis functions"
    if lmax is None:
        lmax = basis_xy.dim()-1

    x = _start_guesses()
    y = _get_unl_real(basis_xy, x, lmax)
    x_idx = _sampling_points(y)

    sample = x[x_idx]
    return _full_interval(sample, basis_xy.statistics)


def sampling_points_x(b, whichl):
    """Computes "optimal" sampling points in x space for given basis"""
    xroots = _find_roots(b._u[whichl])
    xroots_ex = _np.hstack((-1.0, xroots, 1.0))
    return 0.5 * (xroots_ex[:-1] + xroots_ex[1:])


def sampling_points_y(b, whichl):
    """Computes "optimal" sampling points in y space for given basis"""

    roots_positive_half = 0.5 * _find_roots(lambda y: b.vly(whichl, (y + 1)/2)) + 0.5
    if whichl % 2 == 0:
        roots_ex = _np.sort(
            _np.hstack([-1, -roots_positive_half, roots_positive_half, 1]))
    else:
        roots_ex = _np.sort(
            _np.hstack([-1, -roots_positive_half, 0, roots_positive_half, 1]))
    return 0.5 * (roots_ex[:-1] + roots_ex[1:])


def sampling_points_matsubara(b, whichl):
    """
    Computes "optimal" sampling points in Matsubara domain for given basis

    Parameters
    ----------
    b :
        basis object
    whichl: int
        Index of reference basis function "l"

    Returns
    -------
    sampling_points: 1D array of int
        sampling points in Matsubara domain

    """
    stat = b.statistics

    assert stat == 'F' or stat == 'B' or stat == 'barB'

    if whichl > b.dim()-1:
        raise RuntimeError("Too large whichl")

    return _get_mats_sampling(b, whichl)
