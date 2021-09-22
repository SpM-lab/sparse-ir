# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import numpy.polynomial.legendre as np_legendre
import scipy.special as sp_special

from . import _roots


class PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial.

    Models a function on the interval `[-1, 1]` as a set of segments on the
    intervals `S[i] = [a[i], a[i+1]]`, where on each interval the function
    is expanded in scaled Legendre polynomials.
    """
    def __init__(self, data, knots, dx=None):
        """Piecewise Legendre polynomial"""
        if isinstance(knots, self.__class__):
            self.__dict__.update(knots.__dict__)
            self.data = data
            return

        data = np.array(data)
        knots = np.array(knots)
        polyorder, nsegments = data.shape[:2]
        if knots.shape != (nsegments+1,):
            raise ValueError("Invalid knots array")
        if not (knots[1:] >= knots[:-1]).all():
            raise ValueError("Knots must be monotonically increasing")
        if dx is None:
            dx = knots[1:] - knots[:-1]
        else:
            dx = np.array(dx)
            if not np.allclose(dx, knots[1:] - knots[:-1]):
                raise ValueError("dx must work with knots")

        self.nsegments = nsegments
        self.polyorder = polyorder
        self.xmin = knots[0]
        self.xmax = knots[-1]

        self.knots = knots
        self.dx = dx
        self.data = data
        self._xm = .5 * (knots[1:] + knots[:-1])
        self._inv_xs = 2/dx
        self._norm = np.sqrt(self._inv_xs)

    def __getitem__(self, l):
        """Return part of a set of piecewise polynomials"""
        if isinstance(l, tuple):
            new_data = self.data[(slice(None), slice(None), *l)]
        else:
            new_data = self.data[:,:,l]
        return self.__class__(new_data, self)

    def __call__(self, x):
        """Evaluate polynomial at position x"""
        i, xtilde = self._split(np.asarray(x))
        data = self.data[:, i]

        # Evaluate for all values of l.  x and data array must be
        # broadcast'able against each other, so we append dimensions here
        func_dims = self.data.ndim - 2
        datashape = i.shape + (1,) * func_dims
        res = np_legendre.legval(xtilde.reshape(datashape), data, tensor=False)
        res *= self._norm[i.reshape(datashape)]

        # Finally, exchange the x and vector dimensions
        order = tuple(range(i.ndim, i.ndim + func_dims)) + tuple(range(i.ndim))
        return res.transpose(*order)

    def value(self, l, x):
        """Return value for l and x."""
        if self.data.ndim != 3:
            raise ValueError("Only allowed for vector of data")

        l, x = np.broadcast_arrays(l, x)
        i, xtilde = self._split(x)
        data = self.data[:, i, l]

        # This should now neatly broadcast against each other
        res = np_legendre.legval(xtilde, data, tensor=False)
        res *= self._norm[i]
        return res

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        ddata = np_legendre.legder(self.data, n)

        _scale_shape = (1, -1) + (1,) * (self.data.ndim - 2)
        scale = self._inv_xs ** n
        ddata *= scale.reshape(_scale_shape)
        return self.__class__(ddata, self.knots, self.dx)

    def hat(self, freq):
        """Get Fourier transformed object"""
        return PiecewiseLegendreFT(self, freq)

    def roots(self, alpha=2):
        """Find all roots of the piecewise polynomial

        Assume that between each two knots (pieces) there are at most `alpha`
        roots.
        """
        if self.data.ndim > 2:
            raise ValueError("select single polynomial before calling roots()")

        grid = _refine_grid(self.knots, alpha)
        return _roots.find_all(self, grid)

    @property
    def shape(self): return self.data.shape[2:]

    @property
    def size(self): return self.data[:1,:1].size

    @property
    def ndim(self): return self.data.ndim - 2

    def _in_domain(self, x):
        return (x >= self.xmin).all() and (x <= self.xmax).all()

    def _split(self, x):
        """Split segment"""
        if not self._in_domain(x):
            raise ValueError("x must be in [%g, %g]" % (self.xmin, self.xmax))

        i = self.knots.searchsorted(x, 'right').clip(None, self.nsegments)
        i -= 1
        xtilde = x - self._xm[i]
        xtilde *= self._inv_xs[i]
        return i, xtilde


class PiecewiseLegendreFT:
    """Fourier transform of a piecewise Legendre polynomial.

    For a given frequency index `n`, the Fourier transform of the Legendre
    function is defined as:

            phat(n) == ∫ dx exp(1j * pi * n * x / (xmax - xmin)) p(x)

    The polynomial is continued either periodically (`freq='even'`), in which
    case `n` must be even, or antiperiodically (`freq='odd'`), in which case
    `n` must be odd.
    """
    _DEFAULT_GRID = np.hstack([np.arange(2**6),
                        (2**np.linspace(6, 25, 16*(25-6)+1)).astype(int)])

    def __init__(self, poly, freq='even'):
        if poly.xmin != -1 or poly.xmax != 1:
            raise NotImplementedError("Only interval [-1, 1] supported")
        self.poly = poly
        self.freq = freq
        self.zeta = {'even': 0, 'odd': 1}[freq]

    def __getitem__(self, l):
        return self.__class__(self.poly[l], self.freq)

    def __call__(self, n):
        """Obtain Fourier transform of polynomial for given frequencies"""
        n = self._check_domain(n)
        result_flat = _compute_unl(self.poly, n.ravel())
        return result_flat.reshape(result_flat.shape[:-1] + n.shape)

    def extrema(self, part=None, grid=None):
        """Obtain extrema of fourier-transformed polynomial."""
        if self.poly.shape:
            raise ValueError("select single polynomial")
        if grid is None:
            grid = self._DEFAULT_GRID
        if part is None:
            parity = self.poly(self.poly.xmax) / self.poly(self.poly.xmin)
            if np.allclose(parity, 1):
                part = 'real' if self.zeta == 0 else 'imag'
            elif np.allclose(parity, -1):
                part = 'imag' if self.zeta == 0 else 'real'
            else:
                raise ValueError("cannot detect parity.")
        if part == 'real':
            f = lambda n: self(2*n + self.zeta).real
        elif part == 'imag':
            f = lambda n: self(2*n + self.zeta).imag
        else:
            raise ValueError("part must be either 'real' or 'imag'")

        x0 = _roots.discrete_extrema(f, PiecewiseLegendreFT._DEFAULT_GRID)
        x0 = 2 * x0 + self.zeta
        if x0[0] == 0:
            x0 = np.hstack([-x0[::-1], x0[1:]])
        else:
            x0 = np.hstack([-x0[::-1], x0])
        return x0

    def _check_domain(self, n):
        n = np.asarray(n)
        if np.issubdtype(n.dtype, np.integer):
            nint = n
        else:
            nint = n.astype(int)
            if not (n == nint).all():
                raise ValueError("n must be integers")
        if not (nint % 2 == self.zeta).all():
            raise ValueError("n have wrong parity")
        return nint


def sampling_points_x(poly):
    maxima = poly.deriv().roots()
    left = .5 * (maxima[:1] + poly.xmin)
    right = .5 * (maxima[-1:] + poly.xmax)
    return np.concatenate([left, maxima, right])


def _imag_power(n):
    """Imaginary unit raised to an integer power without numerical error"""
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        raise ValueError("expecting set of integers here")
    cycle = np.array([1, 0+1j, -1, 0-1j], complex)
    return cycle[n % 4]


def _get_tnl(l, w):
    r"""Fourier integral of the l-th Legendre polynomial:

        T_l(w) = \int_{-1}^1 dx \exp(iwx) P_l(x)
    """
    i_pow_l = _imag_power(l)
    return 2 * np.where(
        w >= 0,
        i_pow_l * sp_special.spherical_jn(l, w),
        (i_pow_l * sp_special.spherical_jn(l, -w)).conj(),
        )


def _shift_xmid(knots, dx):
    r"""Return midpoint relative to the nearest integer plus a shift

    Return the midpoints `xmid` of the segments, as pair `(diff, shift)`,
    where shift is in `(0,1,-1)` and `diff` is a float such that
    `xmid == shift + diff` to floating point accuracy.
    """
    dx_half = dx / 2
    xmid_m1 = dx.cumsum() - dx_half
    xmid_p1 = -dx[::-1].cumsum()[::-1] + dx_half
    xmid_0 = knots[1:] - dx_half

    shift = np.round(xmid_0).astype(int)
    diff = np.choose(shift+1, (xmid_m1, xmid_0, xmid_p1))
    return diff, shift


def _phase_stable(poly, wn):
    """Phase factor for the piecewise Legendre to Matsubara transform.

    Compute the following phase factor in a stable way:
        np.exp(1j * np.pi/2 * wn[:,None] * poly.dx.cumsum()[None,:])
    """
    # A naive implementation is losing precision close to x=1 and/or x=-1:
    # there, the multiplication with `wn` results in `wn//4` almost extra turns
    # around the unit circle.  The cosine and sines will first map those
    # back to the interval [-pi, pi) before doing the computation, which loses
    # digits in dx.  To avoid this, we extract the nearest integer dx.cumsum()
    # and rewrite above expression like below.
    #
    # Now `wn` still results in extra revolutions, but the mapping back does
    # not cut digits that were not there in the first place.
    xmid_diff, extra_shift = _shift_xmid(poly.knots, poly.dx)
    phase_shifted = np.exp(1j * np.pi/2 * wn[None,:] * xmid_diff[:,None])
    corr = _imag_power((extra_shift[:,None] + 1) * wn[None,:])
    return corr * phase_shifted


def _compute_unl(poly, wn):
    """Compute piecewise Legendre to Matsubara transform."""
    dx_half = poly.dx / 2

    data_flat = poly.data.reshape(*poly.data.shape[:2], -1)
    data_sc = data_flat * np.sqrt(dx_half/2)[None,:,None]
    p = np.arange(poly.polyorder)

    wred = np.pi/2 * wn
    phase_wi = _phase_stable(poly, wn)
    t_pin = _get_tnl(p[:,None,None], wred[None,:] * dx_half[:,None]) * phase_wi

    # Perform the following, but faster:
    #   resulth = einsum('pin,pil->nl', t_pin, data_sc)
    npi = poly.polyorder * poly.nsegments
    result_flat = t_pin.reshape(npi,-1).T.dot(data_sc.reshape(npi,-1)).T
    return result_flat.reshape(*poly.data.shape[2:], wn.size)


def _refine_grid(knots, alpha):
    """Linear refinement of grid"""
    result = np.linspace(knots[:-1], knots[1:], alpha, endpoint=False)
    return np.hstack((result.T.ravel(), knots[-1]))
