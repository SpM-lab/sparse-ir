# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn

from . import kernel as _kernel
from . import sve


class IRBasis:
    """Intermediate representation (IR) basis in reduced variables.

    For a continuation kernel from real frequencies, ω ∈ [-ωmax, ωmax], to
    imaginary time, τ ∈ [0, β], this class stores the truncated singular
    value expansion or IR basis::

        K(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))

    The functions are given in reduced variables, ``x = 2*τ/β - 1`` and
    ``y = ω/ωmax``, which scales both sides to the interval ``[-1, 1]``.  The
    kernel then only depends on a cutoff parameter ``Λ = β * ωmax``.

    Example:
        The following example code assumes the spectral function is a single
        pole at x = 0.2::

            # Compute IR basis suitable for fermions and β*W <= 42
            import sparse_ir
            basis = sparse_ir.IRBasis(statistics='F', lambda_=42)

            # Assume spectrum is a single pole at x = 0.2, compute G(iw)
            # on the first few Matsubara frequencies
            gl = basis.s * basis.v(0.2)
            giw = gl @ basis.uhat([1, 3, 5, 7])

    See also:
        :class:`FiniteTempBasis` for a basis directly in time/frequency.

    Attributes:
        u (sparse_ir.poly.PiecewiseLegendrePoly):
            Set of IR basis functions on the reduced imaginary time (`x`) axis.
            These functions are stored as piecewise Legendre polynomials.

            To obtain the value of all basis functions at a point or a array of
            points `x`, you can call the function ``u(x)``.  To obtain a single
            basis function, a slice or a subset `l`, you can use ``u[l]``.

        uhat (sparse_ir.poly.PiecewiseLegendreFT):
            Set of IR basis functions on the Matsubara frequency (`wn`) axis.
            These objects are stored as a set of Bessel functions.

            To obtain the value of all basis functions at a Matsubara frequency
            or a array of points `wn`, you can call the function ``uhat(wn)``.
            Note that we expect reduced frequencies, which are simply even/odd
            numbers for bosonic/fermionic objects. To obtain a single basis
            function, a slice or a subset `l`, you can use ``uhat[l]``.

        s:
            vector of singular values of the continuation kernel

        v (sparse_ir.poly.PiecewiseLegendrePoly):
            Set of IR basis functions on the reduced real frequency (`y`) axis.
            These functions are stored as piecewise Legendre polynomials.

            To obtain the value of all basis functions at a point or a array of
            points `y`, you can call the function ``v(y)``.  To obtain a single
            basis function, a slice or a subset `l`, you can use ``v[l]``.
    """
    def __init__(self, statistics, lambda_, eps=None, *, kernel=None,
                 sve_result=None):
        if not (lambda_ >= 0):
            raise ValueError("kernel cutoff lambda must be non-negative")

        self.kernel = _get_kernel(statistics, lambda_, kernel)
        if sve_result is None:
            u, s, v = sve.compute(self.kernel, eps)
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        if eps is None and sve_result is None and not sve.HAVE_XPREC:
            warn("xprec package is not available:\n"
                 "expect single precision (1.5e-8) only as both cutoff and\n"
                 "accuracy of the basis functions")

        # The radius of convergence of the asymptotic expansion is Lambda/2,
        # so for significantly larger frequencies we use the asymptotics,
        # since it has lower relative error.
        _even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self.u = u
        self.uhat = u.hat(_even_odd, n_asymp=self.kernel.conv_radius)
        self.s = s
        self.v = v
        roots_ = v[-1].roots()
        self.sampling_points_v = np.hstack(
            (v.xmin, 0.5 * (roots_[0:-1] + roots_[1:]), v.xmax))
        self.statistics = statistics

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: `basis[:3]`.
        """
        sve_result = self.u[index], self.s[index], self.v[index]
        return self.__class__(self.kernel, self.statistics, sve_result=sve_result)

    @property
    def lambda_(self):
        """Basis cutoff parameter Λ = β * ωmax"""
        return self.kernel.lambda_

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self.u.size

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.u.shape

    @property
    def beta(self):
        """Inverse temperature (this is `None` because unscaled basis)"""
        return None

    @property
    def wmax(self):
        """Frequency cutoff (this is `None` because unscaled basis)"""
        return None

    @property
    def sve_result(self):
        return self.u, self.s, self.v

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time/x axis"""
        return _default_tau_sampling_points(self.u)

    def default_matsubara_sampling_points(self, *, mitigate=True):
        """Default sampling points on the imaginary frequency axis"""
        return _default_matsubara_sampling_points(self.uhat, mitigate)


class FiniteTempBasis:
    """Intermediate representation (IR) basis for given temperature.

    For a continuation kernel from real frequencies, ω ∈ [-wmax, wmax], to
    imaginary time, τ ∈ [0, beta], this class stores the truncated singular
    value expansion or IR basis::

        K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in range(L))

    This basis is inferred from a reduced form by appropriate scaling of
    the variables.

    Example:
        The following example code assumes the spectral function is a single
        pole at ω = 2.5::

            # Compute IR basis for fermions and β = 10, W <= 4.2
            import sparse_ir
            basis = sparse_ir.FiniteTempBasis(statistics='F', beta=10, wmax=4.2)

            # Assume spectrum is a single pole at ω = 2.5, compute G(iw)
            # on the first few Matsubara frequencies
            gl = basis.s * basis.v(2.5)
            giw = gl @ basis.uhat([1, 3, 5, 7])

    Attributes:
        u (sparse_ir.poly.PiecewiseLegendrePoly):
            Set of IR basis functions on the imaginary time (`tau`) axis.
            These functions are stored as piecewise Legendre polynomials.

            To obtain the value of all basis functions at a point or a array of
            points `x`, you can call the function ``u(x)``.  To obtain a single
            basis function, a slice or a subset `l`, you can use ``u[l]``.

        uhat (sparse_ir.poly.PiecewiseLegendreFT):
            Set of IR basis functions on the Matsubara frequency (`wn`) axis.
            These objects are stored as a set of Bessel functions.

            To obtain the value of all basis functions at a Matsubara frequency
            or a array of points `wn`, you can call the function ``uhat(wn)``.
            Note that we expect reduced frequencies, which are simply even/odd
            numbers for bosonic/fermionic objects. To obtain a single basis
            function, a slice or a subset `l`, you can use ``uhat[l]``.

        s:
            vector of singular values of the continuation kernel

        v (sparse_ir.poly.PiecewiseLegendrePoly):
            Set of IR basis functions on the real frequency (`w`) axis.
            These functions are stored as piecewise Legendre polynomials.

            To obtain the value of all basis functions at a point or a array of
            points `w`, you can call the function ``v(w)``.  To obtain a single
            basis function, a slice or a subset `l`, you can use ``v[l]``.
    """
    def __init__(self, statistics, beta, wmax, eps=None, *, kernel=None,
                 sve_result=None):
        if not (beta > 0):
            raise ValueError("inverse temperature beta must be positive")
        if not (wmax >= 0):
            raise ValueError("frequency cutoff must be non-negative")

        self.kernel = _get_kernel(statistics, beta * wmax, kernel)
        if sve_result is None:
            u, s, v = sve.compute(self.kernel, eps)
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        self.statistics = statistics
        self.beta = beta

        if eps is None and sve_result is None and not sve.HAVE_XPREC:
            warn("xprec package is not available:\n"
                 "expect single precision (1.5e-8) only as both cutoff and\n"
                 "accuracy of the basis functions")

        # The polynomials are scaled to the new variables by transforming the
        # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
        # the data is not necessary as the normalization is inferred.
        wmax = self.kernel.lambda_ / self.beta
        self.u = u.__class__(u.data, beta/2 * (u.knots + 1))
        self.v = v.__class__(v.data, wmax * v.knots)
        roots_ = v[-1].roots()
        self.sampling_points_v = np.hstack(
            (-wmax, 0.5 * (roots_[0:-1] + roots_[1:]), wmax))

        # The singular values are scaled to match the change of variables, with
        # the additional complexity that the kernel may have an additional
        # power of w.
        self.s = np.sqrt(beta/2 * wmax) * (wmax**(-self.kernel.ypower)) * s

        # HACK: as we don't yet support Fourier transforms on anything but the
        # unit interval, we need to scale the underlying data.  This breaks
        # the correspondence between U.hat and Uhat though.
        uhat_base = u.__class__(np.sqrt(beta) * u.data, u.knots)

        conv_radius = 40 * self.kernel.lambda_
        _even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self.uhat = uhat_base.hat(_even_odd, conv_radius)

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: `basis[:3]`.
        """
        sve_result = self.u[index], self.s[index], self.v[index]
        return self.__class__(self.kernel, self.statistics, self.beta,
                              sve_result=sve_result)

    @property
    def wmax(self):
        """Cutoff in real frequency."""
        return self.kernel.lambda_ / self.beta

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self.u.size

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.u.shape

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time/x axis"""
        return _default_tau_sampling_points(self.u)

    def default_matsubara_sampling_points(self, *, mitigate=True):
        """Default sampling points on the imaginary frequency axis"""
        return _default_matsubara_sampling_points(self.uhat, mitigate)


def _default_tau_sampling_points(u):
    poly = u[-1]
    maxima = poly.deriv().roots()
    left = .5 * (maxima[:1] + poly.xmin)
    right = .5 * (maxima[-1:] + poly.xmax)
    return np.concatenate([left, maxima, right])


def _default_matsubara_sampling_points(uhat, mitigate=True):
    # Use the (discrete) extrema of the corresponding highest-order basis
    # function in Matsubara.  This turns out to be close to optimal with
    # respect to conditioning for this size (within a few percent).
    polyhat = uhat[-1]
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

    # For boson, include "0".
    if wn[0] % 2 == 0:
        wn = np.unique(np.hstack((0, wn)))

    return wn


def _get_kernel(statistics, lambda_, kernel):
    if statistics not in 'BF':
        raise ValueError("statistics must either be 'B' (for bosonic basis) "
                         "or 'F' (for fermionic basis)")
    if kernel is None:
        if statistics == 'F':
            kernel = _kernel.KernelFFlat(lambda_)
        else:
            kernel = _kernel.KernelBFlat(lambda_)
    else:
        try:
            lambda_kernel = kernel.lambda_
        except AttributeError:
            pass
        else:
            if not np.allclose(lambda_kernel, lambda_, atol=0, rtol=4e-16):
                raise ValueError("lambda of kernel and basis mismatch")
    return kernel
