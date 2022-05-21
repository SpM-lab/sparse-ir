# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
from typing import Tuple
import numpy as np
from warnings import warn

from . import kernel as _kernel
from . import poly
from . import sve


class AbstractBasis:
    """Abstract base class for intermediate representation bases."""
    @property
    def u(self):
        """Basis functions on the (reduced) imaginary time axis.

        Set of IR basis functions on the imaginary time (`tau`) or reduced
        imaginary time (`x`) axis.

        To obtain the value of all basis functions at a point or a array of
        points `x`, you can call the function ``u(x)``.  To obtain a single
        basis function, a slice or a subset `l`, you can use ``u[l]``.
        """
        raise NotImplementedError()

    @property
    def uhat(self):
        """Basis functions on the reduced Matsubara frequency (`wn`) axis.

        To obtain the value of all basis functions at a Matsubara frequency
        or a array of points `wn`, you can call the function ``uhat(wn)``.
        Note that we expect reduced frequencies, which are simply even/odd
        numbers for bosonic/fermionic objects. To obtain a single basis
        function, a slice or a subset `l`, you can use ``uhat[l]``.
        """
        raise NotImplementedError()

    @property
    def s(self):
        """Vector of singular values of the continuation kernel"""
        raise NotImplementedError()

    @property
    def v(self):
        """Basis functions on the (reduced) real frequency axis.

        Set of IR basis functions on the real frequency (`omega`) or reduced
        real-frequency (`y`) axis.

        To obtain the value of all basis functions at a point or a array of
        points `y`, you can call the function ``v(y)``.  To obtain a single
        basis function, a slice or a subset `l`, you can use ``v[l]``.
        """
        raise NotImplementedError()

    @property
    def statistics(self):
        """Quantum statistic (`"F"` for fermionic, `"B"` for bosonic)"""
        raise NotImplementedError()

    @property
    def accuracy(self):
        """Accuracy of singular value cutoff"""
        return self.s[-1] / self.s[0]

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: `basis[:3]`.
        """
        raise NotImplementedError()

    @property
    def size(self):
        """Number of basis functions / singular values"""
        return self.s.size

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.s.shape

    @property
    def kernel(self):
        """Kernel of which this is the singular value expansion"""
        raise NotImplementedError()

    @property
    def sve_result(self):
        raise NotImplementedError()

    @property
    def lambda_(self):
        """Basis cutoff parameter Λ = β * ωmax"""
        return self.kernel.lambda_

    @property
    def beta(self):
        """Inverse temperature or `None` because unscaled basis"""
        raise NotImplementedError()

    @property
    def wmax(self):
        """Real frequency cutoff (this is `None` because unscaled basis)"""
        raise NotImplementedError()

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time/x axis"""
        return _default_sampling_points(self.u)

    def default_omega_sampling_points(self):
        """Default sampling points on the real frequency axis"""
        return self.v[-1].deriv().roots()

    def default_matsubara_sampling_points(self, *, mitigate=True):
        """Default sampling points on the imaginary frequency axis"""
        return _default_matsubara_sampling_points(self.uhat, mitigate)

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return True


class DimensionlessBasis(AbstractBasis):
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
            basis = sparse_ir.DimensionlessBasis(statistics='F', lambda_=42)

            # Assume spectrum is a single pole at x = 0.2, compute G(iw)
            # on the first few Matsubara frequencies
            gl = basis.s * basis.v(0.2)
            giw = gl @ basis.uhat([1, 3, 5, 7])

    See also:
        :class:`FiniteTempBasis` for a basis directly in time/frequency.
    """
    def __init__(self, statistics, lambda_, eps=None, *, kernel=None,
                 sve_result=None):
        if not (lambda_ >= 0):
            raise ValueError("kernel cutoff lambda must be non-negative")

        if eps is None and sve_result is None and not sve.HAVE_XPREC:
            warn("xprec package is not available:\n"
                 "expect single precision (1.5e-8) only as both cutoff and\n"
                 "accuracy of the basis functions")

        # Calculate basis functions from truncated singular value expansion
        self._kernel = _get_kernel(statistics, lambda_, kernel)
        if sve_result is None:
            sve_result = sve.compute(self._kernel, eps)
            u, s, v = sve_result
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        self._statistics = statistics

        # The radius of convergence of the asymptotic expansion is Lambda/2,
        # so for significantly larger frequencies we use the asymptotics,
        # since it has lower relative error.
        even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self._u = u
        self._uhat = u.hat(even_odd, n_asymp=self._kernel.conv_radius)
        self._s = s
        self._v = v

    def __getitem__(self, index):
        u, s, v = self.sve_result
        sve_result = u[index], s[index], v[index]
        return DimensionlessBasis(self._statistics, self._kernel.lambda_,
                                  kernel=self._kernel, sve_result=sve_result)

    @property
    def statistics(self): return self._statistics

    @property
    def u(self) -> poly.PiecewiseLegendrePoly: return self._u

    @property
    def uhat(self) -> poly.PiecewiseLegendreFT: return self._uhat

    @property
    def s(self) -> np.ndarray: return self._s

    @property
    def v(self) -> poly.PiecewiseLegendrePoly: return self._v

    @property
    def kernel(self): return self._kernel

    @property
    def beta(self): return None

    @property
    def wmax(self): return None

    @property
    def sve_result(self):
        return self._u, self._s, self._v



class FiniteTempBasis(AbstractBasis):
    """Intermediate representation (IR) basis for given temperature.

    For a continuation kernel from real frequencies, ω ∈ [-ωmax, ωmax], to
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
    """
    def __init__(self, statistics, beta, wmax, eps=None, *, kernel=None,
                 sve_result=None):
        if not (beta > 0):
            raise ValueError("inverse temperature beta must be positive")
        if not (wmax >= 0):
            raise ValueError("frequency cutoff must be non-negative")

        if eps is None and sve_result is None and not sve.HAVE_XPREC:
            warn("xprec package is not available:\n"
                 "expect single precision (1.5e-8) only as both cutoff and\n"
                 "accuracy of the basis functions")

        # Calculate basis functions from truncated singular value expansion
        self._kernel = _get_kernel(statistics, beta * wmax, kernel)
        if sve_result is None:
            sve_result = sve.compute(self._kernel, eps)
            u, s, v = sve_result
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        if u.xmin != -1 or u.xmax != 1:
            raise RuntimeError("u must be defined in the reduced variable.")

        self._sve_result = sve_result
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._accuracy = s[-1] / s[0]

        # The polynomials are scaled to the new variables by transforming the
        # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
        # the data is not necessary as the normalization is inferred.
        self._u = u.__class__(u.data, beta/2 * (u.knots + 1), beta/2 * u.dx, u.symm)
        self._v = v.__class__(v.data, wmax * v.knots, wmax * v.dx, v.symm)

        # The singular values are scaled to match the change of variables, with
        # the additional complexity that the kernel may have an additional
        # power of w.
        self._s = np.sqrt(beta/2 * wmax) * (wmax**(-self.kernel.ypower)) * s

        # HACK: as we don't yet support Fourier transforms on anything but the
        # unit interval, we need to scale the underlying data.  This breaks
        # the correspondence between U.hat and Uhat though.
        uhat_base = u.__class__(np.sqrt(beta) * u.data, u, symm=u.symm)

        conv_radius = 40 * self.kernel.lambda_
        _even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self._uhat = uhat_base.hat(_even_odd, conv_radius)

    def __getitem__(self, index):
        u, s, v = self.sve_result
        sve_result = u[index], s[index], v[index]
        return FiniteTempBasis(self._statistics, self._beta, self._wmax,
                               kernel=self._kernel, sve_result=sve_result)

    @property
    def statistics(self): return self._statistics

    @property
    def beta(self): return self._beta

    @property
    def wmax(self): return self._wmax

    @property
    def u(self) -> poly.PiecewiseLegendrePoly: return self._u

    @property
    def uhat(self) -> poly.PiecewiseLegendreFT: return self._uhat

    @property
    def s(self) -> np.ndarray: return self._s

    @property
    def v(self) -> poly.PiecewiseLegendrePoly: return self._v

    @property
    def kernel(self): return self._kernel

    @property
    def sve_result(self): return self._sve_result


def finite_temp_bases(
            beta: float, wmax: float, eps: float = None,
            sve_result: tuple = None
        )-> Tuple[FiniteTempBasis, FiniteTempBasis]:
    """Construct FiniteTempBasis objects for fermion and bosons

    Construct FiniteTempBasis objects for fermion and bosons using
    the same LogisticKernel instance.
    """
    if sve_result is None:
        sve_result = sve.compute(_kernel.LogisticKernel(beta*wmax), eps)
    basis_f = FiniteTempBasis("F", beta, wmax, eps, sve_result=sve_result)
    basis_b = FiniteTempBasis("B", beta, wmax, eps, sve_result=sve_result)
    return basis_f, basis_b



def _default_sampling_points(u):
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
        kernel = _kernel.LogisticKernel(lambda_)
    else:
        try:
            lambda_kernel = kernel.lambda_
        except AttributeError:
            pass
        else:
            if not np.allclose(lambda_kernel, lambda_, atol=0, rtol=4e-16):
                raise ValueError("lambda of kernel and basis mismatch")
    return kernel
