# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
from typing import Tuple
import numpy as np
from warnings import warn

from . import abstract
from . import kernel as _kernel
from . import poly
from . import sve


class FiniteTempBasis(abstract.AbstractBasis):
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
    def __init__(self, statistics, beta, wmax, eps=None, *,
                 max_size=None, kernel=None, sve_result=None):
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

        self._sve_result = sve_result
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax

        u, s, v = sve_result.part(eps, max_size)
        if sve_result.s.size > s.size:
            self._accuracy = sve_result.s[s.size] / s[0]
        else:
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
        # unit interval, we need to scale the underlying data.
        uhat_base_full = poly.PiecewiseLegendrePoly(
                            np.sqrt(beta) * sve_result.u.data, sve_result.u,
                            symm=sve_result.u.symm)
        conv_radius = 40 * self.kernel.lambda_
        even_odd = {'F': 'odd', 'B': 'even'}[statistics]
        self._uhat_full = poly.PiecewiseLegendreFT(uhat_base_full, even_odd,
                                                   n_asymp=conv_radius)
        self._uhat = self._uhat_full[:s.size]

    def __getitem__(self, index):
        return FiniteTempBasis(
                    self._statistics, self._beta, self._wmax, None,
                    max_size=_slice_to_size(index), kernel=self._kernel,
                    sve_result=self._sve_result)

    @property
    def statistics(self): return self._statistics

    @property
    def beta(self): return self._beta

    @property
    def wmax(self): return self._wmax

    @property
    def shape(self): return self._s.shape

    @property
    def size(self): return self._s.size

    @property
    def u(self) -> poly.PiecewiseLegendrePoly: return self._u

    @property
    def uhat(self) -> poly.PiecewiseLegendreFT: return self._uhat

    @property
    def s(self) -> np.ndarray:
        """Vector of singular values of the continuation kernel"""
        return self._s

    @property
    def v(self) -> poly.PiecewiseLegendrePoly:
        """Basis functions on the (reduced) real frequency axis.

        Set of IR basis functions on the real frequency (`omega`) axis.
        To obtain the value of all basis functions at a point or a array of
        points `y`, you can call the function ``v(omega)``.  To obtain a single
        basis function, a slice or a subset `l`, you can use ``v[l]``.
        """
        return self._v

    @property
    def significance(self):
        return self._s / self._s[0]

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def kernel(self):
        """Kernel of which this is the singular value expansion"""
        return self._kernel

    @property
    def sve_result(self):
        return self._sve_result

    def default_tau_sampling_points(self):
        x = _default_sampling_points(self._sve_result.u, self.size)
        return self._beta/2 * (x + 1)

    def default_matsubara_sampling_points(self):
        return _default_matsubara_sampling_points(self._uhat_full, self.size)

    def default_omega_sampling_points(self):
        y = _default_sampling_points(self._sve_result.v, self.size)
        return self._wmax * y

    def rescale(self, new_beta):
        """Return a basis for different temperature.

        Uses the same kernel with the same ``eps``, but a different
        temperature.  Note that this implies a different UV cutoff ``wmax``,
        since ``lambda_ == beta * wmax`` stays constant.
        """
        new_wmax = self._kernel.lambda_ / new_beta
        return FiniteTempBasis(self._statistics, new_beta, new_wmax, None,
                               max_size=self.size, kernel=self._kernel,
                               sve_result=self._sve_result)


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


def _default_sampling_points(u, L):
    if u.xmin != -1 or u.xmax != 1:
        raise ValueError("expecting unscaled functions here")

    # For orthogonal polynomials (the high-T limit of IR), we know that the
    # ideal sampling points for a basis of size L are the roots of the L-th
    # polynomial.  We empirically find that these stay good sampling points
    # for our kernels (probably because the kernels are totally positive).
    if L < u.size:
        return u[L].roots()
    if L > u.size:
        warn(f"Requesting {L} sampling points but we only have {u.size} "
             f"basis functions in SVE.", UserWarning, 3)

    # If we do not have enough polynomials in the basis, we approximate the
    # roots of the L'th polynomial by the extrema of the (L-1)'st basis
    # function, which is sensible due to the strong interleaving property
    # of these functions' roots.
    maxima = u[-1].deriv().roots()

    # Putting the sampling points right at [0, beta], which would be the
    # local extrema, is slightly worse conditioned than putting it in the
    # middel.  This can be understood by the fact that the roots never
    # occur right at the border.
    left = .5 * (maxima[:1] + u.xmin)
    right = .5 * (maxima[-1:] + u.xmax)
    return np.concatenate([left, maxima, right])


def _default_matsubara_sampling_points(uhat, L, fence=False):
    l_requested = L

    # The number of sign changes is always odd for bosonic basis (freq=='even')
    # and even for fermionic basis (freq='odd').  So in order to get at least
    # as many sign changes as basis functions.
    if uhat.freq == 'odd' and l_requested % 2 == 1:
        l_requested += 1
    elif uhat.freq == 'even' and l_requested % 2 == 0:
        l_requested += 1

    # As with the zeros, the sign changes provide excellent sampling points
    if l_requested < uhat.size:
        wn = uhat[l_requested].sign_changes()
    else:
        if l_requested > uhat.size:
            warn(f"Requesting {L} sampling frequencies but only {uhat.size} "
                 f"basis functions in SVE.", UserWarning, 3)

        # As a fallback, use the (discrete) extrema of the corresponding
        # highest-order basis function in Matsubara.  This turns out to be okay.
        polyhat = uhat[-1]
        wn = polyhat.extrema()

        # For bosonic bases, we must explicitly include the zero frequency,
        # otherwise the condition number blows up.
        if wn[0] % 2 == 0:
            wn = np.unique(np.hstack((0, wn)))

    if fence:
        wn = _fence_matsubara_sampling(wn)
    return wn


def _fence_matsubara_sampling(wn):
    # While the condition number for sparse sampling in tau saturates at a
    # modest level, the conditioning in Matsubara steadily deteriorates due
    # to the fact that we are not free to set sampling points continuously.
    # At double precision, tau sampling is better conditioned than iwn
    # by a factor of ~4 (still OK). To battle this, we fence the largest
    # frequency with two carefully chosen oversampling points, which brings
    # the two sampling problems within a factor of 2.
    wn_outer = wn[[0, -1]]
    wn_diff = 2 * np.round(0.025 * wn_outer).astype(int)
    if wn.size >= 20:
        wn = np.hstack([wn, wn_outer - np.sign(wn_outer) * wn_diff])
    if wn.size >= 42:
        wn = np.hstack([wn, wn_outer + np.sign(wn_outer) * wn_diff])
    return np.unique(wn)


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


def _slice_to_size(index):
    if not isinstance(index, slice):
        raise ValueError("argument must be a slice (`n:m`)")
    if index.start is not None and index.start != 0:
        raise ValueError("slice must start at zero")
    if index.step is not None and index.step != 1:
        raise ValueError("slice must step in ones")
    return index.stop
