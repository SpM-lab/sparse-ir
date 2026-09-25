# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
High-level Python classes for FiniteTempBasis
"""
from typing import Optional
import numpy as np

from . import _util
from pylibsparseir.core import (
    basis_new,
    basis_get_svals,
    basis_get_u,
    basis_get_v,
    basis_get_uhat,
    basis_get_default_tau_sampling_points,
    basis_get_default_matsubara_sampling_points,
)
from pylibsparseir.constants import (
    SPIR_STATISTICS_FERMIONIC,
    SPIR_STATISTICS_BOSONIC,
)
from .kernel import LogisticKernel
from .abstract import AbstractBasis, AbstractKernel
from .sve import SVEResult
from .poly import (
    PiecewiseLegendrePolyVector,
    PiecewiseLegendrePolyFTVector,
    FunctionSet,
    FunctionSetFT,
)

class FiniteTempBasis(AbstractBasis):
    r"""Intermediate representation (IR) basis for given temperature.

    For the logistic kernel, which continues functions of the real frequency
    ω ∈ [-ωmax, ωmax] to the imaginary time τ ∈ [0, β],

    .. math::

        K(\tau, \omega) = \frac{e^{-\tau\omega}}{1 + e^{-\beta\omega}},

    this class stores the truncated singular value expansion or IR basis:

    .. math::

        K(\tau, \omega) \approx \sum_{l=0}^{L-1} U_l(\tau) S_l V_l(\omega),
        \qquad S_0 \ge S_1 \ge \cdots > 0,

    where the :math:`U_l` are the IR basis functions on the imaginary-time
    axis, stored in :py:attr:`u`, the :math:`S_l` are the singular values,
    stored in :py:attr:`s`, and the :math:`V_l` are the IR basis functions on
    the real-frequency axis, stored in :py:attr:`v`.  Their Matsubara
    transforms :math:`\hat U_l(\mathrm{i}\nu)` are stored in :py:attr:`uhat`.

    - :math:`U_l` is orthonormal on [0, β] and :math:`V_l` on [-ωmax, ωmax].
    - The sign of each pair :math:`U_l, V_l` is fixed by
      :math:`U_l(\beta^-) > 0`.
    - :math:`U_l(\beta - \tau) = (-1)^l U_l(\tau)` and
      :math:`V_l(-\omega) = (-1)^l V_l(\omega)`.
    - Fermionic and bosonic bases share :math:`U_l`, :math:`S_l` and
      :math:`V_l`; only :py:attr:`uhat` differs.
    - The basis keeps :math:`l = 0, \ldots, L-1`, the functions with
      :math:`S_l/S_0 \ge \varepsilon` (``eps``), at most ``max_size`` of
      them; :math:`L` is :py:attr:`size`.
    - :math:`S_l = \sqrt{\beta\omega_\mathrm{max}/2}\, s_l`, where the
      :math:`s_l` are the dimensionless singular values ``sve_result.s`` of
      :class:`~sparse_ir.LogisticKernel` (see :class:`~sparse_ir.SVEResult`).

    A Green's function :math:`G(\tau) = -\langle T_\tau c(\tau) c^\dagger(0)
    \rangle` with the spectral function :math:`A(\omega) = -\frac{1}{\pi}
    \mathrm{Im}\, G^\mathrm{R}(\omega)` is expanded as

    .. math::

        G(\tau) \approx \sum_{l=0}^{L-1} G_l U_l(\tau), \qquad
        G(\mathrm{i}\nu) \approx \sum_{l=0}^{L-1} G_l \hat U_l(\mathrm{i}\nu),
        \qquad G_l = -S_l \int d\omega\, \rho(\omega) V_l(\omega),

    with :math:`\rho(\omega) = A(\omega)` for fermions and
    :math:`\rho(\omega) = A(\omega)/\tanh(\beta\omega/2)` for bosons.  The
    notation is that of
    https://spm-lab.github.io/sparse-ir-doc/src/notation.html.

    Example:
        The following example code assumes the spectral function is a single
        pole at ω = 2.5, :math:`A(\omega) = \delta(\omega - 2.5)`::

            # Compute IR basis for fermions, β = 10 and ωmax = 4.2
            import sparse_ir
            basis = sparse_ir.FiniteTempBasis(statistics='F', beta=10, wmax=4.2)

            # G_l = -S_l V_l(2.5); compute G(iν) = 1/(iν - 2.5) on the first
            # few positive Matsubara frequencies, ν = nπ/β for n = 1, 3, 5, 7
            gl = -basis.s * basis.v(2.5)
            giv = gl @ basis.uhat([1, 3, 5, 7])
    """

    def __init__(
        self,
        statistics: str,
        beta: float,
        wmax: float,
        eps: Optional[float] = None,
        *,
        max_size: Optional[int] = None,
        kernel: Optional[AbstractKernel] = None,
        sve_result: Optional[SVEResult] = None,
    ):
        """
        Initialize finite temperature basis.

        Parameters
        ----------
        statistics : str
            'F' for fermions, 'B' for bosons
        beta : float
            Inverse temperature β > 0
        wmax : float
            Frequency cutoff ωmax > 0: the basis represents spectral functions
            that vanish outside [-wmax, wmax].
        eps : float, optional
            Relative cutoff on the singular values: the basis keeps the
            functions with S_l/S_0 >= eps.  Defaults to machine epsilon
            (~2.2e-16).
        max_size : int, optional
            Maximum basis size. If given, only at most the ``max_size`` most
            significant singular values and associated basis functions are
            retained.  None or -1: no limit.
        kernel : LogisticKernel or RegularizedBoseKernel, optional
            Kernel of the basis; its ``lambda_`` must equal ``beta * wmax``.
            If not given, ``LogisticKernel(beta * wmax)`` is used, the kernel
            for both statistics.  The deprecated RegularizedBoseKernel
            requires ``statistics='B'``.
        sve_result : SVEResult, optional
            Precomputed SVE of the kernel. If not given, the SVE is computed.
        """
        if statistics not in ('F', 'B'):
            raise ValueError(
                f"Invalid statistics: {statistics}, expected 'F' or 'B'")
        if not (np.isfinite(beta) and beta > 0):
            raise ValueError(
                f"inverse temperature beta must be positive and finite, got {beta!r}")
        if not (np.isfinite(wmax) and wmax > 0):
            raise ValueError(
                f"frequency cutoff wmax must be positive and finite, got {wmax!r}")
        if eps is not None and not (np.isfinite(eps) and eps > 0):
            raise ValueError(
                f"accuracy eps must be positive and finite, got {eps!r}")
        if max_size is not None and max_size != -1 and not max_size >= 1:
            raise ValueError(
                f"max_size must be None, -1 or a positive integer, got {max_size!r}")

        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._lambda = beta * wmax

        # Handle eps default
        if eps is None:
            eps = np.finfo(np.float64).eps
        self._eps = eps

        # Handle max_size
        if max_size is None:
            max_size = -1

        # Create or use provided kernel
        if kernel is not None:
            # Backward compatibility: use provided kernel
            _check_kernel(kernel, statistics, self._lambda)
            self._kernel = kernel
        else:
            self._kernel = LogisticKernel(self._lambda)

        # Compute SVE if not provided
        if sve_result is None:
            self._sve = SVEResult(self._kernel, eps)
        else:
            _check_sve_result(sve_result, self._kernel)
            self._sve = sve_result

        # Create basis
        stats_int = (
            SPIR_STATISTICS_FERMIONIC if statistics == 'F'
            else SPIR_STATISTICS_BOSONIC
        )
        self._ptr = basis_new(
            stats_int, self._beta, self._wmax, self._eps,
            self._kernel._ptr, self._sve._ptr, max_size
        )

        u_funcs = FunctionSet(basis_get_u(self._ptr))
        v_funcs = FunctionSet(basis_get_v(self._ptr))
        uhat_funcs = FunctionSetFT(basis_get_uhat(self._ptr),
                                   zeta=1 if statistics == 'F' else 0)

        self._s = basis_get_svals(self._ptr)
        # u_funcs uses [0, beta] as default overlap range
        self._u = PiecewiseLegendrePolyVector(u_funcs, -self._beta, self._beta, 
                                            self._beta, 
                                            default_overlap_range=(0, self._beta))
        # v_funcs uses default range (existing xmin, xmax)
        self._v = PiecewiseLegendrePolyVector(v_funcs, -self._wmax, self._wmax, 
                                            0.0)
        self._uhat = PiecewiseLegendrePolyFTVector(uhat_funcs)

    @property
    def statistics(self):
        """Quantum statistic ('F' for fermionic, 'B' for bosonic)"""
        return self._statistics

    @property
    def beta(self):
        """Inverse temperature β"""
        return self._beta

    @property
    def wmax(self):
        """Real frequency cutoff ωmax"""
        return self._wmax

    @property
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax"""
        return self._lambda

    @property
    def size(self):
        """Basis size L, the number of basis functions and singular values"""
        return self._s.size

    @property
    def s(self):
        r"""Singular values :math:`S_l` of the kernel in physical units.

        They are positive and non-increasing.  For the default
        :class:`~sparse_ir.LogisticKernel`,
        :math:`S_l = \sqrt{\beta\omega_\mathrm{max}/2}\, s_l` in terms of the
        dimensionless singular values :math:`s_l` = ``sve_result.s[l]``.
        """
        if self._s is None:
            self._s = basis_get_svals(self._ptr)
        return self._s

    @property
    def u(self):
        r"""Basis functions on the imaginary time axis.

        ``u[l](tau)`` is :math:`U_l(\tau)`, the l-th basis function at
        imaginary time ``tau``.  The functions are orthonormal on [0, β],
        the default range of ``u.overlap``.

        ``tau`` may lie anywhere in ``[-beta, beta]``; points outside that
        interval raise :class:`ValueError`.  Negative times follow
        :math:`U_l(\tau) = (-1)^\zeta U_l(\tau + \beta)`, where
        :math:`(-1)^\zeta` is -1 for fermions and +1 for bosons.  The
        endpoints are one-sided limits: ``+0.0`` is 0⁺, ``beta`` is β⁻,
        ``-0.0`` is 0⁻, giving :math:`(-1)^\zeta U_l(\beta^-)`, and ``-beta``
        is (-β)⁺, giving :math:`(-1)^\zeta U_l(0^+)`.
        """
        return self._u

    @property
    def v(self):
        r"""Basis functions on the real frequency axis.

        Set of IR basis functions :math:`V_l(\omega)` on the real frequency
        (omega) axis, where omega is a real number in ``[-wmax, wmax]``.  The
        functions are orthonormal on [-ωmax, ωmax], the default range of
        ``v.overlap``.  To get the ``l``-th basis function at real frequency
        ``omega`` of some basis ``basis``, use::

            vlomega = basis.v[l](omega)    # l-th basis function at freq. omega

        Note that ``v`` supports vectorization both over ``l`` and ``omega``.
        In particular, omitting the subscript yields a vector with all basis
        functions, evaluated at that position::

            basis.v(omega) == [basis.v[l](omega) for l in range(basis.size)]

        Similarly, supplying a vector of `omega` points yields a matrix ``A``,
        where ``A[l,n]`` corresponds to the ``l``-th basis function evaluated
        at ``omega[n]``::

            omega = [0.5, 1.0]
            basis.v(omega) == \
                [[basis.v[l](t) for t in omega] for l in range(basis.size)]
        """
        return self._v

    @property
    def uhat(self):
        r"""Basis functions in Matsubara frequency.

        ``uhat[l](n)`` is the Matsubara transform

        .. math::

            \hat U_l(\mathrm{i}\nu) = \int_0^\beta d\tau\,
            e^{\mathrm{i}\nu\tau} U_l(\tau), \qquad \nu = \frac{n\pi}{\beta},

        at the reduced Matsubara frequency ``n``, an integer that is odd for
        fermions and even for bosons (``uhat.zeta`` is its parity ζ).  A
        non-integral or wrong-parity ``n`` raises :class:`ValueError`.  For
        fermions, :math:`\hat U_l` is purely imaginary for even l and real for
        odd l; for bosons it is the other way round.
        """
        return self._uhat

    @property
    def significance(self):
        """Relative significance S_l/S_0 of the basis functions."""
        return self.s / self.s[0]

    @property
    def accuracy(self):
        """Overall truncation error bound.

        S_L/S_0, the first discarded singular value relative to the largest
        one; if the SVE holds no further singular value, that of the last one.
        """
        sve_s = self.sve_result.s
        if sve_s.size > self.size:
            return sve_s[self.size] / sve_s[0]
        return sve_s[-1] / sve_s[0]

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.s.shape

    def default_tau_sampling_points(self, npoints=None, use_positive_taus=True):
        """Get default tau sampling points.

        They are the roots of U_L, the first basis function beyond the basis
        (L = :py:attr:`size`).

        Arguments:
            npoints (int):
                Minimum number of sampling points to return (currently unused).
            use_positive_taus (bool):
                If True (default), fold the points into [0, β) with
                ``np.mod`` and sort them; they lie in (0, β).
                If False, the points are unfolded: they lie in (-β/2, β/2]
                and are symmetric about 0 (for odd L, the point β/2 is its
                own mirror image modulo β).

                .. versionadded:: 1.2
        """
        points = basis_get_default_tau_sampling_points(self._ptr)
        if use_positive_taus:
            points = np.mod(points, self.beta)
            points = np.sort(points)
        return points

    def default_omega_sampling_points(self, npoints=None):
        """Return default sampling points on the real-frequency axis.

        They are the L roots of V_L, the first real-frequency basis function
        beyond the basis, in (-ωmax, ωmax); they are the default poles of
        :class:`~sparse_ir.DiscreteLehmannRepresentation`.

        Arguments:
            npoints (int):
                Minimum number of sampling points to return (currently unused).

                .. versionadded:: 1.1
        """
        from pylibsparseir.core import basis_get_default_omega_sampling_points
        return basis_get_default_omega_sampling_points(self._ptr)

    def default_matsubara_sampling_points(self, npoints=None, positive_only=False):
        """Get default Matsubara sampling points.

        Returns reduced Matsubara frequencies n (ν = nπ/β): the sign changes
        of the first discarded Matsubara basis function Û_l, with l >= L
        chosen to fit the parity.  Bosonic sets always include n = 0.

        Arguments:
            npoints (int):
                Minimum number of sampling points to return (currently unused).
            positive_only (bool):
                If True, return only the non-negative frequencies n >= 0 of
                the set, for ``MatsubaraSampling(..., positive_only=True)``,
                which assumes G(-iν) = G(iν)*.
        """
        return basis_get_default_matsubara_sampling_points(self._ptr, positive_only)

    def __repr__(self):
        return (f"FiniteTempBasis(statistics='{self.statistics}', "
                f"beta={self.beta}, wmax={self.wmax}, size={self.size})")

    def __getitem__(self, index):
        """Truncate the basis to its ``n`` most significant singular values.

        Only ``basis[:n]`` (start 0, unit step, ``0 < n <= size``) is
        supported.  The truncated basis shares the kernel and the SVE of this
        one, so its singular values and functions are the first ``n`` of this
        basis.
        """
        stop = _util.slice_to_size(index, self.size)
        return FiniteTempBasis(self.statistics, self.beta, self.wmax, self._eps,
                               max_size=stop, kernel=self._kernel,
                               sve_result=self._sve)

    @property
    def kernel(self):
        """The kernel used to generate the basis."""
        return self._kernel

    @property
    def sve_result(self):
        """The singular value expansion result.

        Its ``s`` are the dimensionless singular values s_l of the kernel.
        """
        return self._sve

    def rescale(self, new_beta):
        """Return a basis for different temperature.

        Uses the same kernel with the same ``eps``, but a different
        temperature.  Note that this implies a different UV cutoff ``wmax``,
        since ``lambda_ == beta * wmax`` stays constant.
        """
        new_beta = float(new_beta)
        if not new_beta > 0:
            raise ValueError(
                f"inverse temperature must be positive, got {new_beta!r}")

        # lambda_ == beta * wmax is held fixed, so the SVE (which depends only
        # on lambda_ and eps) can be reused as is.
        new_wmax = self._lambda / new_beta
        return FiniteTempBasis(self.statistics, new_beta, new_wmax, self._eps,
                               kernel=self._kernel, sve_result=self._sve)


def _check_kernel(kernel, statistics, lambda_):
    """Check a user-supplied kernel against the statistics and beta * wmax."""
    from .kernel import RegularizedBoseKernel
    if statistics == 'F' and isinstance(kernel, RegularizedBoseKernel):
        raise ValueError(
            "RegularizedBoseKernel is incompatible with fermionic statistics")
    if not np.isclose(kernel.lambda_, lambda_, rtol=1e-12, atol=0):
        raise ValueError(
            f"kernel cutoff lambda_ = {kernel.lambda_!r} does not match "
            f"beta * wmax = {lambda_!r}")


def _check_sve_result(sve_result, kernel):
    """Check that a precomputed SVE belongs to the kernel of the basis.

    The C library builds a basis from a mismatched SVE without complaint, and
    the result is a wrong basis.
    """
    sve_kernel = sve_result._kernel
    if type(sve_kernel) is not type(kernel):
        raise ValueError(
            f"sve_result was computed for a {type(sve_kernel).__name__}, but "
            f"the basis uses a {type(kernel).__name__}")
    if not np.isclose(sve_kernel.lambda_, kernel.lambda_, rtol=1e-12, atol=0):
        raise ValueError(
            f"sve_result was computed for lambda_ = {sve_kernel.lambda_!r}, but "
            f"the basis has beta * wmax = {kernel.lambda_!r}")


def finite_temp_bases(beta, wmax, eps=None, sve_result=None):
    """Construct FiniteTempBasis objects for fermion and bosons

    Construct FiniteTempBasis objects for fermion and bosons with the same
    ``beta``, ``wmax`` and ``eps``.  Both use ``LogisticKernel(beta * wmax)``,
    so they share U_l, S_l and V_l and differ only in ``uhat``.  Each basis
    creates its own kernel instance; the SVE is shared only if
    ``sve_result`` is given, otherwise it is computed for each basis.

    Returns:
        tuple: ``(fermion_basis, boson_basis)``
    """
    fermion_basis = FiniteTempBasis('F', beta, wmax, eps, sve_result=sve_result)
    boson_basis = FiniteTempBasis('B', beta, wmax, eps, sve_result=sve_result)
    return fermion_basis, boson_basis
