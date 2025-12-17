# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
High-level Python classes for FiniteTempBasis
"""
from typing import Optional
import numpy as np
from pylibsparseir.core import basis_new, basis_get_size, basis_get_svals, basis_get_u, basis_get_v, basis_get_uhat, basis_get_default_tau_sampling_points, basis_get_default_omega_sampling_points, basis_get_default_matsubara_sampling_points
from pylibsparseir.constants import SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC
from .kernel import LogisticKernel
from .abstract import AbstractBasis
from .sve import SVEResult
from .poly import PiecewiseLegendrePolyVector, PiecewiseLegendrePolyFTVector, FunctionSet, FunctionSetFT

class FiniteTempBasis(AbstractBasis):
    r"""Intermediate representation (IR) basis for given temperature.

    For a continuation kernel from real frequencies, `ω` ∈ [-ωmax, ωmax], to
    imaginary time, `τ` ∈ [0, beta], this class stores the truncated singular
    value expansion or IR basis:

    .. math::

        K(\tau, \omega) \approx \sum_{l=0}^{L-1} U_l(\tau) S_l V_l(\omega),

    where `U` are the IR basis functions on the imaginary-time axis, stored
    in :py:attr:`u`, `S` are the singular values, stored in :py:attr:`s`,
    and `V` are the IR basis functions on the real-frequency axis, stored
    in :py:attr:`V`.  The IR basis functions in Matsubara frequency are
    stored in :py:attr:`uhat`.

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

    def __init__(self, statistics: str, beta: float, wmax: float, eps: float = np.finfo(np.float64).eps, sve_result: Optional[SVEResult] = None, max_size: int =-1):
        """
        Initialize finite temperature basis.

        Parameters:
        -----------
        statistics : str
            'F' for fermions, 'B' for bosons
        beta : float
            Inverse temperature
        wmax : float
            Frequency cutoff
        eps : float
            Relative truncation threshold for the singular values,
            defaulting to the machine epsilon (2.2e-16)
        """
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._lambda = beta * wmax
        self._eps = eps

        # Create kernel
        if statistics == 'F' or statistics == 'B':
            self._kernel = LogisticKernel(self._lambda)
        else:
            raise ValueError(f"Invalid statistics: {statistics} expected 'F' or 'B'")

        # Compute SVE
        if sve_result is None:
            self._sve = SVEResult(self._kernel, eps)
        else:
            self._sve = sve_result

        # Create basis
        stats_int = SPIR_STATISTICS_FERMIONIC if statistics == 'F' else SPIR_STATISTICS_BOSONIC
        self._ptr = basis_new(stats_int, self._beta, self._wmax, self._eps, self._kernel._ptr, self._sve._ptr, max_size)

        u_funcs = FunctionSet(basis_get_u(self._ptr))
        v_funcs = FunctionSet(basis_get_v(self._ptr))
        uhat_funcs = FunctionSetFT(basis_get_uhat(self._ptr))

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
        """Inverse temperature"""
        return self._beta

    @property
    def wmax(self):
        """Real frequency cutoff"""
        return self._wmax

    @property
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax"""
        return self._lambda

    @property
    def size(self):
        return self._s.size

    @property
    def s(self):
        """Vector of singular values of the continuation kernel"""
        if self._s is None:
            self._s = basis_get_svals(self._ptr)
        return self._s

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        r"""Basis functions on the real frequency axis.

        Set of IR basis functions on the real frequency (omega) axis, where
        omega is a real number of magnitude less than :py:attr:`wmax`.  To get
        the ``l``-th basis function at real frequency ``omega`` of some basis
        ``basis``, use::

            ulomega = basis.v[l](omega)    # l-th basis function at freq. omega

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
        return self._uhat

    @property
    def significance(self):
        """Relative significance of basis functions."""
        return self.s / self.s[0]

    @property
    def accuracy(self):
        """Overall accuracy bound."""
        return self.s[-1] / self.s[0]

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.s.shape

    def default_tau_sampling_points(self, npoints=None, use_positive_taus=True):
        """Get default tau sampling points.
        
        Arguments:
            npoints (int):
                Minimum number of sampling points to return (currently unused).
            use_positive_taus (bool):
                If True, fold points to [0, β] range and sort them (default: True).
                If False, points are in symmetric range around β/2.
                
                .. versionadded:: 1.2
        """
        points = basis_get_default_tau_sampling_points(self._ptr)
        if use_positive_taus:
            points = np.mod(points, self.beta)
            points = np.sort(points)
        return points

    def default_omega_sampling_points(self, npoints=None):
        """Return default sampling points in imaginary time.

        Arguments:
            npoints (int):
                Minimum number of sampling points to return.

                .. versionadded:: 1.1
        """
        from pylibsparseir.core import basis_get_default_omega_sampling_points
        return basis_get_default_omega_sampling_points(self._ptr)

    def default_matsubara_sampling_points(self, npoints=None, positive_only=False):
        """Get default Matsubara sampling points."""
        return basis_get_default_matsubara_sampling_points(self._ptr, positive_only)

    def __repr__(self):
        return (f"FiniteTempBasis(statistics='{self.statistics}', "
                f"beta={self.beta}, wmax={self.wmax}, size={self.size})")

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: basis[:3].
        """
        # TODO: Implement basis truncation when C API supports it
        raise NotImplementedError("Basis truncation not yet implemented in C API")

    @property
    def kernel(self):
        """The kernel used to generate the basis."""
        return self._kernel

    @property
    def sve_result(self):
        """The singular value expansion result."""
        return self._sve

    def rescale(self, new_beta):
        """Return a basis for different temperature.

        Uses the same kernel with the same ``eps``, but a different
        temperature.  Note that this implies a different UV cutoff ``wmax``,
        since ``lambda_ == beta * wmax`` stays constant.
        """
        # Calculate new beta and wmax that give the desired lambda
        # We keep the ratio beta/wmax constant
        ratio = self.beta / self.wmax
        new_wmax = np.sqrt(new_lambda / ratio)
        new_beta = new_lambda / new_wmax

        # Get epsilon from the current basis accuracy
        eps = self.accuracy

        return FiniteTempBasis(self.statistics, new_beta, new_wmax, eps)


def finite_temp_bases(beta, wmax, eps=None, sve_result=None):
    """Construct FiniteTempBasis objects for fermion and bosons

    Construct FiniteTempBasis objects for fermion and bosons using
    the same LogisticKernel instance.
    """
    fermion_basis = FiniteTempBasis('F', beta, wmax, eps, sve_result=sve_result)
    boson_basis = FiniteTempBasis('B', beta, wmax, eps, sve_result=sve_result)
    return fermion_basis, boson_basis