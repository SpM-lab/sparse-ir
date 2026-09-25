# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Abstract base class for basis objects.

This module provides the abstract interface that all basis types should implement.
"""

from abc import ABC, abstractmethod

class AbstractKernel(ABC):
    """Abstract base class for kernels.

    It defines no interface of its own: the supported kernels are
    :class:`~sparse_ir.LogisticKernel` and the deprecated
    :class:`~sparse_ir.RegularizedBoseKernel`.
    """
    pass

class AbstractBasis(ABC):
    r"""Abstract base class for bases on the imaginary-time axis.

    This class stores a set of basis functions. We can then expand a two-point
    propagator G(τ), where τ is imaginary time:

    .. math::

        G(\tau) \approx \sum_{l=0}^{L-1} G_l U_l(\tau),

    where :math:`U_l` is the l-th basis function, stored in :py:attr:`u`, and
    the :math:`G_l` are the expansion coefficients.  Similarly, the Fourier
    transform G(iν) at the Matsubara frequency ν = nπ/β, where n is a reduced
    Matsubara frequency, can be expanded as follows:

    .. math::

        G(\mathrm{i}\nu) \approx \sum_{l=0}^{L-1} G_l \hat U_l(\mathrm{i}\nu),

    where :math:`\hat U_l` is the Fourier transform of the l-th basis
    function, stored in :py:attr:`uhat`.  The Fourier transform and its
    inverse are

    .. math::

        G(\mathrm{i}\nu) = \int_0^\beta d\tau\, e^{\mathrm{i}\nu\tau} G(\tau),
        \qquad
        G(\tau) = \frac{1}{\beta} \sum_\nu e^{-\mathrm{i}\nu\tau}
        G(\mathrm{i}\nu).

    Assuming that ``basis`` is an instance of some abstract basis, ``gl`` is
    a vector of expansion coefficients G_l, ``tau`` is some imaginary time and
    ``n`` some reduced frequency, we can write this in the library as
    follows::

        G_tau = basis.u(tau).T @ gl
        G_n = basis.uhat(n).T @ gl
    """

    @property
    @abstractmethod
    def u(self):
        r"""Basis functions on the imaginary time axis.

        Set of basis functions :math:`U_l(\tau)` on the imaginary time (tau)
        axis, where tau is a real number in [-beta, beta].  Negative times
        follow :math:`U_l(\tau) = (-1)^\zeta U_l(\tau + \beta)`, where
        :math:`(-1)^\zeta` is -1 for fermions and +1 for bosons, and the
        endpoints are one-sided limits: ``+0.0`` is 0⁺, ``beta`` is β⁻,
        ``-0.0`` is 0⁻ and ``-beta`` is (-β)⁺.  To get the l-th basis function
        at imaginary time tau of some basis, use::

            ultau = basis.u[l](tau)        # l-th basis function at time tau

        Note that u supports vectorization both over l and tau.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def uhat(self):
        r"""Basis functions on the reduced Matsubara frequency axis.

        Set of basis functions :math:`\hat U_l(\mathrm{i}\nu)` at the
        Matsubara frequencies ν = nπ/β, where the reduced frequency n is an
        integer.  These are related to u by the following Fourier transform:

        .. math::

            \hat U_l(\mathrm{i}\nu) = \int_0^\beta d\tau\,
            e^{\mathrm{i}\nu\tau} U_l(\tau).

        To get the l-th basis function at some reduced frequency n of
        some basis, use::

            uln = basis.uhat[l](n)        # l-th basis function at iν, ν = nπ/β

        Note:
            Instead of the value of the Matsubara frequency, these functions
            expect the reduced frequency n = βν/π, the integer prefactor of
            π/β.  For example, the first few positive fermionic frequencies
            would be specified as [1, 3, 5, 7], and the first bosonic
            frequencies are [0, 2, 4, 6]: n is odd for fermions and even for
            bosons.  It is not the ordinary Matsubara index m, which gives
            n = 2m + ζ (ζ = 1 for fermions, 0 for bosons).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def statistics(self):
        """Quantum statistic ("F" for fermionic, "B" for bosonic)"""
        raise NotImplementedError()

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: basis[:3].
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """Shape of the basis function set"""
        return (self.size,)

    @property
    @abstractmethod
    def size(self):
        """Number of basis functions / singular values"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def significance(self):
        """Significances of the basis functions

        Vector of significance values, one for each basis function. Each
        value is a number between 0 and 1 which is an a-priori bound on the
        (relative) error made by discarding the associated coefficient.
        """
        raise NotImplementedError()

    @property
    def accuracy(self):
        """Accuracy of the basis.

        Upper bound to the relative error of representing a propagator with
        the given number of basis functions (number between 0 and 1).  This
        default returns the smallest significance, ``significance[-1]``;
        :class:`~sparse_ir.FiniteTempBasis` overrides it with S_L/S_0 of the
        first discarded singular value, and the DLR and augmented bases
        return the accuracy of their underlying basis.
        """
        return self.significance[-1]

    @property
    @abstractmethod
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax, or None if not present"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def beta(self):
        """Inverse temperature"""
        raise NotImplementedError()

    @property
    def wmax(self):
        """Real frequency cutoff or None if not present"""
        if self.lambda_ is None or self.beta is None:
            return None
        return self.lambda_ / self.beta

    @abstractmethod
    def default_tau_sampling_points(self, *, npoints=None):
        """Default sampling points on the imaginary time axis

        Parameters
        ----------
        npoints : int, optional
            Minimum number of sampling points to return.
        """
        raise NotImplementedError()

    @abstractmethod
    def default_matsubara_sampling_points(self, *, npoints=None,
                                          positive_only=False):
        """Default sampling points on the imaginary frequency axis

        Parameters
        ----------
        npoints : int, optional
            Minimum number of sampling points to return.
        positive_only : bool
            Only return non-negative frequencies, n >= 0. This is useful if
            the object to be fitted is symmetric in Matsubara frequency,
            G(-iν) = G(iν)*, or, equivalently, real in imaginary time.
        """
        raise NotImplementedError()

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return True