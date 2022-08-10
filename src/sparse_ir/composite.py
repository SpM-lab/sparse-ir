# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np


class _AbstractCompositeBasisFunction:
    """Union of several basis functions"""
    def __init__(self, polys):
        self._polys = tuple(polys)
        self._sizes = np.array([p.size for p in self._polys])
        self._cumsum_sizes = np.cumsum(self._sizes)
        self._size = np.sum(self._sizes)

    def __getitem__(self, l):
        """Return part of a set of basis functions"""
        if isinstance(l, int) or issubclass(type(l), np.integer):
            idx_p, l_ = self._internal_pos(l)
            return self._polys[idx_p][l_]
        else:
            raise ValueError(f"Unsupported type of l {type(l)}!")

    def _internal_pos(self, l):
        idx_p = np.searchsorted(self._cumsum_sizes, l, "right")
        l_ = l - self._cumsum_sizes[idx_p-1] if idx_p >= 1 else l
        return idx_p, l_

    @property
    def shape(self): return (self.size,)

    @property
    def size(self): return self._size


class CompositeBasisFunction(_AbstractCompositeBasisFunction):
    """Union of several basis functions for the imaginary-time/real-frequency
    domains"""
    def __init__(self, polys):
        """Initialize CompositeBasisFunction

        Arguments:
        ----------
         - polys: iterable object of basis-function-like instances
        """
        super().__init__(polys)

    def __call__(self, x):
        """Evaluate basis function at position x"""
        return np.vstack([p(x) for p in self._polys])

    def value(self, l, x):
        """Return value for l and x."""
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return np.squeeze(np.asarray([self[ll](xx) for ll, xx in zip(l, x)]))

    def overlap(self, f, axis=None, _deg=None):
        r"""Evaluate overlap integral of this basis function with function `f`"""
        return np.vstack((p.overlap(f, axis, _deg) for p in self._polys))

    def deriv(self, n=1):
        """Get the n'th derivative"""
        return np.vstack([p.deriv(n) for p in self._polys])

    #def roots(self, alpha=2):
        #"""Find all roots of the basis function """
        #return np.unique(np.hstack([p.roots(alpha) for p in self._polys]))


class CompositeBasisFunctionFT(_AbstractCompositeBasisFunction):
    """Union of fourier transform of several basis functions for the
    Matsubara domain"""
    def __init__(self, polys):
        """Initialize CompositeBasisFunctionFT

        Arguments:
        ----------
         - polys: iterable object of basis-function-like instances
        """
        super().__init__(polys)

    def __call__(self, n):
        """Obtain Fourier transform of basis function for given frequencies"""
        return np.vstack([p(n) for p in self._polys])


class CompositeBasis:
    """Union of several basis sets"""
    def __init__(self, bases):
        """Initialize a composite basis

        Args:
            bases): iterable object of FiniteTempBasis instances
        """
        if not all(b.statistics == bases[0].statistics for b in bases):
            raise ValueError("All bases must have the same statistics!")
        if not all(b.beta == bases[0].beta for b in bases):
            raise ValueError("All bases must have the same beta!")

        self._beta = bases[0].beta
        self._size = np.sum([b.size for b in bases])
        self.bases = bases
        self.u = CompositeBasisFunction([b.u for b in bases]) \
                    if all(hasattr(b, 'u') for b in bases) else None
        self.v = CompositeBasisFunction([b.v for b in bases]) \
                    if all(hasattr(b, 'v') for b in bases) else None
        self.uhat = CompositeBasisFunctionFT([b.uhat for b in bases]) \
                    if all(hasattr(b, 'uhat')  for b in bases) else None

    @property
    def beta(self): return self._beta

    @property
    def size(self): return self._size

    @property
    def shape(self): return (self._size,)

    def default_tau_sampling_points(self):
        return np.unique(np.hstack(
                    [b.default_tau_sampling_points() for b in self.bases]))

    def default_matsubara_sampling_points(self, *, mitigate=True):
        return np.unique(np.hstack(
                    [b.default_matsubara_sampling_points(mitigate=mitigate)
                     for b in self.bases]))

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return False
