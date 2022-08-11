# Copyright (C) 2020-2021 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import numpy.polynomial.legendre as np_legendre

from .abstract import AbstractBasis
from .poly import PiecewiseLegendreFT, PiecewiseLegendrePoly
from .basis import _default_matsubara_sampling_points


class LegendreBasis(AbstractBasis):
    r"""Legendre basis

    In the original paper [L. Boehnke et al., PRB 84, 075145 (2011)],
    they used:

        G(\tau) = \sum_{l=0} \sqrt{2l+1} P_l[x(\tau)] G_l/beta,

    where P_l[x] is the $l$-th Legendre polynomial.

    In this class, the basis functions are defined by

        U_l(\tau) \equiv c_l (\sqrt{2l+1}/beta) * P_l[x(\tau)],

    where c_l are additional l-depenent constant factors.
    By default, we take c_l = 1, which reduces to the original definition.
    """
    def __init__(self, statistics, beta, size, cl=None):
        if statistics not in 'BF':
            raise ValueError("Statistics must be either 'B' for bosonic"
                            "or 'F' for fermionic")
        if not (beta > 0):
            raise ValueError("inverse temperature beta must be positive")
        if not (size > 0):
            raise ValueError("size of basis must be positive")

        self._statistics = statistics
        self._beta = beta
        self._size = size
        if cl is None:
            cl = np.ones(self.size)
        self._cl = cl

        # self.u
        knots = np.array([0, beta])
        data = np.zeros((size, knots.size-1, size))
        symm = (-1)**np.arange(size)
        for l in range(size):
            data[l, 0, l] = np.sqrt((l+0.5)/beta) * cl[l]
        self._u = PiecewiseLegendrePoly(data, knots, symm=symm)

        # self.uhat
        # Hack: See basis.py
        uhat_base = PiecewiseLegendrePoly(np.sqrt(beta) * self.u.data,
                                          np.array([-1,1]), symm=symm)
        odd_even = {'F': 'odd', 'B': 'even'}[statistics]
        self._uhat = PiecewiseLegendreFT(uhat_base, odd_even)

    def significance(self):
        return np.ones(self.size)

    def default_tau_sampling_points(self):
        return 0.5 * self.beta * (np_legendre.leggauss(self.size)[0] + 1)

    def default_matsubara_sampling_points(self):
        return _default_matsubara_sampling_points(self.uhat, self.size)

    @property
    def statistics(self): return self._statistics

    @property
    def u(self): return self._u

    @property
    def uhat(self): return self._uhat

    @property
    def shape(self): return self._size,

    @property
    def size(self): return self._size

    @property
    def beta(self): return self._beta

    @property
    def is_well_conditioned(self):
        return True


class MatsubaraConstBasis(AbstractBasis):
    """Constant term in matsubara-frequency domain

    The unity in the matsubara-frequency domain
    """
    def __init__(self, statistics, beta, value=1):
        if statistics not in 'BF':
            raise ValueError("Statistics must be either 'B' for bosonic"
                            "or 'F' for fermionic")
        if not (beta > 0):
            raise ValueError("inverse temperature beta must be positive")

        self._statistics = statistics
        self._beta = beta
        self._value = value

    def significance(self):
        return np.ones(1)

    def default_tau_sampling_points(self):
        return np.array([])

    def default_matsubara_sampling_points(self):
        return np.array([])

    @property
    def statistics(self): return self._statistics

    @property
    def u(self): return _ConstTerm(np.nan)

    @property
    def uhat(self): return _ConstTerm(self._value)

    @property
    def shape(self): return 1,

    @property
    def size(self): return 1

    @property
    def beta(self): return self._beta

    @property
    def is_well_conditioned(self):
        return True


class _ConstTerm:
    def __init__(self, value):
        self.size = 1
        self.value = value

    def __call__(self, n):
        """Return value for given frequencies"""
        return (self.value * np.ones_like(n))[None,:]
