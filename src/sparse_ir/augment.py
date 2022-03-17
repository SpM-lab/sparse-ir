# Copyright (C) 2020-2021 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import numpy.polynomial.legendre as np_legendre

from .poly import PiecewiseLegendrePoly
from .basis import _default_matsubara_sampling_points


class LegendreBasis(object):
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

        self.statistics = statistics
        self.beta = beta
        self.size = size
        if cl is None:
            cl = np.ones(self.size)
        self.cl = cl

        # self.u
        knots = np.array([0, beta])
        data = np.zeros((size, knots.size-1, size))
        symm = (-1)**np.arange(size)
        for l in range(size):
            data[l, 0, l] = np.sqrt((l+0.5)/beta) * cl[l]
        self.u = PiecewiseLegendrePoly(data, knots, symm=symm)

        # self.uhat
        # Hack: See basis.py
        uhat_base = PiecewiseLegendrePoly(np.sqrt(beta) * self.u.data,
                                          np.array([-1,1]), symm=symm)
        self.uhat = uhat_base.hat({'F': 'odd', 'B': 'even'}[statistics])

        # self.v
        self.v = None

    def default_tau_sampling_points(self):
        return 0.5 * self.beta * (np_legendre.leggauss(self.size)[0] + 1)

    def default_matsubara_sampling_points(self, *, mitigate=True):
        return _default_matsubara_sampling_points(self.uhat, mitigate)

    @property
    def is_well_conditioned(self):
        return True


class MatsubaraConstBasis(object):
    """Constant term in matsubara-frequency domain

    The unity in the matsubara-frequency domain
    """
    def __init__(self, statistics, beta, value=1):
        if statistics not in 'BF':
            raise ValueError("Statistics must be either 'B' for bosonic"
                            "or 'F' for fermionic")
        if not (beta > 0):
            raise ValueError("inverse temperature beta must be positive")

        self.statistics = statistics
        self.beta = beta
        self.size = 1
        self.value = value

        # self.u
        self.u = None

        # self.uhat
        self.uhat = _ConstTerm(value)

        # self.v
        self.v = None

    def default_tau_sampling_points(self):
        return np.array([])

    def default_matsubara_sampling_points(self, *, mitigate=True):
        return np.array([])

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
