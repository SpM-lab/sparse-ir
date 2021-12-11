import numpy as np
from .poly import PiecewiseLegendrePoly
from .basis import _default_matsubara_sampling_points, _default_tau_sampling_points

class LegendreBasis(object):
    """Legendre basis

    G(tau) = \sum_{l=0} \sqrt{2l+1} P_l[x(\tau)] G_l/beta,
    where P_l[x] is the $l$-th Legendre polynomial.

    The basis functions are defined by
        U_l(\tau) \equiv (\sqrt{2l+1}/beta) * P_l[x(\tau)].

    Ref: L. Boehnke et al., PRB 84, 075145 (2011)
    """
    def __init__(self, statistics, beta, size, _mitigate_sampling_points=True):
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

        # self.u
        knots = np.array([0, beta])
        data = np.zeros((size, knots.size-1, size))
        for l in range(size):
            data[l, 0, l] = np.sqrt((l+0.5)/beta)
        self.u = PiecewiseLegendrePoly(data, knots)

        # self.uhat
        # Hack: See basis.py
        uhat_base = PiecewiseLegendrePoly(np.sqrt(beta) * self.u.data, np.array([-1,1]))
        self.uhat = uhat_base.hat({'F': 'odd', 'B': 'even'}[statistics])

        # self.v
        self.v = None

        # Default sampling points
        self.default_tau_sampling_points = 0.5 * beta * (np.polynomial.legendre.leggauss(size)[0]+1)
        assert self.default_tau_sampling_points.size == size
        self.default_matsubara_sampling_points = _default_matsubara_sampling_points(
            self.uhat, _mitigate_sampling_points)
            #self.default_matsubara_sampling_points = np.unique(np.hstack((0, self.default_matsubara_sampling_points)))

class MatsubaraConstBasis(object):
       """Constant term in matsubara-frequency domain

       The unity in the matsubara-frequency domain
       """
       def __init__(self, statistics, beta):
            if statistics not in 'BF':
                raise ValueError("Statistics must be either 'B' for bosonic"
                               "or 'F' for fermionic")
            if not (beta > 0):
                raise ValueError("inverse temperature beta must be positive")
  
            self.statistics = statistics
            self.beta = beta
            self.size = 1
    
            # self.u
            self.u = None
    
            # self.uhat
            self.uhat = _ConstTerm()
    
            # self.v
            self.v = None
    
            # Default sampling points
            self.default_tau_sampling_points = np.array([])
            self.default_matsubara_sampling_points = np.array([])
class _ConstTerm:
    def __init__(self):
        self.size = 1

    def __call__(self, n):
        """Return 1 for given frequencies"""
        return np.ones_like(n)