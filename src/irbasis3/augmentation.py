import numpy as np
from .poly import PiecewiseLegendrePoly
from .basis import _default_matsubara_sampling_points, _default_tau_sampling_points

class LegendreBasis(object):
    """Legendre basis

    In the original paper [L. Boehnke et al., PRB 84, 075145 (2011)],
    they used 
        G(tau) = \sum_{l=0} \sqrt{2l+1} P_l[x(\tau)] G_l/beta,
    where P_l[x] is the $l$-th Legendre polynomial.

    In this class, the basis functions are defined by
        U_l(\tau) \equiv c_l (\sqrt{2l+1}/beta) * P_l[x(\tau)],
    where c_l are additional l-depenent constant factors.
    By default, we take c_l = 1, which reduces to the original definition.
    """
    def __init__(self, statistics, beta, size, _mitigate_sampling_points=True, cl=None):
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
        for l in range(size):
            data[l, 0, l] = np.sqrt((l+0.5)/beta) * cl[l]
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
    
            # Default sampling points
            self.default_tau_sampling_points = np.array([])
            self.default_matsubara_sampling_points = np.array([])
class _ConstTerm:
    def __init__(self, value):
        self.size = 1
        self.value = value

    def __call__(self, n):
        """Return value for given frequencies"""
        return self.value * np.ones_like(n)