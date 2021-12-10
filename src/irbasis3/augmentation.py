import numpy as np
from .poly import PiecewiseLegendrePoly

class LegendreBasis(object):
    """Legendre basis

    G(tau) = \sum_{l=0} \sqrt{2l+1} P_l[x(\tau)] G_l/beta,
    where P_l[x] is the $l$-th Legendre polynomial.

    The basis functions are defined by
        U_l(\tau) \equiv (\sqrt{2l+1}/beta) * P_l[x(\tau)].

    Ref: L. Boehnke et al., PRB 84, 075145 (2011)
    """
    def __init__(self, statistics, beta, size):
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
            data[l, 0, l] = np.sqrt(2*l+1)/beta
        self.u = PiecewiseLegendrePoly(data, knots)

        # self.uhat
        # Hack: See basis.py
        uhat_base = PiecewiseLegendrePoly(np.sqrt(beta) * self.u.data, np.array([-1,1]))
        self.uhat = uhat_base.hat({'F': 'odd', 'B': 'even'}[statistics])

        # self.v
        self.v = None