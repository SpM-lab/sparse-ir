# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import warnings
import numpy as np

import scipy.linalg as sp_linalg
import numpy.polynomial.legendre as np_legendre


class Rule:
    """Quadrature rule.

    Approximation of an integral by a weighted sum over discrete points:

         ∫ f(x) * omega(x) * dx ~ sum(f(xi) * wi for (xi, wi) in zip(x, w))

    where we generally have superexponential convergence for smooth ``f(x)``
    with the number of quadrature points.
    """
    def __init__(self, x, w, x_forward=None, x_backward=None, a=-1, b=1):
        x = np.asarray(x)
        if x_forward is None:
            x_forward = x - a
        if x_backward is None:
            x_backward = b - x

        self.x = x
        self.w = np.asarray(w)
        self.x_forward = np.asarray(x_forward)
        self.x_backward = np.asarray(x_backward)
        self.a = a
        self.b = b

    def reseat(self, a, b):
        """Reseat current quadrature rule to new domain"""
        scaling = (b - a) / (self.b - self.a)
        x = (self.x - self.a) * scaling + a
        w = self.w * scaling
        x_forward = self.x_forward * scaling
        x_backward = self.x_backward * scaling
        return Rule(x, w, x_forward, x_backward, a, b)

    def scale(self, factor):
        """Scale weights by factor"""
        return Rule(self.x, self.w * factor, self.x_forward, self.x_backward,
                    self.a, self.b)

    def piecewise(self, edges):
        """Piecewise quadrature with the same quadrature rule, but scaled"""
        edges = np.asarray(edges)
        start = edges[:-1]
        stop = edges[1:]
        if not (stop > start).all():
            raise ValueError("segments ends must be ordered ascendingly")

        return self.join(*(self.reseat(start_i, stop_i)
                         for (start_i, stop_i) in zip(start, stop)))

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        return Rule(self.x.astype(dtype), self.w.astype(dtype),
                    self.x_forward.astype(dtype), self.x_backward.astype(dtype),
                    dtype.type(self.a), dtype.type(self.b))

    @staticmethod
    def join(*gauss_list):
        """Join multiple Gauss quadratures together"""
        if not gauss_list:
            return Rule((), ())

        a = gauss_list[0].a
        b = gauss_list[-1].b
        prev_b = a
        parts = []

        for curr in gauss_list:
            if curr.a != prev_b:
                raise ValueError("Gauss rules must be ascending")
            prev_b = curr.b
            x_forward = curr.x_forward + (curr.a - a)
            x_backward = curr.x_backward + (b - curr.b)
            parts.append((curr.x, curr.w, x_forward, x_backward))

        x, w, x_forward, x_backward = map(np.hstack, zip(*parts))
        return Rule(x, w, x_forward, x_backward, a, b)


def legendre(n, dtype=float):
    """Gauss-Legendre quadrature"""
    return rule_from_recurrence(*_legendre_recurrence(n, dtype))


def legendre_collocation(rule, n=None):
    """Generate collocation matrix from Gauss-Legendre rule"""
    if n is None:
        n = rule.x.size

    res = np_legendre.legvander(rule.x, n - 1).T.copy()
    res *= rule.w

    invnorm = np.arange(0.5, n + 0.5, dtype=rule.x.dtype)
    res *= invnorm[:,None]
    return res


def rule_from_recurrence(alpha, beta, a, b):
    """Make new Gauss scheme based on recurrence coefficients.

    Given a set of polynomials ``P[n]`` defined by the following three-term
    recurrence relation::

        P[0](x)   == 1
        P[1](x)   == x - alpha[0]
        P[n+1](x) == (x - alpha[n]) * P[n] - beta[n] * P[n-1]

    we construct both a set of quadrature points ``x`` and weights ``w`` for
    Gaussian quadrature.  It is usually a good idea to work in extended
    precision for extra acccuracy in the quadrature rule.
    """
    dtype = np.result_type(alpha, beta)

    # First approximation of roots by finding eigenvalues of tridiagonal system
    # corresponding to the recursion
    beta[0] = b - a
    beta_is_pos = beta >= 0
    if not beta_is_pos.all():
        raise NotImplementedError("scipy solver cannot handle complex")

    sqrt_beta = np.sqrt(beta[1:])
    x = sp_linalg.eigvalsh_tridiagonal(alpha, sqrt_beta)
    x = x.astype(dtype)

    # These roots are usually only accurate to 100 ulps or so, so we improve
    # on them using a few iterations of the Newton method.
    prevdiff = 1.0
    maxiter = 5
    for _ in range(maxiter):
        p, dp, _, _ = _polyvalderiv(x, alpha, beta)
        diff = p / dp
        x -= diff

        # check convergence without relying on ATOL
        currdiff = np.abs(diff).max()
        #print(currdiff)
        if not (2 * currdiff <= prevdiff):
            break
        prevdiff = currdiff
    else:
        warnings.warn("Newton iteration did not converge, error = {.2g}"
                      .format(currdiff))

    # Now we know that the weights are proportional to the following:
    _, dp1, p0, _ = _polyvalderiv(x, alpha, beta)
    with np.errstate(over='ignore'):
        w = 1 / (dp1 * p0)
    w *= beta[0] / w.sum(initial=dtype.type(0))
    return Rule(x, w, x - a, b - x, a, b)


def _polyvalderiv(x, alpha, beta):
    """Return value and derivative of polynomial.

    Given a set of polynomials ``P[n]`` defined by a three-term recurrence,
    we evaluate both value and derviative for the highest polynomial and
    the second highest one.
    """
    n = len(alpha)
    p0 = np.ones_like(x)
    p1 = x - alpha[0] * p0
    dp0 = np.zeros_like(x)
    dp1 = p0
    for k in range(1, n):
        x_minus_alpha = x - alpha[k]
        p2 = x_minus_alpha * p1 - beta[k] * p0
        dp2 = p1 + x_minus_alpha * dp1 - beta[k] * dp0
        p0 = p1
        p1 = p2
        dp0 = dp1
        dp1 = dp2

    return p1, dp1, p0, dp0


def _legendre_recurrence(n, dtype=float):
    """Returns the alpha, beta for Gauss-Legendre integration"""
    # The Legendre polynomials are defined by the following recurrence:
    #
    #     (n + 1) * P[n+1](x) == (2 * n + 1) * x * P[n](x) - n * P[n-1](x)
    #
    # To normalize this, we realize that the prefactor of the highest power
    # of P[n] is (2n -1)!! / n!, which we divide by to obtain the "scaled"
    # beta values.
    dtype = np.dtype(dtype)
    k = np.arange(n, dtype=dtype)
    ksq = k**2
    alpha = np.zeros_like(k)
    beta = ksq / (4 * ksq - 1)
    beta[0] = 2
    one = dtype.type(1)
    return alpha, beta, -one, one


class NestedRule(Rule):
    """Nested Gauss quadrature rule."""
    def __init__(self, x, w, v, x_forward=None, x_backward=None, a=-1, b=1):
        super().__init__(x, w, x_forward, x_backward, a, b)
        self.v = np.asarray(v)
        self.vsel = slice(1, None, 2)

    def reseat(self, a, b):
        """Reseat current quadrature rule to new domain"""
        res = super().reseat(a, b)
        new_v = 0.5 * (b - a) * self.v
        return NestedRule(res.x, res.w, new_v, res.x_forward, res.x_backward,
                          res.a, res.b)

    def scale(self, factor):
        """Scale weights by factor"""
        res = super().scale(factor)
        new_v = factor * self.v
        return NestedRule(res.x, res.w, new_v, res.x_forward, res.x_backward,
                          res.a, res.b)

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        res = super().astype(dtype)
        new_v = self.v.astype(dtype)
        return NestedRule(res.x, res.w, new_v, res.x_forward, res.x_backward,
                          res.a, res.b)


def kronrod_21():
    """Gauss-Kronrod rule order 21/10"""
    x = (-0.995657163025808090, -0.973906528517171743, -0.930157491355708244,
         -0.865063366688984536, -0.780817726586416905, -0.679409568299024436,
         -0.562757134668604664, -0.433395394129247213, -0.294392862701460201,
         -0.148874338981631216, +0.000000000000000000, +0.148874338981631216,
         +0.294392862701460201, +0.433395394129247213, +0.562757134668604664,
         +0.679409568299024436, +0.780817726586416905, +0.865063366688984536,
         +0.930157491355708244, +0.973906528517171743, +0.995657163025808090)
    w = (+0.011694638867371874, +0.032558162307964725, +0.054755896574351995,
         +0.075039674810919957, +0.093125454583697601, +0.109387158802297643,
         +0.123491976262065845, +0.134709217311473339, +0.142775938577060085,
         +0.147739104901338486, +0.149445554002916897, +0.147739104901338486,
         +0.142775938577060085, +0.134709217311473339, +0.123491976262065845,
         +0.109387158802297643, +0.093125454583697601, +0.075039674810919957,
         +0.054755896574351995, +0.032558162307964725, +0.011694638867371874)
    v = (+0.066671344308688138, +0.149451349150580587, +0.219086362515982042,
         +0.269266719309996350, +0.295524224714752870, +0.295524224714752870,
         +0.269266719309996350, +0.219086362515982042, +0.149451349150580587,
         +0.066671344308688138)
    return NestedRule(np.array(x), np.array(w), np.array(v))
