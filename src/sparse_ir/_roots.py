# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
"""
Auxiliary module for root finding routines.
"""
import numpy as np


def find_all(f, xgrid, type='continuous'):
    """Find all roots of function between gridpoints"""
    xgrid = np.asarray(xgrid)
    if xgrid.ndim != 1:
        raise ValueError("grid must be a one-dimensional array")

    # First, extract roots that lie directly on the grid points
    fx = f(xgrid)
    hit = fx == 0
    x_hit = xgrid[hit]

    # Next, find out where the sign changes (sign bit flips) remove the
    # previously found points from consideration (we need to remove both
    # directions for transitions + -> - and - -> +)
    sign_change = np.signbit(fx[:-1]) != np.signbit(fx[1:])
    sign_change &= ~hit[:-1] & ~hit[1:]
    if not sign_change.any():
        return x_hit

    # sign_change[i] being set means that the sign changes from xgrid[i] to
    # xgrid[i+1].  This means a corresponds to those xgrid[i] and b to those
    # xgrid[i+1] where sign_change[i] is set.
    where_a = np.hstack((sign_change, False))
    where_b = np.hstack((False, sign_change))
    a = xgrid[where_a]
    b = xgrid[where_b]
    fa = fx[where_a]
    fb = fx[where_b]

    # Depending on whether we have a discrete or continuous function, do
    # this.
    if type == 'continuous':
        xeps = np.finfo(xgrid.dtype).eps * np.abs(xgrid).max()
        x_bisect = _bisect_cont(f, a, b, fa, fb, xeps)
    elif type == 'discrete':
        x_bisect = _bisect_discr(f, a, b, fa, fb)
    else:
        raise ValueError("invalid type")
    return np.sort(np.hstack([x_hit, x_bisect]))


def _bisect_cont(f, a, b, fa, fb, xeps):
    """Bisect roots already found"""
    while True:
        mid = 0.5 * (a + b)
        fmid = f(mid)
        towards_a = np.signbit(fa) != np.signbit(fmid)
        a = np.where(towards_a, a, mid)
        fa = np.where(towards_a, fa, fmid)
        b = np.where(towards_a, mid, b)
        fb = np.where(towards_a, fmid, fb)
        found = b - a < xeps
        if found.any():
            break

    roots = mid[found]
    if found.all():
        return roots
    more = _bisect_cont(f, a[~found], b[~found], fa[~found], fb[~found], xeps)
    return np.hstack([roots, more])


def _bisect_discr(f, a, b, fa, fb):
    """Bisect roots already found"""
    while True:
        mid = (a + b) // 2
        found = a == mid
        if found.any():
            break

        fmid = f(mid)
        towards_a = np.signbit(fa) != np.signbit(fmid)
        a = np.where(towards_a, a, mid)
        fa = np.where(towards_a, fa, fmid)
        b = np.where(towards_a, mid, b)
        fb = np.where(towards_a, fmid, fb)

    roots = mid[found]
    if found.all():
        return roots
    more = _bisect_discr(f, a[~found], b[~found], fa[~found], fb[~found])
    return np.hstack([roots, more])


def discrete_extrema(f, xgrid):
    """Find extrema of Bessel-like discrete function"""
    fx = f(xgrid)
    absfx = np.abs(fx)

    # Forward differences: where[i] now means that the secant changes sign
    # fx[i+1]. This means that the extremum is STRICTLY between x[i] and
    # x[i+2]
    gx = fx[1:] - fx[:-1]
    sgx = np.signbit(gx)
    where = sgx[:-1] != sgx[1:]
    where_a = np.hstack([where, False, False])
    where_b = np.hstack([False, False, where])

    a = xgrid[where_a]
    b = xgrid[where_b]
    absf_a = absfx[where_a]
    absf_b = absfx[where_b]
    res = [_bisect_discr_extremum(f, *args)
           for args in zip(a, b, absf_a, absf_b)]

    # We consider the outer point to be extremua if there is a decrease
    # in magnitude or a sign change inwards
    sfx = np.signbit(fx)
    if absfx[0] > absfx[1] or sfx[0] != sfx[1]:
        res.insert(0, xgrid[0])
    if absfx[-1] > absfx[-2] or sfx[-1] != sfx[-2]:
        res.append(xgrid[-1])

    return np.array(res)


def _bisect_discr_extremum(f, a, b, absf_a, absf_b):
    """Bisect extremum of f on the set {a+1, ..., b-1}"""
    d = b - a
    if d <= 1:
        return a if absf_a > absf_b else b
    if d == 2:
        return a + 1

    m = (a + b) // 2
    n = m + 1
    absf_m = np.abs(f(m))
    absf_n = np.abs(f(n))
    if absf_m > absf_n:
        return _bisect_discr_extremum(f, a, n, absf_a, absf_n)
    else:
        return _bisect_discr_extremum(f, m, b, absf_m, absf_b)
