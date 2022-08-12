# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import functools
import numpy as np


def ravel_argument(last_dim=False):
    """Wrap function operating on 1-D numpy array to allow arbitrary shapes.

    This decorator allows to write functions which only need to operate over
    one-dimensional (ravelled) arrays.  This often simplifies the "shape logic"
    of the computation.
    """
    return lambda fn: RavelArgumentDecorator(fn, last_dim)


class RavelArgumentDecorator(object):
    def __init__(self, inner, last_dim=False):
        self.instance = None
        self.inner = inner
        self.last_dim = last_dim
        functools.update_wrapper(self, inner)

    def __get__(self, instance, _owner=None):
        self.instance = instance
        return self

    def __call__(self, x):
        x = np.asarray(x)
        if self.instance is None:
            res = self.inner(x.ravel())
        else:
            res = self.inner(self.instance, x.ravel())
        if self.last_dim:
            return res.reshape(res.shape[:-1] + x.shape)
        else:
            return res.reshape(x.shape + res.shape[1:])


def check_reduced_matsubara(n, zeta=None):
    """Checks that ``n`` is a reduced Matsubara frequency.

    Check that the argument is a reduced Matsubara frequency, which is an
    integer obtained by scaling the freqency `w[n]` as follows::

        beta / np.pi * w[n] == 2 * n + zeta

    Note that this means that instead of a fermionic frequency (``zeta == 1``),
    we expect an odd integer, while for a bosonic frequency (``zeta == 0``),
    we expect an even one.  If ``zeta`` is omitted, any one is fine.
    """
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        nfloat = n
        n = nfloat.astype(int)
        if not (n == nfloat).all():
            raise ValueError("reduced frequency n must be integer")
    if zeta is not None:
        if not (n & 1 == zeta).all():
            raise ValueError("n have wrong parity")
    return n


def check_range(x, xmin, xmax):
    """Checks each element is in range [xmin, xmax]"""
    x = np.asarray(x)
    if not (x >= xmin).all():
        raise ValueError(f"Some x violate lower bound {xmin}")
    if not (x <= xmax).all():
        raise ValueError(f"Some x violate upper bound {xmax}")
    return x
