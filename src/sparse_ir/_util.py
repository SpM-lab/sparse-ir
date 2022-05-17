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
