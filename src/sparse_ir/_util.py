# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import functools
import numpy as np


def ravel_argument(last_dim=False):
    """Guarantees to pass a ravelled array into the method"""
    def ravel_argument_decorator(inner_fn):
        @functools.wraps(inner_fn)
        def ravelled_func(self, x):
            x = np.asarray(x)
            x_flat = x.ravel()
            res = inner_fn(self, x_flat)
            if last_dim:
                return res.reshape(res.shape[:-1] + x.shape)
            else:
                return res.reshape(x.shape + res.shape[1:])

        return ravelled_func
    return ravel_argument_decorator
