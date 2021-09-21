# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

from . import sve


class IRBasis:
    def __init__(self, kernel, eps=None, sve_result=None):
        self.kernel = kernel
        if sve_result is None:
            u, s, v = sve.compute(kernel, eps)
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        even_odd = {'F': 'odd', 'B': 'even'}[kernel.statistics]
        self.u = u
        self.uhat = u.hat(even_odd)
        self.s = s
        self.v = v

    @property
    def lambda_(self):
        return self.kernel.lambda_

    @property
    def statistics(self):
        return self.kernel.statistics

    @property
    def size(self):
        return self.u.size
