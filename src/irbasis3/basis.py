# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn

from . import kernel as _kernel
from . import sve


class IRBasis:
    """Intermediate representation (IR) basis in reduced variables.

    For a continuation kernel from real frequencies, ω ∈ [-ωmax, ωmax], to
    imaginary time, τ ∈ [0, β], this class stores the truncated singular
    value expansion or IR basis:

        K(x, y) ≈ sum(u[l](x) * s[l] * v[l](y) for l in range(L))

    The functions are given in reduced variables, `x = 2*τ/β - 1` and
    `y = ω/ωmax`, which scales both sides to the interval `[-1, 1]`.  The
    kernel then only depends on a cutoff parameter `Λ = β * ωmax`.

    Members:
    --------
     - `u`: IR basis functions on the reduced imaginary time (`x`) axis.
     - `s`: singular values of the continuation kernel
     - `v`: IR basis functions on the scaled real frequency (`y`) axis.
     - `uhat`: IR basis functions on the Matsubara frequency axis (`wn`).

    Code example:
    -------------

        # Compute IR basis suitable for β*W <= 42
        import irbasis3
        K = irbasis3.KernelFFlat(lambda_=42)
        basis = irbasis3.IRBasis(K)

        # Assume spectrum is a single pole at x = 0.2, compute G(iw)
        # on the first few Matsubara frequencies
        gl = basis.s * basis.v(0.2)
        giw = gl @ basis.uhat([1, 3, 5, 7])

    See also:
    ---------
     - `FiniteTempBasis`: for a basis directly in time/frequency.
    """
    def __init__(self, kernel, eps=None, sve_result=None):
        self.kernel = kernel
        if sve_result is None:
            u, s, v = sve.compute(kernel, eps)
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        _even_odd = {'F': 'odd', 'B': 'even'}[kernel.statistics]
        self.u = u
        self.uhat = u.hat(_even_odd)
        self.s = s
        self.v = v

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: `basis[:3]`.
        """
        sve_result = self.u[index], self.s[index], self.v[index]
        return self.__class__(self.kernel, sve_result=sve_result)

    @property
    def lambda_(self):
        """Basis cutoff parameter Λ = β * ωmax"""
        return self.kernel.lambda_

    @property
    def statistics(self):
        """Statistics: 'F' for fermions, 'B' for bosons."""
        return self.kernel.statistics

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self.u.size

    @property
    def shape(self):
        return self.u.shape

    @property
    def sve_result(self):
        return self.u, self.s, self.v


class FiniteTempBasis:
    """Intermediate representation (IR) basis for given temperature.

    For a continuation kernel from real frequencies, ω ∈ [-wmax, wmax], to
    imaginary time, τ ∈ [0, beta], this class stores the truncated singular
    value expansion or IR basis:

        K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in range(L))

    This basis is inferred from a reduced form by appropriate scaling of
    the variables.

    Members:
    --------
     - `u`: IR basis functions on the imaginary time axis.
     - `s`: singular values of the continuation kernel
     - `v`: IR basis functions on the real frequency axis.
     - `uhat`: IR basis functions on the Matsubara frequency axis (`wn`).

    Code example:
    -------------

        # Compute IR basis for β = 10, W <= 4.2
        import irbasis3
        K = irbasis3.KernelFFlat(lambda_=42)
        basis = irbasis3.FiniteTempBasis(K, beta=10)

        # Assume spectrum is a single pole at ω = 2.5, compute G(iw)
        # on the first few Matsubara frequencies
        gl = basis.s * basis.v(2.5)
        giw = gl @ basis.uhat([1, 3, 5, 7])
    """
    def __init__(self, kernel, beta, eps=None, sve_result=None):
        self.kernel = kernel
        if sve_result is None:
            u, s, v = sve.compute(kernel, eps)
        else:
            u, s, v = sve_result
            if u.shape != s.shape or s.shape != v.shape:
                raise ValueError("mismatched shapes in SVE")

        self.kernel = kernel
        self.beta = beta

        # The polynomials are scaled to the new variables by transforming the
        # knots according to: tau = beta/2 * (x + 1), w = wmax * y.  Scaling
        # the data is not necessary as the normalization is inferred.
        wmax = self.kernel.lambda_ / self.beta
        self.u = u.__class__(u.data, beta/2 * (u.knots + 1))
        self.v = v.__class__(v.data, wmax * v.knots)

        # HACK: as we don't yet support Fourier transforms on anything but the
        # unit interval, we need to scale the underlying data.  This breaks
        # the correspondence between U.hat and Uhat though.
        _even_odd = {'F': 'odd', 'B': 'even'}[kernel.statistics]
        self.uhat = u.__class__(np.sqrt(beta) * u.data, u.knots).hat(_even_odd)

        # TODO: this is a bit hackish, but at least it should break in a
        # well-defined way if someone tries to add a kernel.
        if isinstance(kernel, _kernel.KernelBFlat):
            self.s = np.sqrt(beta/2 * wmax**3) * s
        else:
            if not isinstance(kernel, _kernel.KernelFFlat):
                warn("Unknown kernel: guessing scaling of singular values...")
            self.s = np.sqrt(beta/2 * wmax) * s

    @property
    def wmax(self):
        return self.kernel.lambda_ / self.beta

    @property
    def statistics(self):
        """Statistics: 'F' for fermions, 'B' for bosons."""
        return self.kernel.statistics

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self.u.size

    @property
    def shape(self):
        return self.u.shape
