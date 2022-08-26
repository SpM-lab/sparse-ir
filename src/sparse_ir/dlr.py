# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np

from . import abstract
from . import kernel
from . import sampling
from . import basis as _basis
from . import _util
from . import svd


class DiscreteLehmannRepresentation(abstract.AbstractBasis):
    """
    Discrete Lehmann representation (DLR) with poles selected according to extrema of IR.

    This class implements a variant of the discrete Lehmann representation (`DLR`_) [1].
    Instead of a truncated singular value expansion of the analytic continuation
    kernel ``K`` like the IR, the discrete Lehmann representation is
    based on a "sketching" of ``K``.  The resulting basis is a linear combination of
    discrete set of poles on the real-frequency axis,
    continued to the imaginary-frequency axis:

         G(iv) == sum(a[i] / (iv - w[i]) for i in range(L))

    Warning
        The poles on the real-frequency axis selected for the DLR are based
        on a rank-revealing decomposition, which offers accuracy guarantees.
        Here, we instead select the pole locations
        based on the zeros of the IR basis functions on the real axis,
        which is a heuristic. We do not expect that difference to matter,
        but please don't blame the DLR authors if we were wrong :-)

    [1]: https://doi.org/10.48550/arXiv.2110.06765
    """
    def __init__(self, basis: _basis.FiniteTempBasis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_omega_sampling_points()
        if not isinstance(basis.kernel, kernel.LogisticKernel):
            raise ValueError("DLR supports only LogisticKernel")

        self._basis = basis
        self._poles = np.asarray(sampling_points)
        self._y_sampling_points = self._poles/basis.wmax

        self._u = TauPoles(basis.statistics, basis.beta, self._poles)
        self._uhat = MatsubaraPoles(basis.statistics, basis.beta, self._poles)

        # Fitting matrix from IR
        F = -basis.s[:, None] * basis.v(self._poles)

        # Now, here we *know* that F is ill-conditioned in very particular way:
        # it is a product A * B * C, where B is well conditioned and A, C are
        # scalings.  This is handled with guaranteed relative accuracy by a
        # Jacobi SVD, implied by the 'accurate' strategy.
        uF, sF, vF = svd.compute(F, strategy='accurate')
        self.matrix = sampling.DecomposedMatrix(F, svd_result=(uF, sF, vF.T))

    @property
    def u(self): return self._u

    @property
    def uhat(self): return self._uhat

    @property
    def statistics(self):
        return self.basis.statistics

    @property
    def sampling_points(self):
        return self._poles

    @property
    def shape(self): return self.size,

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self._poles.size

    @property
    def basis(self) -> _basis.FiniteTempBasis:
        """ Underlying basis """
        return self._basis

    @property
    def lambda_(self):
        return self.basis.lambda_

    @property
    def beta(self):
        return self.basis.beta

    @property
    def wmax(self):
        return self.basis.wmax

    @property
    def significance(self):
        return np.ones(self.shape)

    @property
    def accuracy(self):
        return self.basis.accuracy

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """
        From IR to DLR

        gl:
            Expansion coefficients in IR
        """
        return self.matrix.lstsq(gl, axis)

    def to_IR(self, g_dlr: np.ndarray, axis=0) -> np.ndarray:
        """
        From DLR to IR

        g_dlr:
            Expansion coefficients in DLR
        """
        return self.matrix.matmul(g_dlr, axis)

    def default_tau_sampling_points(self):
        return self.basis.default_tau_sampling_points()

    def default_matsubara_sampling_points(self):
        return self.basis.default_matsubara_sampling_points()

    @property
    def is_well_conditioned(self):
        return False


class MatsubaraPoles:
    def __init__(self, statistics: str, beta: float, poles: np.ndarray):
        self._statistics = statistics
        self._beta = beta
        self._poles = np.array(poles)

    @_util.ravel_argument(last_dim=True)
    def __call__(self, n):
        """Evaluate basis functions at given frequency n"""
        iv = 1j*n * np.pi/self._beta
        if self._statistics == 'F':
            return 1 /(iv[None, :] - self._poles[:, None])
        else:
            return np.tanh(0.5 * self._beta * self._poles)[:, None]\
                /(iv[None, :] - self._poles[:, None])


class TauPoles:
    def __init__(self, statistics: str, beta: float, poles: np.ndarray):
        self._beta = beta
        self._statistics = statistics
        self._poles = np.array(poles)
        self._wmax = np.abs(poles).max()

    @_util.ravel_argument(last_dim=True)
    def __call__(self, tau):
        """ Evaluate basis functions at tau """
        tau = np.asarray(tau)
        if (tau < 0).any() or (tau > self._beta).any():
            raise RuntimeError("tau must be in [0, beta]!")

        x = 2 * tau/self._beta - 1
        y = self._poles/self._wmax
        lambda_ = self._beta * self._wmax

        res = -kernel.LogisticKernel(lambda_)(x[:, None], y[None, :])
        return res.T
