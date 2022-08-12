# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np

from .kernel import LogisticKernel
from . import sampling
from . import basis as _basis
from . import _util
from . import svd


class MatsubaraPoleBasis:
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


class TauPoleBasis:
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

        res = -LogisticKernel(lambda_)(x[:, None], y[None, :])
        return res.T


class SparsePoleRepresentation:
    """
    Sparse pole representation (SPR)
    The poles are the extrema of V'_{L-1}(ω)
    """
    def __init__(self, basis: _basis.FiniteTempBasis, sampling_points=None):
        if sampling_points is None:
            sampling_points = basis.default_omega_sampling_points()
        if not isinstance(basis.kernel, LogisticKernel):
            raise ValueError("SPR supports only LogisticKernel")

        self._basis = basis
        self._poles = np.asarray(sampling_points)
        self._y_sampling_points = self._poles/basis.wmax

        self.u = TauPoleBasis(basis.statistics, basis.beta, self._poles)
        self.uhat = MatsubaraPoleBasis(basis.statistics, basis.beta, self._poles)

        # Fitting matrix from IR
        F = -basis.s[:, None] * basis.v(self._poles)

        # Now, here we *know* that F is ill-conditioned in very particular way:
        # it is a product A * B * C, where B is well conditioned and A, C are
        # scalings.  This is handled with guaranteed relative accuracy by a
        # Jacobi SVD, implied by the 'accurate' strategy.
        uF, sF, vF = svd.compute(F, strategy='accurate')
        self.matrix = sampling.DecomposedMatrix(F, svd_result=(uF, sF, vF.T))

    @property
    def statistics(self):
        return self.basis.statistics

    @property
    def sampling_points(self):
        return self._poles

    @property
    def size(self):
        """Number of basis functions / singular values."""
        return self._poles.size

    @property
    def basis(self) -> _basis.FiniteTempBasis:
        """ Underlying basis """
        return self._basis

    @property
    def beta(self) -> float:
        """Inverse temperature (this is `None` because unscaled basis)"""
        return self.basis.beta

    @property
    def wmax(self) -> float:
        """Frequency cutoff (this is `None` because unscaled basis)"""
        return self.basis.wmax

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """
        From IR to SPR

        gl:
            Expansion coefficients in IR
        """
        return self.matrix.lstsq(gl, axis)

    def to_IR(self, g_spr: np.ndarray, axis=0) -> np.ndarray:
        """
        From SPR to IR

        g_spr:
            Expansion coefficients in SPR
        """
        return self.matrix.matmul(g_spr, axis)

    def default_tau_sampling_points(self):
        """Default sampling points on the imaginary time/x axis"""
        return self.basis.default_tau_sampling_points()

    def default_matsubara_sampling_points(self):
        """Default sampling points on the imaginary frequency axis"""
        return self.basis.default_matsubara_sampling_points()

    @property
    def is_well_conditioned(self):
        """Returns True if the sampling is expected to be well-conditioned"""
        return False
