# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from .basis import FiniteTempBasis
from .sampling import DecomposedMatrix
from typing import Optional

class SparsePoleRepresentation:
    """
    Sparse pole representation (SPR)
    Use a set of poles selected according to the roots of Vl(ω)
    """
    def __init__(
            self, basis: FiniteTempBasis,
            sampling_points: Optional[np.ndarray]=None) -> None:
        self._basis = basis

        self._poles = basis.default_omega_sampling_points() if sampling_points is None\
            else np.asarray(sampling_points)
        self._y_sampling_points = basis.beta*self._poles/basis.wmax

        # Fitting matrix from IR
        weight = basis.kernel.weight_func(self._y_sampling_points)
        fit_mat = np.einsum(
            'l,lp,p->lp',
            -basis.s,
            basis.v(self._poles),
            weight,
            optimize=True
        )
        self.matrix = DecomposedMatrix(fit_mat)

    @property
    def sampling_points(self):
        return self._poles

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
        y = self._basis.beta * self.sampling_points/self._basis.wmax
        return np.einsum('l,p,lp,p...->l...',
            -self._basis.s,
            self._basis.kernel.weight_func(y),
            self._basis.v(self.sampling_points),
            g_spr
        )


    def evaluate_matsubara(
            self,
            g_spr: np.ndarray,
            vsample: np.ndarray,
            axis=0) -> np.ndarray:
        """
        Evaluate on Matsubara frequencies

        vsample:
            Imaginary frequencies

        g_spr:
            SPR coefficients
        """
        coeffs = np.moveaxis(g_spr, source=axis, destination=0)
        iv = 1j*vsample * np.pi/self._basis.beta
        giv = np.einsum(
            'wp, p...->w...', 1/(iv[:, None] - self._poles[None, :]), coeffs)
        giv = np.moveaxis(giv, source=0, destination=axis)
        return giv
