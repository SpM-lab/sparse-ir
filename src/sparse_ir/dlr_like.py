# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from .basis import FiniteTempBasis
from .sampling import DecomposedMatrix


class DLRLike:
    """
    Representation like Discrete Lehmann representation (DLR)
    Use a set of poles selected according to the roots of Vl(ω)
    """
    def __init__(self, basis: FiniteTempBasis) -> None:
        self._basis = basis
        roots_ = basis.v[-1].roots()
        self._poles = np.hstack(
            (basis.v.xmin, 0.5 * (roots_[0:-1] + roots_[1:]), basis.v.xmax))

        omega_sampling_points = basis.default_omega_sampling_points()
        y_sampling_points = basis.beta*omega_sampling_points/basis.wmax

        # Fitting matrix from IR
        weight = basis.kernel.weight_func(basis.statistics)(y_sampling_points)
        fit_mat = np.einsum(
            'l,lp,p->lp',
            -basis.s,
            basis.v(omega_sampling_points),
            weight,
            optimize=True
        )
        self.matrix = DecomposedMatrix(fit_mat)

    @property
    def sampling_points(self):
        return self._poles

    def from_IR(self, gl: np.ndarray, axis=0) -> np.ndarray:
        """
        From IR to DLRLike

        gl:
            Expansion coefficients in IR
            It is assumed that KFFlat is used for both of fermion and boson.
        """
        return self.matrix.lstsq(gl, axis)

    def evaluate_matsubara(
            self,
            g_dlr: np.ndarray,
            vsample: np.ndarray,
            axis=0) -> np.ndarray:
        """
        Evaluate on Matsubara frequencies

        vsample:
            Imaginary frequencies

        g_dlr:
            DLRLike coefficients
        """
        coeffs = np.moveaxis(g_dlr, source=axis, destination=0)
        iv = 1j*vsample * np.pi/self._basis.beta
        giv = np.einsum(
            'wp, p...->w...', 1/(iv[:, None] - self._poles[None, :]), coeffs)
        giv = np.moveaxis(giv, source=0, destination=axis)
        return giv
