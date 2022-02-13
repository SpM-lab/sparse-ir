# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from .basis import FiniteTempBasis

scaling_func = {
        "F": lambda beta_omega: np.ones_like(beta_omega),
        "B": lambda beta_omega: 1/np.tanh(0.5*beta_omega)
    }

class DLR:
    """
    Discrete Lehmann representation (DLR)
    """
    def __init__(self, basis: FiniteTempBasis) -> None:
        self._basis = basis
        roots_ = basis.v[-1].roots()
        self._poles = np.hstack((basis.v.xmin, 0.5 * (roots_[0:-1] + roots_[1:]), basis.v.xmax))
        assert np.abs(self._poles).max() <= basis.wmax

    @property
    def poles(self):
        return self._poles

    def from_IR(self, gl: np.ndarray, axis=0)->np.ndarray:
        """
        From IR to DLR

        gl:
            Expansion coefficients in IR
        """
        basis = self._basis
        fit_mat = np.einsum(
            'l,lp,p->lp',
            -basis.s,
            basis.v(basis.sampling_points_v),
            scaling_func[basis.statistics](basis.beta*basis.sampling_points_v),
            optimize=True
        )
        gl = np.moveaxis(gl, source=axis, destination=0)
        gp = np.linalg.lstsq(fit_mat, gl)[0]
        return np.moveaxis(gp, source=0, destination=axis)


    def evaluate_matsubara(
            self,
            g_dlr: np.ndarray,
            vsample: np.ndarray,
            axis = 0
        )->np.ndarray:
        """
        Evaluate on Matsubara frequencies

        vsample:
            Imaginary frequencies

        g_dlr:
            DLR coefficients
        """
        coeffs = np.moveaxis(g_dlr, source=axis, destination=0)
        iv = 1j*vsample * np.pi/self._basis.beta
        giv = np.einsum('wp, p...->w...', 1/(iv[:,None] - self._poles[None,:]), coeffs)
        giv = np.moveaxis(giv, source=0, destination=axis)
        return giv