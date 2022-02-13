# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from .basis import FiniteTempBasis
from typing import Tuple

def to_discrete_lehmann_rep(
        basis: FiniteTempBasis,
        gl: np.ndarray,
        axis = 0
    )->Tuple[np.ndarray, np.ndarray]:
    """
    Transform data to disrete Lehmann represenation (DLR)

    gl:
        Expansion coefficients in u(tau)

    return:
        Real frequencies (1d array) and transformed data
    """
    fit_mat = - basis.s[:,None] * basis.v(basis.sampling_points_v)
    gl = np.moveaxis(gl, source=axis, destination=0)
    gp = np.linalg.lstsq(fit_mat, gl)[0]
    return basis.sampling_points_v, np.moveaxis(gp, source=0, destination=axis)


def eval_matsubara_from_discrete_lehmann_rep(
        beta: float,
        vsample: np.ndarray,
        g_dlr: Tuple[np.ndarray, np.ndarray],
        axis = 0
    )->np.ndarray:
    """
    Transform disrete Lehmann represenation (DLR) to Matsubara frequencies

    beta:
        Inverse temperature

    vsample:
        Imaginary frequencies

    d_dlr:
        poles (1d array) and coefficients

    return:
        Expansion coefficients in IR
    """
    poles = g_dlr[0]
    coeffs = np.moveaxis(g_dlr[1], source=axis, destination=0)
    iv = 1j*vsample * np.pi/beta
    giv = np.einsum('wp, p...->w...', 1/(iv[:,None] - poles[None,:]), coeffs)
    giv = np.moveaxis(giv, source=0, destination=axis)
    return giv