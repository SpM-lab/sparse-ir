# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from irbasis3 import adapter

all_stat_lambda = [(stat, lambda_) for stat in ['F', 'B'] for lambda_ in [1E+1, 1E+2, 1E+4, 1E+6]]

"""
A pole at omega=pole. Compare analytic results of G(iwn) and numerical results computed by using unl.
"""
@pytest.mark.parametrize("stat, lambda_", all_stat_lambda)
def test_single_pole(stat, lambda_):
    wmax = 1.0
    pole = 0.1 * wmax
    beta = lambda_/wmax

    basis = adapter.load(stat, lambda_)
    dim = basis.dim()

    if stat == 'F':
        rho_l = np.sqrt(1/wmax)* np.array([basis.vly(l, pole/wmax) for l in range(dim)])
        Sl = np.sqrt(0.5 * beta * wmax) * np.array([basis.sl(l) for l in range(dim)])
        stat_shift = 1
    else:
        rho_l = np.sqrt(1/wmax)* np.array([basis.vly(l, pole/wmax) for l in range(dim)])/pole
        Sl = np.sqrt(0.5 * beta * wmax**3) * np.array([basis.sl(l) for l in range(dim)])
        stat_shift = 0

    gl = - Sl * rho_l

    func_G = lambda n: 1/(1J * (2*n+stat_shift)*np.pi/beta - pole)

    # Compute G(iwn) using unl
    matsu_test = np.array([-1, 0, 1, 1E+2, 1E+4, 1E+6, 1E+8], dtype=np.int64)
    # This fails. There seems to be some loss of precision at high frequencies
    #matsu_test = np.array([1E+14], dtype=np.int64)
    Uwnl_plt = np.sqrt(beta) * basis.compute_unl(matsu_test)
    Giwn_t = Uwnl_plt @ gl

    # Compute G(iwn) from analytic expression
    Giwn_ref = func_G(matsu_test)

    magnitude = np.abs(Giwn_ref).max()
    diff = np.abs(Giwn_t - Giwn_ref)

    # Absolute error
    tol = 10 * Sl[-1]/Sl[0]
    assert (diff/magnitude).max() < tol

    # Relative error
    assert np.amax(np.abs(diff/Giwn_ref)) < tol