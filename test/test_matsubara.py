# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from irbasis3.basis import FiniteTempBasis
from irbasis3.kernel import KernelFFlat, KernelBFlat

all_basis_sets = [(kernel, lambda_) for kernel in [KernelFFlat, KernelBFlat] for lambda_ in [1E+1, 1E+2, 1E+4, 1E+5, 1E+6]]
#all_basis_sets = [(kernel, lambda_) for kernel in [KernelBFlat] for lambda_ in [1E+4]]

"""
A pole at omega=pole. Compare analytic results of G(iwn) and numerical results computed by using unl.
"""
@pytest.mark.parametrize("K, lambda_", all_basis_sets)
def test_single_pole(K, lambda_):
    wmax = 1.0
    pole = 0.1 * wmax
    beta = lambda_/wmax

    kernel = K(lambda_)
    basis = FiniteTempBasis(kernel, beta)
    stat = basis.statistics

    if stat == 'F':
        stat_shift = 1
    else:
        stat_shift = 0
    rho_l = basis.v(pole/wmax) * pole**(-kernel.ypower)
    gl = - basis.s * rho_l

    func_G = lambda n: 1/(1J * (2*n+stat_shift)*np.pi/beta - pole)

    # Compute G(iwn) using unl
    matsu_test = np.array([-1, 0, 1, 1E+2, 1E+4, 1E+6, 1E+8, 1E+10, 1E+12], dtype=np.int64)
    prj_w = basis.uhat(2*matsu_test+stat_shift).T
    Giwn_t = prj_w @ gl

    # Compute G(iwn) from analytic expression
    Giwn_ref = func_G(matsu_test)

    magnitude = np.abs(Giwn_ref).max()
    diff = np.abs(Giwn_t - Giwn_ref)

    tol = max(10 * basis.s[-1]/basis.s[0], 1e-10)

    # Absolute error
    assert (diff/magnitude).max() < tol

    # Relative error
    #print("relative: error ", np.abs(diff/Giwn_ref))
    assert np.amax(np.abs(diff/Giwn_ref)) < tol