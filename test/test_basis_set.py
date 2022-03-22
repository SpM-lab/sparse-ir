# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

from sparse_ir import FiniteTempBasisSet,\
    TauSampling, MatsubaraSampling, finite_temp_bases
import pytest

def test_consistency(sve_logistic):
    beta = 2
    wmax = 5
    eps = 1e-5

    sve_result = sve_logistic[beta * wmax]
    basis_f, basis_b = finite_temp_bases(beta, wmax, eps, sve_result)
    smpl_tau_f = TauSampling(basis_f)
    smpl_tau_b = TauSampling(basis_b)
    smpl_wn_f = MatsubaraSampling(basis_f)
    smpl_wn_b = MatsubaraSampling(basis_b)

    bs = FiniteTempBasisSet(beta, wmax, eps, sve_result=sve_result)
    np.testing.assert_array_equal(smpl_tau_f.sampling_points, smpl_tau_b.sampling_points)
    np.testing.assert_array_equal(bs.smpl_tau_f.matrix.a, smpl_tau_f.matrix.a)
    np.testing.assert_array_equal(bs.smpl_tau_b.matrix.a, smpl_tau_b.matrix.a)

    np.testing.assert_array_equal(bs.smpl_wn_f.matrix.a, smpl_wn_f.matrix.a)
    np.testing.assert_array_equal(bs.smpl_wn_b.matrix.a, smpl_wn_b.matrix.a)
