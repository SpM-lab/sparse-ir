# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np

from sparse_ir import FiniteTempBasisSet,\
    TauSampling, MatsubaraSampling, finite_temp_bases


def test_consistency():
    beta = 2
    wmax = 5
    eps = 1e-5

    basis_f, basis_b = finite_temp_bases(beta, wmax, eps)
    smpl_tau_f = TauSampling(basis_f)
    smpl_tau_b = TauSampling(basis_b)
    smpl_wn_f = MatsubaraSampling(basis_f)
    smpl_wn_b = MatsubaraSampling(basis_b)

    bs = FiniteTempBasisSet(beta, wmax, eps)
    np.testing.assert_array_equal(smpl_tau_f.sampling_points, smpl_tau_b.sampling_points)
    np.testing.assert_array_equal(bs.smpl_tau_f.tau, smpl_tau_f.tau)
    np.testing.assert_array_equal(bs.smpl_tau_b.tau, smpl_tau_b.tau)

    np.testing.assert_array_equal(bs.smpl_wn_f.wn, smpl_wn_f.wn)
    np.testing.assert_array_equal(bs.smpl_wn_b.wn, smpl_wn_b.wn)
