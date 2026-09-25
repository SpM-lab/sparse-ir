# Copyright (C) 2020-2026 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""IR <-> DLR <-> tau <-> Matsubara consistency on multi-dimensional data.

Port of SparseIR.jl's ``test/spir/integration_tests.jl``: random DLR
coefficients are converted to IR coefficients and evaluated on the default
tau and Matsubara points, both through the IR basis and directly through
the DLR, and the two must agree to ``10 * eps`` relative to the largest
value.  The data have extra dimensions of unequal length (2, 3, 4) and the
basis axis is moved to every position, so a transposition bug cannot pass.
"""
import numpy as np
import pytest

import sparse_ir
from sparse_ir import DiscreteLehmannRepresentation, MatsubaraSampling, TauSampling
from ._helpers import assert_close

BETA, WMAX, EPS = 1e4, 2.0, 1e-10
TOL = 10 * EPS

# (statistics, positive_only, dtype, extra_dims, axis).  positive_only asserts
# real coefficients, so it is combined with real data only.
CASES = (
    [(s, po, np.float64, (), 0) for s in "FB" for po in (False, True)]
    + [(s, False, np.complex128, (), 0) for s in "FB"]
    + [(s, po, np.float64, (2, 3, 4), ax)
       for s in "FB" for po in (False, True) for ax in (0, 1, 2, 3, -1)]
    + [(s, False, np.complex128, (2, 3, 4), ax) for s in "FB" for ax in (0, 3)]
)


def _case_id(case):
    s, po, dtype, extra, ax = case
    return f"{s}-po{int(po)}-{np.dtype(dtype).name}-nd{len(extra) + 1}-axis{ax}"


def _random_dlr_coefficients(rng, poles, dtype, extra, axis):
    shape = (poles.size,) + extra
    scale = np.sqrt(np.abs(poles)).reshape((-1,) + (1,) * len(extra))
    coeffs = (2 * rng.random(shape) - 1) * scale
    if np.issubdtype(dtype, np.complexfloating):
        coeffs = coeffs + 1j * (2 * rng.random(shape) - 1) * scale
    return np.moveaxis(coeffs, 0, axis)


def _assert_rel(a, b, what):
    assert_close(a, b, TOL * np.abs(a).max(), what)


@pytest.fixture(scope="module")
def setups(get_basis):
    """Samplings shared by all cases with the same statistics and positive_only.

    Building a Matsubara sampling for this basis takes about a second, so it
    is done once per combination rather than once per case.
    """
    cache = {}

    def get(stat, positive_only):
        if (stat, positive_only) not in cache:
            basis = get_basis(stat, BETA, WMAX, EPS)
            tau_points = basis.default_tau_sampling_points()
            matsu_points = basis.default_matsubara_sampling_points(
                positive_only=positive_only)
            dlr = DiscreteLehmannRepresentation(basis)
            cache[stat, positive_only] = {
                "basis": basis,
                "dlr": dlr,
                "tau": TauSampling(basis, tau_points),
                "matsu": MatsubaraSampling(basis, matsu_points,
                                           positive_only=positive_only),
                "tau_dlr": TauSampling(dlr, tau_points),
                "matsu_dlr": MatsubaraSampling(dlr, matsu_points,
                                               positive_only=positive_only),
            }
        return cache[stat, positive_only]

    return get


@pytest.mark.parametrize("case", CASES, ids=[_case_id(c) for c in CASES])
def test_ir_dlr_tau_matsubara_agree(case, setups):
    stat, positive_only, dtype, extra, axis = case
    su = setups(stat, positive_only)
    basis, dlr = su["basis"], su["dlr"]
    tau_smpl, matsu_smpl = su["tau"], su["matsu"]
    tau_smpl_dlr, matsu_smpl_dlr = su["tau_dlr"], su["matsu_dlr"]

    rng = np.random.default_rng(982743)
    coeffs = _random_dlr_coefficients(rng, dlr.sampling_points, dtype, extra, axis)

    g_ir = dlr.to_IR(coeffs, axis=axis)
    assert g_ir.shape[axis] == basis.size
    assert np.iscomplexobj(g_ir) == np.issubdtype(dtype, np.complexfloating)
    g_dlr_back = dlr.from_IR(g_ir, axis=axis)

    gtau = tau_smpl.evaluate(g_ir, axis=axis)
    _assert_rel(gtau, tau_smpl_dlr.evaluate(coeffs, axis=axis), "tau: IR vs DLR")
    _assert_rel(gtau, tau_smpl_dlr.evaluate(g_dlr_back, axis=axis),
                "tau: IR vs DLR after from_IR")

    giw = matsu_smpl.evaluate(g_ir, axis=axis)
    giw_dlr = matsu_smpl_dlr.evaluate(coeffs, axis=axis)
    _assert_rel(giw, giw_dlr, "Matsubara: IR vs DLR")

    # Matsubara -> IR -> tau -> IR -> Matsubara
    g_ir_1 = matsu_smpl.fit(giw_dlr, axis=axis)
    if not np.issubdtype(dtype, np.complexfloating):
        assert_close(g_ir_1.imag, 0.0, TOL * np.abs(g_ir_1).max(),
                     "real data must give real IR coefficients")
        g_ir_1 = g_ir_1.real
    g_ir_2 = tau_smpl.fit(tau_smpl.evaluate(g_ir_1, axis=axis), axis=axis)
    _assert_rel(giw_dlr, matsu_smpl.evaluate(g_ir_2, axis=axis),
                "Matsubara after the full round trip")
