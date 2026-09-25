# Copyright (C) 2020-2026 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""Smoke test of the public surface (spm-agent-rules testing.md).

``SMOKE`` maps every name in ``sparse_ir.__all__`` to a function that uses the
symbol once with minimal valid arguments and checks a meaningful property of
the result.  The table is compared with ``__all__`` in both directions, so a
new export without an entry and a stale entry both fail.

The second table does the same for the function sets that the bases return
(``u``, ``v``, ``uhat`` of FiniteTempBasis, AugmentedBasis and
DiscreteLehmannRepresentation): evaluation, element access with an integer, a
negative integer and a slice, ``size``/``shape``, and ``xmin``/``xmax``,
``zeta`` or ``deriv`` where the object is documented to provide them.
Nothing in this file may raise NotImplementedError, AttributeError or
TypeError from a documented call.
"""
import numpy as np
import pytest

import sparse_ir
import sparse_ir.augment as aug
from sparse_ir.kernel import kernel_domain

BETA, WMAX, EPS = 10.0, 1.0, 1e-6


@pytest.fixture(scope="module")
def ctx(get_basis):
    basis_f = get_basis("F", BETA, WMAX, EPS)
    basis_b = get_basis("B", BETA, WMAX, EPS)
    return {
        "F": basis_f,
        "B": basis_b,
        "aug": aug.AugmentedBasis(basis_b, aug.TauConst, aug.TauLinear),
        "vertex": aug.AugmentedBasis(basis_f, aug.MatsubaraConst),
        "dlr": sparse_ir.DiscreteLehmannRepresentation(basis_f),
    }


def _roundtrip(smpl, n):
    gl = np.linspace(1.0, 0.1, n)
    values = smpl.evaluate(gl)
    assert np.linalg.norm(values) > 0
    tol = 100 * smpl.cond * np.finfo(np.float64).eps * np.abs(gl).max()
    np.testing.assert_allclose(smpl.fit(values), gl, rtol=0, atol=tol)


def _abstract_basis(c):
    for name in ("F", "aug", "dlr"):
        assert isinstance(c[name], sparse_ir.AbstractBasis)
    with pytest.raises(TypeError, match="abstract"):
        sparse_ir.AbstractBasis()


def _finite_temp_basis(c):
    basis = sparse_ir.FiniteTempBasis("F", BETA, WMAX, EPS)
    assert basis.size == c["F"].size
    np.testing.assert_allclose(basis.s, c["F"].s, rtol=1e-14)


def _finite_temp_bases(c):
    f, b = sparse_ir.finite_temp_bases(BETA, WMAX, EPS)
    assert (f.statistics, b.statistics) == ("F", "B")
    assert f.beta == b.beta == BETA and f.wmax == b.wmax == WMAX


def _tau_sampling(c):
    _roundtrip(sparse_ir.TauSampling(c["F"]), c["F"].size)


def _matsubara_sampling(c):
    _roundtrip(sparse_ir.MatsubaraSampling(c["B"]), c["B"].size)


def _basis_set(c):
    bset = sparse_ir.FiniteTempBasisSet(BETA, WMAX, EPS)
    np.testing.assert_array_equal(bset.tau, sparse_ir.TauSampling(c["F"]).tau)
    np.testing.assert_array_equal(bset.wn_f, sparse_ir.MatsubaraSampling(c["F"]).wn)
    np.testing.assert_array_equal(bset.wn_b, sparse_ir.MatsubaraSampling(c["B"]).wn)


def _logistic_kernel(c):
    k = sparse_ir.LogisticKernel(42.0)
    assert k.lambda_ == 42.0
    np.testing.assert_allclose(kernel_domain(k), [-1, 1, -1, 1], atol=1e-14)


def _reg_bose_kernel(c):
    k = sparse_ir.RegularizedBoseKernel(42.0)
    assert k.lambda_ == 42.0
    np.testing.assert_allclose(kernel_domain(k), [-1, 1, -1, 1], atol=1e-14)


def _sve_result(c):
    sve = sparse_ir.SVEResult(sparse_ir.LogisticKernel(BETA * WMAX), EPS)
    assert len(sve) == sve.s.size > 0
    assert np.all(np.diff(sve.s) <= 0) and sve.s[-1] > 0


def _compute(c):
    sve = sparse_ir.compute(sparse_ir.LogisticKernel(BETA * WMAX), EPS)
    np.testing.assert_allclose(sve.s, c["F"].sve_result.s, rtol=1e-12)


def _compute_sve(c):
    assert sparse_ir.compute_sve is sparse_ir.compute


def _augmented_basis(c):
    assert c["aug"].size == c["B"].size + 2
    _roundtrip(sparse_ir.TauSampling(c["aug"]), c["aug"].size)


def _augmented_tau_function(c):
    u = c["aug"].u
    assert isinstance(u, sparse_ir.AugmentedTauFunction)
    assert (u.xmin, u.xmax) == (-BETA, BETA)
    assert u(0.5).shape == (c["aug"].size,)


def _augmented_matsubara_function(c):
    uhat = c["aug"].uhat
    assert isinstance(uhat, sparse_ir.AugmentedMatsubaraFunction)
    assert uhat.zeta == 0
    assert uhat(0).shape == (c["aug"].size,)


def _abstract_augmentation(c):
    for cls in (sparse_ir.TauConst, sparse_ir.TauLinear, sparse_ir.MatsubaraConst):
        assert issubclass(cls, sparse_ir.AbstractAugmentation)
    with pytest.raises(NotImplementedError):
        sparse_ir.AbstractAugmentation.create(c["B"])


def _tau_const(c):
    tc = sparse_ir.TauConst(BETA)
    assert tc(3.0) == pytest.approx(1 / np.sqrt(BETA))
    assert tc.hat(0) == pytest.approx(np.sqrt(BETA))
    assert tc.hat(2) == 0


def _tau_linear(c):
    tl = sparse_ir.TauLinear(BETA)
    assert tl(0.0) == pytest.approx(-np.sqrt(3 / BETA))
    n = 4
    assert tl.hat(n) == pytest.approx(np.sqrt(3 / BETA) * 2 / 1j * BETA / (np.pi * n))
    assert tl.hat(0) == 0


def _matsubara_const(c):
    mc = sparse_ir.MatsubaraConst(BETA)
    assert np.isnan(mc(1.0))
    assert mc.hat(4) == 1.0


def _dlr(c):
    dlr = c["dlr"]
    gl = -c["F"].s * c["F"].v(0.3)
    np.testing.assert_allclose(dlr.to_IR(dlr.from_IR(gl)), gl, rtol=0,
                               atol=300 * EPS * np.abs(gl).max())


SMOKE = {
    "AbstractBasis": _abstract_basis,
    "FiniteTempBasis": _finite_temp_basis,
    "finite_temp_bases": _finite_temp_bases,
    "TauSampling": _tau_sampling,
    "MatsubaraSampling": _matsubara_sampling,
    "FiniteTempBasisSet": _basis_set,
    "LogisticKernel": _logistic_kernel,
    "RegularizedBoseKernel": _reg_bose_kernel,
    "SVEResult": _sve_result,
    "compute": _compute,
    "compute_sve": _compute_sve,
    "AugmentedBasis": _augmented_basis,
    "AugmentedTauFunction": _augmented_tau_function,
    "AugmentedMatsubaraFunction": _augmented_matsubara_function,
    "AbstractAugmentation": _abstract_augmentation,
    "TauConst": _tau_const,
    "TauLinear": _tau_linear,
    "MatsubaraConst": _matsubara_const,
    "DiscreteLehmannRepresentation": _dlr,
}


def test_smoke_table_matches_all():
    assert sorted(SMOKE) == sorted(sparse_ir.__all__)


@pytest.mark.parametrize("name", sorted(sparse_ir.__all__))
def test_exported_symbol(ctx, name):
    assert hasattr(sparse_ir, name)
    SMOKE[name](ctx)


# (label, how to get the function set, a valid point, provides deriv, kind)
FUNCTION_SETS = [
    ("FiniteTempBasis.u", lambda c: c["F"].u, 0.5, True, "tau"),
    ("FiniteTempBasis.v", lambda c: c["F"].v, 0.5, True, "omega"),
    ("FiniteTempBasis.uhat", lambda c: c["F"].uhat, 3, False, "matsubara"),
    ("AugmentedBasis.u", lambda c: c["aug"].u, 0.5, True, "tau"),
    ("AugmentedBasis.uhat", lambda c: c["aug"].uhat, 2, False, "matsubara"),
    ("vertex AugmentedBasis.uhat", lambda c: c["vertex"].uhat, 3, False, "matsubara"),
    ("DiscreteLehmannRepresentation.u", lambda c: c["dlr"].u, 0.5, False, "tau"),
    ("DiscreteLehmannRepresentation.uhat", lambda c: c["dlr"].uhat, 3, False, "matsubara"),
]


@pytest.mark.parametrize("label, get, x, has_deriv, kind", FUNCTION_SETS,
                         ids=[f[0] for f in FUNCTION_SETS])
def test_function_set(ctx, label, get, x, has_deriv, kind):
    fs = get(ctx)
    values = np.asarray(fs(x))
    size = fs.size
    assert fs.shape == (size,)
    assert values.shape == (size,)
    assert np.linalg.norm(values) > 0

    assert np.shape(fs[0](x)) == ()
    np.testing.assert_allclose(fs[0](x), values[0], rtol=1e-14)
    np.testing.assert_allclose(fs[-1](x), values[-1], rtol=1e-14)
    np.testing.assert_allclose(fs[size - 1](x), values[-1], rtol=1e-14)
    np.testing.assert_allclose(fs[0:size](x), values, rtol=1e-14)

    if kind == "matsubara":
        assert fs.zeta == (1 if x % 2 else 0)
    else:
        bound = BETA if kind == "tau" else WMAX
        assert (fs.xmin, fs.xmax) == (-bound, bound)
    if has_deriv:
        h = 1e-6
        fd = (np.asarray(fs(x + h)) - np.asarray(fs(x - h))) / (2 * h)
        np.testing.assert_allclose(fs.deriv()(x), fd, rtol=0,
                                   atol=1e-5 * np.abs(fd).max())
