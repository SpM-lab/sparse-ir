# Copyright (C) 2020-2026 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""Boundary contracts that are checked before any call into libsparseir.

Complements test_ffi_boundary.py (dtype matrix and pointer provenance) with
the evaluation domains, index parity, sampling-point validation, parameter
validation, positive_only, empty input and axis handling on arrays with
ndim >= 3.  The same contracts are tested in SparseIR.jl's
``test/spir/boundary_tests.jl``.
"""
import numpy as np
import pytest

import sparse_ir
from sparse_ir import DiscreteLehmannRepresentation as DLR
from sparse_ir.poly import PiecewiseLegendrePoly
from ._helpers import assert_close, strided_views

BETA, WMAX, EPS = 10.0, 1.0, 1e-6


@pytest.fixture(scope="module")
def bases(get_basis):
    return {s: get_basis(s, BETA, WMAX, EPS) for s in "FB"}


# ---------------------------------------------------------------------------
# Evaluation domains of u, v and the DLR functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stat", ["F", "B"])
@pytest.mark.parametrize("tau", [1.5 * BETA, -1.5 * BETA, np.nan, np.inf])
def test_u_rejects_points_outside_minus_beta_beta(bases, stat, tau):
    basis = bases[stat]
    for f in (basis.u, basis.u[0], DLR(basis).u):
        with pytest.raises(ValueError, match="must be finite and lie in"):
            f(tau)
        with pytest.raises(ValueError, match="must be finite and lie in"):
            f(np.array([0.5, tau]))


@pytest.mark.parametrize("w", [1.5 * WMAX, -1.5 * WMAX, np.nan])
def test_v_rejects_points_outside_the_frequency_window(bases, w):
    for f in (bases["F"].v, bases["F"].v[1]):
        with pytest.raises(ValueError, match="must be finite and lie in"):
            f(w)


def test_domain_endpoints_are_accepted(bases):
    basis = bases["F"]
    edges = np.array([-BETA, 0.0, BETA])
    assert np.linalg.norm(basis.u(edges)) > 0
    np.testing.assert_array_equal(basis.u(BETA), basis.u(edges)[:, 2])
    assert np.linalg.norm(basis.v(np.array([-WMAX, WMAX]))) > 0


def test_u_rejects_complex_points(bases):
    with pytest.raises(TypeError, match="must be real-valued"):
        bases["F"].u(0.5 + 0j)


# ---------------------------------------------------------------------------
# Parity of the reduced Matsubara index in uhat
# ---------------------------------------------------------------------------

def test_uhat_rejects_wrong_parity(bases):
    for f in (bases["F"].uhat, bases["F"].uhat[0], DLR(bases["F"]).uhat):
        with pytest.raises(ValueError, match="must be odd"):
            f(2)
        with pytest.raises(ValueError, match="must be odd"):
            f(np.array([1, 2]))
    for f in (bases["B"].uhat, bases["B"].uhat[0], DLR(bases["B"]).uhat):
        with pytest.raises(ValueError, match="must be even"):
            f(1)


# ---------------------------------------------------------------------------
# Empty evaluation arrays
# ---------------------------------------------------------------------------

def test_empty_evaluation_arrays_keep_their_shape(bases):
    basis = bases["F"]
    L = basis.size
    assert basis.u(np.array([])).shape == (L, 0)
    assert basis.u(np.zeros((0, 3))).shape == (L, 0, 3)
    assert basis.u[0](np.array([])).shape == (0,)
    assert basis.v(np.array([])).shape == (L, 0)
    empty_hat = basis.uhat(np.array([], dtype=np.int64))
    assert empty_hat.shape == (L, 0)
    assert empty_hat.dtype == np.complex128


# ---------------------------------------------------------------------------
# Selections of basis functions
# ---------------------------------------------------------------------------

def test_empty_selection_is_rejected(bases):
    # The C library panics on an empty selection (SpM-lab/sparse-ir-rs#269).
    basis = bases["F"]
    for fs in (basis.u, basis.v, basis.uhat):
        with pytest.raises(ValueError, match="empty selection"):
            fs[0:0]
        with pytest.raises(ValueError, match="empty selection"):
            fs[[]]


def test_selections_of_function_sets(bases):
    # As in SparseIR.jl: a one-element selection of u or v gives a single
    # function, a selection of uhat always gives a set.
    basis = bases["F"]
    for fs in (basis.u, basis.v):
        assert type(fs[0:1]) is PiecewiseLegendrePoly
        assert type(fs[[0, 2]]) is type(fs)
        np.testing.assert_array_equal(fs[[0, 2]](0.3), fs(0.3)[[0, 2]])
    for index in (slice(0, 1), [2], [0, 2]):
        assert type(basis.uhat[index]) is type(basis.uhat)
    assert basis.uhat[[2]](3).shape == (1,)
    np.testing.assert_array_equal(basis.uhat[[0, 2]](3), basis.uhat(3)[[0, 2]])


def test_single_functions_have_deriv(bases):
    basis = bases["F"]
    for fs in (basis.u, basis.v):
        for n in (1, 2):
            np.testing.assert_allclose(fs[1].deriv(n)(0.3), fs.deriv(n)(0.3)[1],
                                       rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# DLR poles and C-level failures
# ---------------------------------------------------------------------------

def test_dlr_poles_must_lie_in_the_frequency_window(bases):
    with pytest.raises(ValueError, match="poles must be finite and lie in"):
        DLR(bases["F"], [-5.0, 0.1, 5.0])


def test_c_library_failure_surfaces_as_exception(bases):
    """A genuine C-level failure must raise, not return zeros.

    The DLR basis functions are not piecewise polynomials, so the C library
    reports SPIR_NOT_SUPPORTED (-5) for their derivative.
    """
    with pytest.raises(RuntimeError, match="-5"):
        DLR(bases["F"]).u.deriv()


# ---------------------------------------------------------------------------
# Sampling points
# ---------------------------------------------------------------------------

def test_tau_sampling_rejects_points_outside_minus_beta_beta(bases):
    basis = bases["F"]
    points = np.linspace(0, 1.5 * BETA, basis.size + 2)
    with pytest.raises(ValueError, match=r"must lie in \[-beta, beta\]"):
        sparse_ir.TauSampling(basis, points)


def test_sampling_rejects_duplicate_points(bases):
    basis = bases["F"]
    tau = np.linspace(0.1, 9.9, basis.size)
    tau[3] = tau[1]
    with pytest.raises(ValueError, match="duplicate"):
        sparse_ir.TauSampling(basis, tau)
    with pytest.raises(ValueError, match="duplicate"):
        sparse_ir.MatsubaraSampling(basis, [1, 3, 5, 1])


def test_tau_sampling_keeps_the_given_order(bases):
    basis = bases["F"]
    points = np.linspace(0.1, 9.9, basis.size + 3)[::-1].copy()
    smpl = sparse_ir.TauSampling(basis, points)
    np.testing.assert_array_equal(smpl.tau, points)
    rng = np.random.default_rng(7)
    gl = rng.normal(size=basis.size)
    ref = gl @ basis.u(points)
    assert_close(smpl.evaluate(gl), ref, 1e-13 * np.abs(ref).max(),
                 "evaluate follows the order of the given points")
    assert_close(smpl.fit(ref), gl,
                 100 * smpl.cond * np.finfo(np.float64).eps * np.abs(gl).max(),
                 "fit follows the order of the given points")


def test_tau_sampling_does_not_alias_the_callers_array(bases):
    basis = bases["F"]
    points = np.linspace(0.1, 9.9, basis.size)
    smpl = sparse_ir.TauSampling(basis, points)
    points[0] = 5.0
    assert smpl.tau[0] == pytest.approx(0.1)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_tau_sampling_points_are_widened(bases, dtype):
    basis = bases["F"]
    points = np.arange(1, basis.size + 1).astype(dtype)
    smpl = sparse_ir.TauSampling(basis, points)
    assert smpl.tau.dtype == np.float64
    np.testing.assert_array_equal(smpl.tau, points.astype(np.float64))


def test_tau_sampling_points_may_be_views(bases):
    basis = bases["F"]
    points = np.linspace(0.3, 9.7, basis.size + 1)
    ref = sparse_ir.TauSampling(basis, points).evaluate(np.ones(basis.size))
    for label, view in strided_views(points):
        got = sparse_ir.TauSampling(basis, view).evaluate(np.ones(basis.size))
        assert_close(got, ref, 1e-14 * np.abs(ref).max(), label)


def test_tau_sampling_with_few_points_evaluates(bases):
    """Fewer points than basis functions are accepted for evaluation."""
    basis = bases["F"]
    points = np.array([0.1, 0.4])
    rng = np.random.default_rng(5)
    gl = rng.normal(size=basis.size)
    ref = gl @ basis.u(points)
    assert_close(sparse_ir.TauSampling(basis, points).evaluate(gl), ref,
                 1e-13 * np.abs(ref).max(), "evaluate at two points")


def test_positive_only_requires_non_negative_points(bases):
    with pytest.raises(ValueError, match="non-negative sampling points"):
        sparse_ir.MatsubaraSampling(bases["F"], [-1, 1, 3], positive_only=True)


# ---------------------------------------------------------------------------
# positive_only
# ---------------------------------------------------------------------------

def test_positive_only_rejects_complex_coefficients(bases):
    basis = bases["F"]
    smpl = sparse_ir.MatsubaraSampling(basis, positive_only=True)
    rng = np.random.default_rng(11)
    gl = rng.normal(size=basis.size)
    with pytest.raises(ValueError, match="positive_only"):
        smpl.evaluate(gl + 1j * rng.normal(size=basis.size))
    # A real quantity stays accepted, also when carried in a complex array.
    giv = smpl.evaluate(gl)
    fitted = smpl.fit(giv)
    assert np.iscomplexobj(fitted)
    assert_close(smpl.evaluate(fitted), giv, 1e-12 * np.abs(giv).max(),
                 "fit result of a real quantity")
    # So is an imaginary part below the tolerance max(10 accuracy, 1e-12).
    assert_close(smpl.evaluate(gl + 1e-14j), giv, 1e-12 * np.abs(giv).max(),
                 "negligible imaginary part")


@pytest.mark.parametrize("stat", ["F", "B"])
def test_positive_only_cond_is_that_of_the_real_fit(bases, stat):
    # fit solves the real system [Re A; Im A] x = [Re g; Im g]; the C library
    # reports the condition number of the complex A instead
    # (SpM-lab/sparse-ir-rs#270).
    basis = bases[stat]
    smpl = sparse_ir.MatsubaraSampling(basis, positive_only=True)
    A = basis.uhat(smpl.sampling_points).T
    ref = np.linalg.cond(np.vstack([A.real, A.imag]))
    assert smpl.cond == pytest.approx(ref, rel=1e-10)


# ---------------------------------------------------------------------------
# Axis handling on arrays with ndim >= 3 and unequal dimensions
# ---------------------------------------------------------------------------

def _transforms(basis):
    dlr = DLR(basis)
    return {
        "TauSampling": (sparse_ir.TauSampling(basis), basis.size),
        "MatsubaraSampling": (sparse_ir.MatsubaraSampling(basis), basis.size),
        "MatsubaraSampling(positive_only)": (
            sparse_ir.MatsubaraSampling(basis, positive_only=True), basis.size),
        "DLR.from_IR": (dlr, basis.size),
    }


@pytest.mark.parametrize("axis", [0, 1, -1, 2])
@pytest.mark.parametrize("stat", ["F", "B"])
def test_axis_on_three_dimensional_input(bases, stat, axis):
    basis = bases[stat]
    rng = np.random.default_rng(3)
    for name, (obj, n_in) in _transforms(basis).items():
        data0 = rng.normal(size=(n_in, 3, 5))        # basis axis first
        data = np.moveaxis(data0, 0, axis)
        if name == "DLR.from_IR":
            forward, backward = obj.from_IR, obj.to_IR
        else:
            forward, backward = obj.evaluate, obj.fit
        out = forward(data, axis=axis)
        ref = forward(data0, axis=0)
        np.testing.assert_allclose(np.moveaxis(out, axis, 0), ref,
                                   rtol=0, atol=1e-13 * np.abs(ref).max())
        back = backward(out, axis=axis)
        back_ref = backward(ref, axis=0)
        np.testing.assert_allclose(np.moveaxis(back, axis, 0), back_ref,
                                   rtol=0, atol=1e-12 * np.abs(back_ref).max())


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta, wmax, eps, match", [
    (0.0, 1.0, 1e-6, "beta must be positive"),
    (-1.0, 1.0, 1e-6, "beta must be positive"),
    (np.inf, 1.0, 1e-6, "beta must be positive"),
    (np.nan, 1.0, 1e-6, "beta must be positive"),
    (10.0, 0.0, 1e-6, "wmax must be positive"),
    (10.0, -1.0, 1e-6, "wmax must be positive"),
    (10.0, np.inf, 1e-6, "wmax must be positive"),
    (10.0, 1.0, 0.0, "eps must be positive"),
    (10.0, 1.0, -1e-6, "eps must be positive"),
    (10.0, 1.0, np.nan, "eps must be positive"),
])
def test_basis_parameters_are_validated(beta, wmax, eps, match):
    with pytest.raises(ValueError, match=match):
        sparse_ir.FiniteTempBasis("F", beta, wmax, eps)


def test_max_size_is_validated():
    with pytest.raises(ValueError, match="max_size"):
        sparse_ir.FiniteTempBasis("F", BETA, WMAX, EPS, max_size=0)


def test_kernel_must_match_statistics_and_cutoff():
    with pytest.raises(ValueError, match="incompatible with fermionic"):
        sparse_ir.FiniteTempBasis("F", BETA, WMAX, EPS,
                                  kernel=sparse_ir.RegularizedBoseKernel(BETA * WMAX))
    with pytest.raises(ValueError, match="does not match"):
        sparse_ir.FiniteTempBasis("F", BETA, WMAX, EPS,
                                  kernel=sparse_ir.LogisticKernel(42.0))


def test_sve_result_must_match_the_kernel():
    sve_42 = sparse_ir.compute(sparse_ir.LogisticKernel(42.0), EPS)
    with pytest.raises(ValueError, match="lambda_ = 42.0"):
        sparse_ir.FiniteTempBasis("F", BETA, WMAX, EPS, sve_result=sve_42)
    sve_bose = sparse_ir.compute(sparse_ir.RegularizedBoseKernel(BETA * WMAX), EPS)
    with pytest.raises(ValueError, match="RegularizedBoseKernel"):
        sparse_ir.FiniteTempBasis("B", BETA, WMAX, EPS, sve_result=sve_bose)


@pytest.mark.parametrize("kernel", [sparse_ir.LogisticKernel,
                                    sparse_ir.RegularizedBoseKernel])
@pytest.mark.parametrize("lambda_", [0.0, -1.0, np.inf, np.nan])
def test_kernel_cutoff_is_validated(kernel, lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        kernel(lambda_)


@pytest.mark.parametrize("eps", [0.0, -1e-6, np.nan])
def test_sve_accuracy_is_validated(eps):
    with pytest.raises(ValueError, match="eps must be positive"):
        sparse_ir.compute(sparse_ir.LogisticKernel(10.0), eps)


@pytest.mark.parametrize("eps", [None, "1e-6"])
def test_sve_accuracy_must_be_a_number(eps):
    with pytest.raises(TypeError, match="eps must be a real number"):
        sparse_ir.sve.SVEResult(sparse_ir.LogisticKernel(10.0), eps)
