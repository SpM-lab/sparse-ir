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
