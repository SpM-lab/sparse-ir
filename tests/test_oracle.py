# Copyright (C) 2020-2026 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""Oracle tests: closed forms, definitions and symmetries.

None of the reference values comes from libsparseir.  Tolerance classes:

* T-eps: ``atol = 300 * eps * scale`` for results limited by the IR
  truncation (the bound used by the DLR compression test in test_dlr.py).
* T-m:   ``atol = 1e-10 * scale`` for identities that hold exactly for the
  stored piecewise polynomials.
* T-c:   ``atol = 100 * cond * eps64 * max|input|`` for round trips.

The same parameters and tolerances are used by SparseIR.jl's
``test/spir/oracle_tests.jl``.
"""
import numpy as np
import pytest

import sparse_ir
from ._helpers import (assert_close, gauss_legendre_panels, giv_pole,
                       gtau_pole, zeta)

B1 = (10.0, 1.0, 1e-10)       # beta, wmax, eps
B2 = (1000.0, 1.0, 1e-10)
HARD = {                       # R1..R4 of the design spec
    "R1": (1.0, 0.1, 1e-6),
    "R2": (1000.0, 100.0, 1e-8),
    "R3": (10.0, 1.0, 1e-15),
    "R4": (10.0, 1.0, 1e-6),
}
EPS64 = np.finfo(np.float64).eps
ISSUE_265 = ("https://github.com/SpM-lab/sparse-ir-rs/issues/265: uhat is wrong "
             "for l = 1, 2 (mod 4) at |n| >= n_asymp = 40*lambda")


def _poles(beta, wmax):
    # 2/beta keeps tanh(beta*w0/2) away from +-1, so that the fermionic and
    # bosonic closed forms differ even at beta = 1000.
    return (-0.8 * wmax, 0.3 * wmax, 2.0 / beta)


@pytest.mark.parametrize("params", [B1, B2], ids=["B1", "B2"])
@pytest.mark.parametrize("stat", ["F", "B"])
def test_o1_single_pole_in_tau(stat, params, get_basis):
    beta, wmax, eps = params
    basis = get_basis(stat, beta, wmax, eps)
    smpl = sparse_ir.TauSampling(basis)
    taus = np.array([0.0, beta / 7, beta / 2, beta])
    for w0 in _poles(beta, wmax):
        gl = -basis.s * basis.v(w0)
        ref = gtau_pole(taus, w0, beta)
        got = gl @ basis.u(taus)
        tol = 300 * eps * np.abs(ref).max()
        assert_close(got, ref, tol, f"G(tau) of the pole at {w0}")
        assert_close(smpl.evaluate(gl), gtau_pole(smpl.tau, w0, beta), tol,
                     "TauSampling.evaluate")
        # G(0) + G(beta) = -1 for a normalized pole
        assert_close(got[0] + got[-1], -1.0, 300 * eps, "G(0) + G(beta)")


@pytest.mark.parametrize("params", [B1, B2], ids=["B1", "B2"])
@pytest.mark.parametrize("stat", ["F", "B"])
def test_o1_single_pole_in_matsubara(stat, params, get_basis):
    beta, wmax, eps = params
    basis = get_basis(stat, beta, wmax, eps)
    z = zeta(stat)
    extra = np.array([z, -z, 20 + z, -20 - z])
    for positive_only in (False, True):
        smpl = sparse_ir.MatsubaraSampling(basis, positive_only=positive_only)
        ns = np.unique(np.concatenate([smpl.wn, extra]))
        for w0 in _poles(beta, wmax):
            gl = -basis.s * basis.v(w0)
            ref = giv_pole(stat, ns, w0, beta)
            tol = 300 * eps * np.abs(ref).max()
            assert_close(gl @ basis.uhat(ns), ref, tol,
                         f"Ghat of the pole at {w0}")
            assert_close(smpl.evaluate(gl), giv_pole(stat, smpl.wn, w0, beta),
                         tol, f"MatsubaraSampling.evaluate "
                         f"(positive_only={positive_only})")


@pytest.mark.parametrize("stat", ["F", "B"])
def test_o2_uhat_is_the_fourier_transform_of_u(stat, get_basis):
    """``uhat_l(n) == int_0^beta exp(i*pi*n*tau/beta) u_l(tau) dtau`` (T-m).

    Only ``|n| <= 20*lambda``: from ``n_asymp = 40*lambda`` on, the backend
    uses an asymptotic series that is wrong for some ``l``
    (SpM-lab/sparse-ir-rs#265).  That regime is pinned by the strict xfail
    tests below.
    """
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    xs, ws = gauss_legendre_panels(0.0, beta)
    U = basis.u(xs)
    ls = sorted({0, 1, basis.size // 2, basis.size - 1})
    ns = [1, -1, 11, -11, 101, -101] if stat == 'F' else [0, 10, -10, 100, -100]
    for n in ns:
        ref = U[ls] @ (np.exp(1j * np.pi * n * xs / beta) * ws)
        assert_close(basis.uhat(n)[ls], ref, 1e-10 * np.sqrt(beta), f"uhat({n})")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_265)
@pytest.mark.parametrize("stat", ["F", "B"])
def test_o2_uhat_in_the_asymptotic_regime(stat, get_basis):
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    n = int(80 * beta * wmax) + zeta(stat)     # twice n_asymp
    xs, ws = gauss_legendre_panels(0.0, beta, npanels=1600, order=24)
    ref = basis.u(xs) @ (np.exp(1j * np.pi * n * xs / beta) * ws)
    assert_close(basis.uhat(n), ref, 1e-10 * np.sqrt(beta), f"uhat({n})")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_265)
@pytest.mark.parametrize("stat", ["F", "B"])
def test_o4_uhat_high_frequency_tail(stat, get_basis):
    """``i*nu_n*uhat_l(n) -> -(u_l(beta) + u_l(0))`` (F), ``u_l(beta) - u_l(0)`` (B).

    Follows from integrating the definition by parts; the O(1/nu) remainder
    is below 1e-9 at ``n ~ 2**40``, while truncating the index would give an
    O(1) error.
    """
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    n = 2**40 + zeta(stat)
    nu = np.pi * n / beta
    u0, ub = basis.u(0.0), basis.u(beta)
    limit = -(ub + u0) if stat == 'F' else ub - u0
    assert_close(1j * nu * basis.uhat(n), limit, 1e-6 * np.abs(u0).max(),
                 "i nu uhat(n)")


@pytest.mark.parametrize("stat", ["F", "B"])
def test_o3_symmetries(stat, get_basis):
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    L = basis.size
    sign = (-1.0) ** np.arange(L)

    tau = np.linspace(0, beta, 101)
    U, U_reflected = basis.u(tau), basis.u(beta - tau)
    for l in range(L):
        assert_close(U_reflected[l], sign[l] * U[l],
                     300 * eps * np.abs(U[l]).max(), f"u_{l}(beta - tau)")

    w = np.linspace(-wmax, wmax, 101)
    V, V_reflected = basis.v(w), basis.v(-w)
    for l in range(L):
        assert_close(V_reflected[l], sign[l] * V[l],
                     300 * eps * np.abs(V[l]).max(), f"v_{l}(-w)")

    z = zeta(stat)
    ns = np.array([z, 2 + z, 10 + z, 100 + z])    # |n| <= 20*lambda, see O2
    Up, Um = basis.uhat(ns), basis.uhat(-ns)
    assert_close(Um, np.conj(Up), 1e-10 * np.sqrt(beta), "uhat(-n) = conj(uhat(n))")
    # u_l(beta - tau) = (-1)^l u_l(tau) and exp(i*nu_n*beta) = -1 (F), +1 (B):
    # for fermions Re uhat_l vanishes for even l and Im uhat_l for odd l;
    # for bosons it is the other way round.
    even = (np.arange(L) % 2 == 0)[:, None]
    vanishing = np.where(even == (stat == 'F'), Up.real, Up.imag)
    assert_close(vanishing, 0.0, 300 * eps * np.sqrt(beta),
                 "vanishing Re/Im part of uhat")

    inner = tau[1:-1]
    periodic_sign = -1.0 if stat == 'F' else 1.0
    ref = periodic_sign * basis.u(beta - inner)
    assert_close(basis.u(-inner), ref, 1e-10 * np.abs(ref).max(),
                 "(anti)periodic extension u(-tau)")


@pytest.mark.parametrize("stat", ["F", "B"])
def test_o4_roots_orthonormality_and_default_points(stat, get_basis):
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    L = basis.size

    # An even point count keeps beta/2, where the odd functions vanish, off the grid.
    grid = np.linspace(0, beta, 20000)[1:-1]
    Ug = basis.u(grid)
    changes = [int(np.count_nonzero(np.sign(Ug[l, 1:]) != np.sign(Ug[l, :-1])))
               for l in range(L)]
    assert changes == list(range(L)), "u_l must have exactly l sign changes"

    xs, ws = gauss_legendre_panels(0.0, beta)
    U = basis.u(xs)
    assert_close((U * ws) @ U.T, np.eye(L), 1e-12, "int_0^beta u_l u_m")
    xv, wv = gauss_legendre_panels(-wmax, wmax)
    V = basis.v(xv)
    assert_close((V * wv) @ V.T, np.eye(L), 1e-12, "int v_l v_m")

    # Only structure: the values depend on the backend version.
    taus = basis.default_tau_sampling_points()
    assert taus.size == L
    assert np.all(np.diff(taus) > 0)
    assert np.all((taus > 0) & (taus < beta))
    full = basis.default_matsubara_sampling_points()
    nonneg = basis.default_matsubara_sampling_points(positive_only=True)
    assert full.size >= L
    assert np.all(full % 2 == zeta(stat))
    np.testing.assert_array_equal(np.sort(full), np.sort(-full))
    np.testing.assert_array_equal(np.sort(nonneg), np.sort(full[full >= 0]))


@pytest.mark.parametrize("stat", ["F", "B"])
def test_o4_dlr_functions_have_closed_forms(stat, get_basis):
    beta, wmax, eps = B1
    basis = get_basis(stat, beta, wmax, eps)
    dlr = sparse_ir.DiscreteLehmannRepresentation(basis)
    poles = dlr.sampling_points

    tau = np.array([0.0, 1.3, 7.0, beta])
    ref_u = np.array([gtau_pole(tau, w, beta) for w in poles])
    assert_close(dlr.u(tau), ref_u, 1e-10 * np.abs(ref_u).max(), "dlr.u")

    z = zeta(stat)
    ns = np.array([z, 6 + z, -6 - z])
    ref_uhat = np.array([giv_pole(stat, ns, w, beta) for w in poles])
    assert_close(dlr.uhat(ns), ref_uhat, 1e-10 * np.abs(ref_uhat).max(), "dlr.uhat")

    rng = np.random.default_rng(265)
    c = rng.normal(size=poles.size)
    ref_gl = -basis.s * (basis.v(poles) @ c)
    assert_close(dlr.to_IR(c), ref_gl, 1e-10 * np.abs(ref_gl).max(), "to_IR")

    gl = -basis.s * basis.v(0.3 * wmax)
    g_dlr = dlr.from_IR(gl)
    ref = gl @ basis.u(tau)
    assert_close(g_dlr @ dlr.u(tau), ref, 300 * eps * np.abs(ref).max(),
                 "DLR and IR agree in tau")


@pytest.mark.parametrize("regime", sorted(HARD))
@pytest.mark.parametrize("stat", ["F", "B"])
def test_o5_hard_regimes(stat, regime, get_basis):
    beta, wmax, eps = HARD[regime]
    basis = get_basis(stat, beta, wmax, eps)
    gl = np.linspace(1.0, -0.5, basis.size) * basis.s / basis.s[0]
    for smpl in (sparse_ir.TauSampling(basis),
                 sparse_ir.MatsubaraSampling(basis),
                 sparse_ir.MatsubaraSampling(basis, positive_only=True)):
        tol = 100 * smpl.cond * EPS64 * np.abs(gl).max()
        assert_close(smpl.fit(smpl.evaluate(gl)), gl, tol,
                     f"{smpl!r} round trip")

    w0 = 0.3 * wmax
    taus = np.array([0.0, beta / 3, beta])
    ref = gtau_pole(taus, w0, beta)
    got = (-basis.s * basis.v(w0)) @ basis.u(taus)
    assert_close(got, ref, 300 * eps * np.abs(ref).max(), "single pole")

    assert basis.sve_result.s.size > basis.size    # the basis is truncated by eps
    assert basis.accuracy < eps <= basis.significance[-1]
