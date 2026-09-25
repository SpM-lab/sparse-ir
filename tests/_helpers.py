# Copyright (C) 2020-2026 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""Oracles and utilities shared by the test suite.

The closed forms here do not use libsparseir.  They follow from the
definitions used by sparse-ir v2 for both statistics:

* the logistic kernel ``K(tau, w) = exp(-tau*w) / (1 + exp(-beta*w))``,
* ``G(tau) = -int K(tau, w) rho(w) dw`` and ``G_l = -s_l rho_l``,
* ``Ghat(n) = int_0^beta exp(i*nu_n*tau) G(tau) dtau`` with ``nu_n = n*pi/beta``
  and ``n`` a reduced Matsubara index (odd for fermions, even for bosons).

For a single pole ``rho(w) = delta(w - w0)`` this gives ``gtau_pole`` and
``giv_pole`` below.
"""
import numpy as np


def zeta(statistics):
    """Parity of the reduced Matsubara indices: 1 for 'F', 0 for 'B'."""
    return {'F': 1, 'B': 0}[statistics]


def gtau_pole(tau, w0, beta):
    """``G(tau)`` of a single pole at ``w0``, in a form that cannot overflow."""
    tau = np.asarray(tau, dtype=np.float64)
    if w0 >= 0:
        return -np.exp(-tau * w0) / (1 + np.exp(-beta * w0))
    return -np.exp((beta - tau) * w0) / (1 + np.exp(beta * w0))


def giv_pole(statistics, n, w0, beta):
    """``Ghat(i nu_n)`` of the same pole.

    ``1/(i nu - w0)`` for fermions and ``tanh(beta*w0/2)/(i nu - w0)`` for
    bosons (the tanh factor is the bosonic weight of the logistic kernel).
    """
    iv = 1j * np.pi * np.asarray(n) / beta
    if statistics == 'F':
        return 1 / (iv - w0)
    return np.tanh(beta * w0 / 2) / (iv - w0)


def gauss_legendre_panels(a, b, npanels=400, order=16):
    """Nodes and weights of a composite Gauss-Legendre rule on ``[a, b]``."""
    x0, w0 = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(a, b, npanels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    nodes = (hi - lo) / 2 * x0 + (hi + lo) / 2
    weights = (hi - lo) / 2 * w0
    return nodes.ravel(), weights.ravel()


def strided_views(a):
    """Return ``(label, view)`` pairs that hold the values of ``a``.

    None of the views is C-contiguous, so a binding that hands C the pointer
    of the caller's array instead of a normalized copy reads wrong elements.
    """
    a = np.asarray(a)
    padded = np.zeros((2 * a.shape[0],) + a.shape[1:], dtype=a.dtype)
    padded[::2] = a
    reversed_view = a[::-1].copy()[::-1]
    if a.ndim == 1:
        fortran_backed = np.asfortranarray(np.stack([a, a]))[0]
    else:
        fortran_backed = np.asfortranarray(a)
    views = [("strided slice", padded[::2]),
             ("reversed view", reversed_view),
             ("Fortran-ordered", fortran_backed)]
    for label, view in views:
        assert not view.flags['C_CONTIGUOUS'], label
        np.testing.assert_array_equal(view, a)
    return views


def assert_close(got, ref, atol, what):
    """Assert ``max|got - ref| <= atol`` and report both error measures."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    nref = float(np.linalg.norm(ref))
    rel = float(np.linalg.norm(got - ref)) / nref if nref > 0 else float("inf")
    assert err <= atol, (f"{what}: max|error| = {err:.3e} > atol = {atol:.3e} "
                         f"(relative norm error {rel:.3e})")
