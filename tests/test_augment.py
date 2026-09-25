# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import numpy as np
import sparse_ir
from sparse_ir import poly
from sparse_ir import augment
from sparse_ir import _util

import pytest


def test_augmented_bosonic_basis():
    """Augmented bosonic basis"""
    wmax = 2
    beta = 1000
    basis = sparse_ir.FiniteTempBasis("B", beta, wmax, eps=1e-6)
    basis_comp = augment.AugmentedBasis(basis, augment.TauConst, augment.TauLinear)

    # G(tau) = c - e^{-tau*pole}/(1 - e^{-beta*pole})
    pole = 1.0
    const = 1e-2
    tau_smpl = sparse_ir.TauSampling(basis_comp)
    assert tau_smpl.sampling_points.size == basis_comp.size
    gtau = const + basis.u(tau_smpl.tau).T @ (-basis.s * basis.v(pole))
    magn = np.abs(gtau).max()

    gl_fit = tau_smpl.fit(gtau)
    gtau_reconst = tau_smpl.evaluate(gl_fit)
    np.testing.assert_allclose(gtau_reconst, gtau, atol=1e-13 * magn, rtol=0)


@pytest.mark.parametrize("stat", ["F", "B"])
def test_vertex_basis(stat):
    """Vertex basis"""
    wmax = 2
    beta = 1000
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax, eps=1e-6)
    basis_comp = augment.AugmentedBasis(basis, augment.MatsubaraConst)
    assert basis_comp.uhat is not None

    # G(iv) = c + 1/(iv-pole)
    pole = 1.0
    c = 1.0
    matsu_smpl = sparse_ir.MatsubaraSampling(basis_comp)
    giv = c  + 1/(1J*matsu_smpl.sampling_points * np.pi/beta - pole)
    gl = matsu_smpl.fit(giv)

    giv_reconst = matsu_smpl.evaluate(gl)

    np.testing.assert_allclose(giv, giv_reconst,
                               atol=np.abs(giv).max() * 1e-7, rtol=0)


@pytest.mark.parametrize("stat, augmentations", [
    ("B", (augment.TauConst, augment.TauLinear)),
    ("B", (augment.MatsubaraConst,)),
    ("F", (augment.MatsubaraConst,)),
])
def test_positive_only_matsubara_sampling_of_augmented_basis(stat, augmentations):
    """positive_only=True samples an augmented basis on the non-negative half
    of its full default point set, as for a plain basis."""
    basis = sparse_ir.FiniteTempBasis(stat, 10.0, 1.0, eps=1e-6)
    basis_comp = augment.AugmentedBasis(basis, *augmentations)
    full = basis_comp.default_matsubara_sampling_points()
    half = basis_comp.default_matsubara_sampling_points(positive_only=True)
    np.testing.assert_array_equal(half, full[full >= 0])

    smpl_full = sparse_ir.MatsubaraSampling(basis_comp)
    smpl = sparse_ir.MatsubaraSampling(basis_comp, positive_only=True)
    np.testing.assert_array_equal(smpl.sampling_points, half)
    gl = np.random.default_rng(4321).standard_normal(basis_comp.size)
    giv = smpl.evaluate(gl)
    np.testing.assert_allclose(giv, smpl_full.evaluate(gl)[full >= 0],
                               atol=1e-13 * np.abs(giv).max(), rtol=0)
    # T-c against the full set's condition number: the cond of a
    # positive-only sampling is that of the complex half matrix, which
    # understates the conditioning of the real least-squares fit.
    atol = 100 * smpl_full.cond * np.finfo(np.float64).eps * np.abs(gl).max()
    np.testing.assert_allclose(smpl.fit(giv).real, gl, atol=atol, rtol=0)


def test_normalize_tau_bosonic():
    """Test normalize_tau for bosonic statistics"""
    beta = 10.0
    
    # Test positive tau
    tau_norm, sign = _util.normalize_tau('B', 5.0, beta)
    assert tau_norm == 5.0
    assert sign == 1.0
    
    # Test negative tau (periodic)
    tau_norm, sign = _util.normalize_tau('B', -3.0, beta)
    assert np.isclose(tau_norm, 7.0)
    assert sign == 1.0
    
    # Test negative zero
    tau_norm, sign = _util.normalize_tau('B', -0.0, beta)
    assert tau_norm == beta
    assert sign == 1.0
    
    # Test array input
    taus = np.array([0.0, 5.0, -3.0, beta])
    tau_norms, signs = _util.normalize_tau('B', taus, beta)
    assert np.allclose(tau_norms, [0.0, 5.0, 7.0, beta])
    assert np.allclose(signs, [1.0, 1.0, 1.0, 1.0])


def test_normalize_tau_fermionic():
    """Test normalize_tau for fermionic statistics"""
    beta = 10.0
    
    # Test positive tau
    tau_norm, sign = _util.normalize_tau('F', 5.0, beta)
    assert tau_norm == 5.0
    assert sign == 1.0
    
    # Test negative tau (anti-periodic)
    tau_norm, sign = _util.normalize_tau('F', -3.0, beta)
    assert np.isclose(tau_norm, 7.0)
    assert sign == -1.0
    
    # Test negative zero
    tau_norm, sign = _util.normalize_tau('F', -0.0, beta)
    assert tau_norm == beta
    assert sign == -1.0
    
    # Test array input
    taus = np.array([0.0, 5.0, -3.0])
    tau_norms, signs = _util.normalize_tau('F', taus, beta)
    assert np.allclose(tau_norms, [0.0, 5.0, 7.0])
    assert np.allclose(signs, [1.0, 1.0, -1.0])


def test_normalize_tau_errors():
    """Test normalize_tau error handling"""
    beta = 10.0
    
    # Out of range
    with pytest.raises(ValueError, match="τ must be in"):
        _util.normalize_tau('B', -beta - 1, beta)
    
    with pytest.raises(ValueError, match="τ must be in"):
        _util.normalize_tau('B', beta + 1, beta)
    
    # Invalid statistics
    with pytest.raises(ValueError, match="statistics must be"):
        _util.normalize_tau('X', 0.0, beta)


def test_tau_const_rejects_fermionic():
    """A fermionic TauConst would be identically zero in Matsubara."""
    with pytest.raises(ValueError, match="only allowed for a bosonic basis"):
        augment.TauConst(10.0, 'F')

    basis = sparse_ir.FiniteTempBasis('F', 10.0, wmax=2.0, eps=1e-6)
    with pytest.raises(ValueError, match="only allowed for a bosonic basis"):
        augment.TauConst.create(basis)
    with pytest.raises(ValueError, match="only allowed for a bosonic basis"):
        augment.AugmentedBasis(basis, augment.TauConst)


def test_tau_const_periodicity():
    """Test TauConst periodicity (bosonic: periodic in beta)"""
    beta = 10.0
    tc = augment.TauConst(beta, 'B')

    assert np.isclose(tc(0.0), 1.0 / np.sqrt(beta))
    assert np.isclose(tc(5.0), 1.0 / np.sqrt(beta))
    # tau=-5 goes through the periodicity branch
    assert np.isclose(tc(-5.0), 1.0 / np.sqrt(beta))


@pytest.mark.parametrize("stat", ["F", "B"])
def test_tau_linear_periodicity(stat):
    """Test TauLinear with statistics-dependent periodicity"""
    beta = 10.0
    tl = augment.TauLinear(beta, stat)
    
    # Test at tau=0
    val0 = tl(0.0)
    norm = np.sqrt(3 / beta)
    assert np.isclose(val0, -norm)  # x = 2*0/beta - 1 = -1
    
    # Test at tau=5 (middle)
    val_mid = tl(5.0)
    assert np.isclose(val_mid, 0.0)  # x = 2*5/10 - 1 = 0
    
    # tau = -beta/4 maps to 3*beta/4 (x = 1/2); the sign depends on statistics
    val_neg = tl(-beta / 4)
    expected_sign = -1.0 if stat == 'F' else 1.0
    assert np.isclose(val_neg, expected_sign * norm * 0.5)


def test_matsubara_const_range():
    """Test MatsubaraConst accepts [-β, β] range"""
    beta = 10.0
    mc = augment.MatsubaraConst(beta)
    
    # Should accept negative tau and return NaN
    assert np.isnan(mc(-5.0))
    assert np.isnan(mc(5.0))
    assert np.isnan(mc(0.0))
    assert np.isnan(mc(-beta))
    assert np.isnan(mc(beta))
    
    # Should reject out of range
    with pytest.raises(ValueError):
        mc(-beta - 1)
    with pytest.raises(ValueError):
        mc(beta + 1)


def test_tau_const_with_statistics():
    """Test TauConst can be created with statistics parameter"""
    beta = 10.0
    basis = sparse_ir.FiniteTempBasis('B', beta, wmax=2.0, eps=1e-6)

    # Test factory method
    tc = augment.TauConst.create(basis)
    assert tc._statistics == 'B'

    # Test direct creation
    tc2 = augment.TauConst(beta, 'B')
    assert tc2._statistics == 'B'
    
    # The normalized constant: int_0^beta tc(tau)^2 dtau == 1
    assert np.isclose(tc(5.0), 1 / np.sqrt(beta))


@pytest.mark.parametrize("stat", ["F", "B"])
def test_tau_linear_with_statistics(stat):
    """Test TauLinear can be created with statistics parameter"""
    beta = 10.0
    basis = sparse_ir.FiniteTempBasis(stat, beta, wmax=2.0, eps=1e-6)
    
    # Test factory method
    tl = augment.TauLinear.create(basis)
    assert tl._statistics == stat
    
    # Test direct creation
    tl2 = augment.TauLinear(beta, stat)
    assert tl2._statistics == stat
    
    # tau = beta/2 is the zero of the linear function
    assert np.isclose(tl(5.0), 0.0)
    assert np.isclose(tl(beta), np.sqrt(3 / beta))


def test_backward_compatibility():
    """Test backward compatibility for augmentation constructors"""
    beta = 10.0
    
    # Old API (without statistics) should default to Bosonic
    tc = augment.TauConst(beta)
    assert tc._statistics == 'B'
    
    tl = augment.TauLinear(beta)
    assert tl._statistics == 'B'
    
    # MatsubaraConst can be created without statistics
    mc = augment.MatsubaraConst(beta)
    assert mc._statistics is None
