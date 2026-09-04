# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
"""
Regression tests for the ctypes boundary and for the audit findings A-H.

These cover, for every function that hands a numpy buffer to C:

* the dtype matrix (``float32``/``float64``/``complex64``/``complex128`` plus
  integer input) -- a narrow dtype must be widened, never reinterpreted;
* non-contiguous input -- the pointer must come from the normalized copy;
* concrete exception types with ``match=`` for rejected input;
* nonzero-norm assertions, so a silently zeroed result cannot pass.
"""

import numpy as np
import pytest

import sparse_ir
from sparse_ir import _util
from sparse_ir.dlr import DiscreteLehmannRepresentation


REAL_DTYPES = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
ALL_DTYPES = REAL_DTYPES + COMPLEX_DTYPES

# float32 carries ~7 decimal digits; a widened float32 input can only agree
# with the float64 reference to its own input precision.  The coefficient
# vectors below span many orders of magnitude, so the tolerance is applied
# relative to the norm of the reference rather than element-wise.
DTYPE_TOL = {
    np.float32: 1e-6,
    np.complex64: 1e-6,
    np.float64: 1e-14,
    np.complex128: 1e-14,
}


def assert_close(got, ref, dtype):
    """Compare against a float64 reference at the input dtype's precision."""
    scale = np.linalg.norm(ref)
    assert scale > 0
    np.testing.assert_allclose(got, ref, rtol=0, atol=DTYPE_TOL[dtype] * scale)


@pytest.fixture(scope="module")
def basis():
    return sparse_ir.FiniteTempBasis('F', 2.0, 5.0, 1e-8)


@pytest.fixture(scope="module")
def basis_b():
    return sparse_ir.FiniteTempBasis('B', 2.0, 5.0, 1e-8)


@pytest.fixture(scope="module")
def gl(basis):
    rng = np.random.RandomState(4711)
    return basis.s * rng.randn(basis.size)


def noncontiguous(a):
    """Return a non-contiguous view holding the same values as ``a``."""
    padded = np.zeros(tuple(2 * n for n in a.shape), dtype=a.dtype)
    view = padded[(slice(None, None, 2),) * a.ndim]
    view[...] = a
    assert not view.flags['C_CONTIGUOUS']
    return view


# ---------------------------------------------------------------------------
# Finding B: dtype normalization at the boundary (tau sampling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tau_evaluate_dtype_matrix(basis, gl, dtype):
    smpl = sparse_ir.TauSampling(basis)
    ref = smpl.evaluate(gl.astype(np.float64))
    assert np.linalg.norm(ref) > 0

    got = smpl.evaluate(gl.astype(dtype))
    assert got.dtype == (np.complex128 if np.issubdtype(dtype, np.complexfloating)
                         else np.float64)
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tau_fit_dtype_matrix(basis, gl, dtype):
    smpl = sparse_ir.TauSampling(basis)
    gtau = smpl.evaluate(gl)
    ref = smpl.fit(gtau.astype(np.float64))
    assert np.linalg.norm(ref) > 0

    got = smpl.fit(gtau.astype(dtype))
    assert_close(got, ref, dtype)
    assert_close(np.real(got), gl, dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tau_noncontiguous(basis, gl, dtype):
    """A strided input must be copied, and the pointer taken from the copy."""
    smpl = sparse_ir.TauSampling(basis)
    ref = smpl.evaluate(gl)
    got = smpl.evaluate(noncontiguous(gl.astype(dtype)))
    assert_close(got, ref, dtype)


def test_tau_integer_input(basis):
    """Integer coefficients are widened, not reinterpreted as doubles."""
    smpl = sparse_ir.TauSampling(basis)
    al = np.zeros(basis.size, dtype=np.int64)
    al[0] = 3
    ref = smpl.evaluate(al.astype(np.float64))
    np.testing.assert_allclose(smpl.evaluate(al), ref, rtol=1e-14, atol=0)
    assert np.linalg.norm(ref) > 0


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_tau_axis_coverage(basis, gl, axis):
    smpl = sparse_ir.TauSampling(basis)
    stacked = np.stack([gl, 2 * gl], axis=1)          # (size, 2)
    if axis in (1, -1):
        stacked = stacked.T                            # (2, size)
    out = smpl.evaluate(stacked, axis=axis)
    assert out.shape[axis] == len(smpl.tau)
    assert np.linalg.norm(out) > 0
    back = smpl.fit(out, axis=axis)
    np.testing.assert_allclose(back, stacked, rtol=0, atol=1e-11)


def test_tau_bad_axis(basis, gl):
    smpl = sparse_ir.TauSampling(basis)
    with pytest.raises(IndexError, match="axis 3 is out of bounds"):
        smpl.evaluate(gl, axis=3)
    with pytest.raises(IndexError, match="axis -2 is out of bounds"):
        smpl.evaluate(gl, axis=-2)


def test_tau_bad_length(basis, gl):
    smpl = sparse_ir.TauSampling(basis)
    with pytest.raises(ValueError, match="expected"):
        smpl.evaluate(gl[:-1])
    with pytest.raises(ValueError, match="at least one-dimensional"):
        smpl.evaluate(1.0)


def test_tau_nonfinite_input(basis, gl):
    smpl = sparse_ir.TauSampling(basis)
    poisoned = gl.copy()
    poisoned[2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        smpl.evaluate(poisoned)


def test_tau_sampling_points_validation(basis):
    with pytest.raises(ValueError, match="must not be empty"):
        sparse_ir.TauSampling(basis, sampling_points=np.array([]))
    with pytest.raises(ValueError, match="one-dimensional"):
        sparse_ir.TauSampling(basis, sampling_points=np.zeros((2, 2)))
    with pytest.raises(TypeError, match="must be real-valued"):
        sparse_ir.TauSampling(basis, sampling_points=np.array([1 + 2j]))


# ---------------------------------------------------------------------------
# Finding B/G: dtype normalization and index validation (Matsubara sampling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_matsubara_evaluate_dtype_matrix(basis, gl, dtype):
    smpl = sparse_ir.MatsubaraSampling(basis)
    ref = smpl.evaluate(gl.astype(np.float64))
    assert np.linalg.norm(ref) > 0

    got = smpl.evaluate(gl.astype(dtype))
    assert got.dtype == np.complex128
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
def test_matsubara_fit_dtype_matrix(basis, gl, dtype):
    smpl = sparse_ir.MatsubaraSampling(basis)
    giv = smpl.evaluate(gl)
    ref = smpl.fit(giv)
    assert np.linalg.norm(ref) > 0

    got = smpl.fit(giv.astype(dtype))
    assert got.dtype == np.complex128
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("dtype", REAL_DTYPES + [np.int64])
def test_matsubara_fit_real_input_dtypes(basis, dtype):
    """Real-valued input to a complex fit is widened, not reinterpreted."""
    smpl = sparse_ir.MatsubaraSampling(basis)
    giv = np.arange(1, len(smpl.wn) + 1).astype(dtype)
    ref = smpl.fit(giv.astype(np.complex128))
    assert np.linalg.norm(ref) > 0
    got = smpl.fit(giv)
    assert got.dtype == np.complex128
    assert_close(got, ref, np.float64 if dtype is np.int64 else dtype)


def test_matsubara_noncontiguous(basis, gl):
    smpl = sparse_ir.MatsubaraSampling(basis)
    giv = smpl.evaluate(gl)
    ref = smpl.fit(giv)
    got = smpl.fit(noncontiguous(giv))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_matsubara_indices_are_not_truncated(basis):
    """Finding G: ``int(1.9) == 1`` must not happen silently."""
    with pytest.raises(ValueError, match=r"1\.9"):
        sparse_ir.MatsubaraSampling(basis, sampling_points=[1.9, 3.0])


def test_matsubara_index_parity_is_checked(basis, basis_b):
    with pytest.raises(ValueError, match="must be odd"):
        sparse_ir.MatsubaraSampling(basis, sampling_points=[2, 4])
    with pytest.raises(ValueError, match="must be even"):
        sparse_ir.MatsubaraSampling(basis_b, sampling_points=[1, 3])


def test_matsubara_integral_float_indices_are_accepted(basis):
    smpl = sparse_ir.MatsubaraSampling(basis, sampling_points=[1.0, -1.0, 3.0])
    assert smpl.wn.dtype == np.int64
    np.testing.assert_array_equal(np.sort(smpl.wn), [-1, 1, 3])


def test_matsubara_complex_indices_rejected(basis):
    with pytest.raises(TypeError, match="must be real"):
        sparse_ir.MatsubaraSampling(basis, sampling_points=[1 + 0j])


# ---------------------------------------------------------------------------
# Findings A, B, C: DLR
# ---------------------------------------------------------------------------

def test_dlr_poles_pointer_provenance(basis, gl):
    """Finding C: a non-contiguous/float32 pole array must be normalized."""
    poles = basis.default_omega_sampling_points()
    ref = DiscreteLehmannRepresentation(basis, poles)

    for candidate in (noncontiguous(poles), poles.astype(np.float32),
                      list(poles)):
        dlr = DiscreteLehmannRepresentation(basis, candidate)
        np.testing.assert_allclose(dlr.sampling_points, poles,
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dlr.to_IR(dlr.from_IR(gl)), gl,
                                   rtol=0, atol=1e-6 * np.linalg.norm(gl))
    assert ref.size == len(poles)


def test_dlr_poles_validation(basis):
    with pytest.raises(ValueError, match="must not be empty"):
        DiscreteLehmannRepresentation(basis, np.array([]))
    with pytest.raises(ValueError, match="one-dimensional"):
        DiscreteLehmannRepresentation(basis, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must be finite"):
        DiscreteLehmannRepresentation(basis, np.array([1.0, np.nan]))
    with pytest.raises(TypeError, match="must be real-valued"):
        DiscreteLehmannRepresentation(basis, np.array([1 + 1j]))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dlr_roundtrip_dtype_matrix(basis, gl, dtype):
    dlr = DiscreteLehmannRepresentation(basis)
    ref = dlr.from_IR(gl.astype(np.float64))
    assert np.linalg.norm(ref) > 0

    got = dlr.from_IR(gl.astype(dtype))
    assert_close(got, ref, dtype)

    back = dlr.to_IR(got)
    np.testing.assert_allclose(np.real(back), gl,
                               rtol=0, atol=1e-4 * np.linalg.norm(gl))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dlr_noncontiguous(basis, gl, dtype):
    dlr = DiscreteLehmannRepresentation(basis)
    ref = dlr.from_IR(gl)
    got = dlr.from_IR(noncontiguous(gl.astype(dtype)))
    assert_close(got, ref, dtype)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_dlr_axis_coverage(basis, gl, axis):
    dlr = DiscreteLehmannRepresentation(basis)
    stacked = np.stack([gl, -gl], axis=0 if axis in (1, -1) else 1)
    g_dlr = dlr.from_IR(stacked, axis=axis)
    assert g_dlr.shape[axis] == dlr.size
    back = dlr.to_IR(g_dlr, axis=axis)
    np.testing.assert_allclose(back, stacked, rtol=0,
                               atol=1e-4 * np.linalg.norm(gl))


def test_dlr_bad_axis_and_length(basis, gl):
    dlr = DiscreteLehmannRepresentation(basis)
    with pytest.raises(IndexError, match="out of bounds"):
        dlr.from_IR(gl, axis=1)
    with pytest.raises(ValueError, match="expected"):
        dlr.from_IR(gl[:-1])
    with pytest.raises(ValueError, match="at least one-dimensional"):
        dlr.to_IR(1.0)


def test_dlr_u_are_dlr_functions(basis, gl):
    """Finding A: ``dlr.u`` must be the DLR functions, not the IR ones."""
    dlr = DiscreteLehmannRepresentation(basis)
    g_dlr = dlr.from_IR(gl)

    assert dlr.u.shape == (dlr.size,)
    tau = np.array([0.1, 0.7, 1.3])
    u_vals = dlr.u(tau)
    assert u_vals.shape == (dlr.size, tau.size)
    assert np.linalg.norm(u_vals) > 0

    gtau_dlr = g_dlr @ u_vals
    gtau_ir = gl @ basis.u(tau)
    assert np.linalg.norm(gtau_ir) > 0
    np.testing.assert_allclose(gtau_dlr, gtau_ir, rtol=0,
                               atol=1e-6 * np.linalg.norm(gtau_ir))


def test_dlr_uhat_are_dlr_functions(basis, gl):
    """Finding A, Matsubara side."""
    dlr = DiscreteLehmannRepresentation(basis)
    g_dlr = dlr.from_IR(gl)

    n = np.array([-3, 1, 5])
    uhat_vals = dlr.uhat(n)
    assert uhat_vals.shape == (dlr.size, n.size)
    assert np.linalg.norm(uhat_vals) > 0

    giv_dlr = g_dlr @ uhat_vals
    giv_ir = gl @ basis.uhat(n)
    assert np.linalg.norm(giv_ir) > 0
    np.testing.assert_allclose(giv_dlr, giv_ir, rtol=0,
                               atol=1e-6 * np.linalg.norm(giv_ir))


# ---------------------------------------------------------------------------
# Finding D: shape semantics of FunctionSet.__call__
# ---------------------------------------------------------------------------

def test_function_set_shapes(basis):
    u = basis.u
    size = basis.size

    # Whole set
    assert np.shape(u(0.5)) == (size,)
    assert u(np.array([0.5, 1.0])).shape == (size, 2)
    assert u(np.array([0.5])).shape == (size, 1)
    assert u(np.array([[0.1, 0.2], [0.3, 0.4]])).shape == (size, 2, 2)

    # Single function: the function axis disappears, x's shape survives
    u0 = u[0]
    assert np.isscalar(u0(0.5)) or np.shape(u0(0.5)) == ()
    assert u0(np.array([0.5])).shape == (1,)
    assert u0(np.array([0.5, 1.0])).shape == (2,)
    assert u0(np.array([[0.1, 0.2], [0.3, 0.4]])).shape == (2, 2)

    # A slice of several functions keeps both axes
    assert u[0:3](np.array([0.5, 1.0])).shape == (3, 2)
    assert u[0:3](0.5).shape == (3,)

    # v is defined on omega and follows the same rule
    assert basis.v(np.array([0.0])).shape == (size, 1)


def test_function_set_ft_shapes(basis):
    uhat = basis.uhat
    size = basis.size

    assert np.shape(uhat(1)) == (size,)
    assert uhat(np.array([1])).shape == (size, 1)
    assert uhat(np.array([1, 3])).shape == (size, 2)
    assert uhat(np.array([[1, 3], [5, 7]])).shape == (size, 2, 2)

    uhat0 = uhat[0]
    assert np.shape(uhat0(1)) == ()
    assert uhat0(np.array([1])).shape == (1,)
    assert uhat0(np.array([1, 3])).shape == (2,)
    assert uhat0(np.array([[1, 3], [5, 7]])).shape == (2, 2)

    assert uhat[0:3](np.array([1, 3])).shape == (3, 2)
    assert np.linalg.norm(uhat[0:3](np.array([1, 3]))) > 0


def test_function_set_values_are_consistent_across_shapes(basis):
    u = basis.u
    x = np.array([0.25, 0.5, 1.0, 1.75])
    flat = u(x)
    nested = u(x.reshape(2, 2))
    np.testing.assert_array_equal(nested, flat.reshape(basis.size, 2, 2))
    np.testing.assert_array_equal(u[1](x), [u[1](xi) for xi in x])


@pytest.mark.parametrize("dtype", REAL_DTYPES + [np.int64])
def test_function_set_evaluation_point_dtypes(basis, dtype):
    """Finding B: evaluation points of any real dtype are widened."""
    u = basis.u
    x = np.array([0.0, 1.0, 2.0])
    ref = u(x)
    assert np.linalg.norm(ref) > 0
    np.testing.assert_allclose(u(x.astype(dtype)), ref, rtol=0,
                               atol=1e-6 * np.linalg.norm(ref))
    np.testing.assert_allclose(u(noncontiguous(x)), ref, rtol=1e-14, atol=0)


def test_function_set_rejects_complex_points(basis):
    with pytest.raises(TypeError, match="must be real-valued"):
        basis.u(np.array([0.5 + 0j]))


def test_function_set_ft_rejects_non_integer(basis):
    """Finding G: ``astype(np.int64)`` must not truncate 1.9 to 1."""
    with pytest.raises(ValueError, match=r"1\.9"):
        basis.uhat(np.array([1.9]))
    with pytest.raises(ValueError, match="must be an integer"):
        basis.uhat(np.array([1.0, 2.5]))
    # integral floats are fine
    np.testing.assert_allclose(basis.uhat(np.array([1.0, 3.0])),
                               basis.uhat(np.array([1, 3])),
                               rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# Finding G: index wrap-around
# ---------------------------------------------------------------------------

def test_function_set_index_out_of_range(basis):
    size = basis.size
    with pytest.raises(IndexError, match="out of range"):
        basis.u[size]
    with pytest.raises(IndexError, match="out of range"):
        basis.u[-size - 1]
    with pytest.raises(IndexError, match="out of range"):
        basis.uhat[size + 5]
    with pytest.raises(IndexError, match="out of range"):
        basis.u[[0, size]]


def test_function_set_negative_index_resolves(basis):
    last = basis.u[-1]
    also_last = basis.u[basis.size - 1]
    x = np.array([0.3, 1.1])
    np.testing.assert_array_equal(last(x), also_last(x))
    assert np.linalg.norm(last(x)) > 0


def test_function_set_non_integer_index(basis):
    with pytest.raises(ValueError, match="must be an integer"):
        basis.u[1.5]


def test_resolve_function_indices_slices():
    assert _util.resolve_function_indices(slice(None), 4) == [0, 1, 2, 3]
    assert _util.resolve_function_indices(slice(1, 3), 4) == [1, 2]
    assert _util.resolve_function_indices(-1, 4) == [3]
    assert _util.resolve_function_indices([0, -2], 4) == [0, 2]


# ---------------------------------------------------------------------------
# Finding E: rescale
# ---------------------------------------------------------------------------

def test_rescale_keeps_lambda(basis):
    rescaled = basis.rescale(4.0)
    assert rescaled.beta == 4.0
    assert np.isclose(rescaled.lambda_, basis.lambda_)
    assert np.isclose(rescaled.beta * rescaled.wmax, basis.lambda_)
    assert rescaled.statistics == basis.statistics
    assert rescaled.size == basis.size
    np.testing.assert_allclose(rescaled.s / rescaled.s[0],
                               basis.s / basis.s[0], rtol=1e-10, atol=0)

    # The rescaled basis is usable
    smpl = sparse_ir.TauSampling(rescaled)
    al = np.ones(rescaled.size)
    assert np.linalg.norm(smpl.evaluate(al)) > 0


def test_rescale_rejects_nonpositive_beta(basis):
    with pytest.raises(ValueError, match="must be positive"):
        basis.rescale(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        basis.rescale(-1.0)


# ---------------------------------------------------------------------------
# Finding F: fermionic TauConst
# ---------------------------------------------------------------------------

def test_fermionic_tau_const_rejected():
    import sparse_ir.augment as aug

    fermionic = sparse_ir.FiniteTempBasis('F', 2.0, 5.0, 1e-6)
    with pytest.raises(ValueError, match="only allowed for a bosonic basis"):
        aug.TauConst(2.0, 'F')
    with pytest.raises(ValueError, match="only allowed for a bosonic basis"):
        aug.AugmentedBasis(fermionic, aug.TauConst)

    # The bosonic case still works and is full rank in Matsubara
    bosonic = sparse_ir.FiniteTempBasis('B', 2.0, 5.0, 1e-6)
    augmented = aug.AugmentedBasis(bosonic, aug.TauConst)
    assert augmented.size == bosonic.size + 1
    n = bosonic.default_matsubara_sampling_points()
    matrix = augmented.uhat(n)
    assert np.linalg.norm(matrix[0]) > 0


# ---------------------------------------------------------------------------
# Finding H / python.md: the public API is importable
# ---------------------------------------------------------------------------

def test_all_names_exist():
    missing = [name for name in sparse_ir.__all__
               if not hasattr(sparse_ir, name)]
    assert not missing, f"__all__ names missing from the module: {missing}"


def test_star_import_works():
    namespace = {}
    exec("from sparse_ir import *", namespace)
    for name in sparse_ir.__all__:
        assert name in namespace


# ---------------------------------------------------------------------------
# Augmented bases: an augmentation undefined in tau cannot be tau-sampled
# ---------------------------------------------------------------------------

def test_tau_sampling_of_vertex_basis_raises():
    """MatsubaraConst is NaN in imaginary time; sampling it must not silently
    hand NaNs to the C factorization."""
    import sparse_ir.augment as aug

    bosonic = sparse_ir.FiniteTempBasis('B', 2.0, 5.0, 1e-6)
    vertex = aug.AugmentedBasis(bosonic, aug.MatsubaraConst)

    with pytest.raises(ValueError, match="undefined in imaginary"):
        sparse_ir.TauSampling(vertex)

    # The Matsubara side of the same basis is fine
    smpl = sparse_ir.MatsubaraSampling(vertex)
    al = np.ones(vertex.size)
    assert np.linalg.norm(smpl.evaluate(al)) > 0


def test_augmented_tau_sampling_roundtrip():
    """A bosonic TauConst/TauLinear augmented basis still round-trips in tau."""
    import sparse_ir.augment as aug

    bosonic = sparse_ir.FiniteTempBasis('B', 2.0, 5.0, 1e-6)
    augmented = aug.AugmentedBasis(bosonic, aug.TauConst, aug.TauLinear)
    smpl = sparse_ir.TauSampling(augmented)

    rng = np.random.RandomState(1234)
    al = rng.randn(augmented.size)
    ax = smpl.evaluate(al)
    assert np.linalg.norm(ax) > 0
    np.testing.assert_allclose(smpl.fit(ax), al, rtol=0, atol=1e-8)
