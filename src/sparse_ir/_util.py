# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import functools
import numpy as np


def ravel_argument(last_dim=False):
    """Wrap function operating on 1-D numpy array to allow arbitrary shapes.

    This decorator allows to write functions which only need to operate over
    one-dimensional (ravelled) arrays.  This often simplifies the "shape logic"
    of the computation.
    """
    return lambda fn: RavelArgumentDecorator(fn, last_dim)


class RavelArgumentDecorator(object):
    def __init__(self, inner, last_dim=False):
        self.instance = None
        self.inner = inner
        self.last_dim = last_dim
        functools.update_wrapper(self, inner)

    def __get__(self, instance, _owner=None):
        self.instance = instance
        return self

    def __call__(self, x):
        x = np.asarray(x)
        if self.instance is None:
            res = self.inner(x.ravel())
        else:
            res = self.inner(self.instance, x.ravel())
        if self.last_dim:
            return res.reshape(res.shape[:-1] + x.shape)
        else:
            return res.reshape(x.shape + res.shape[1:])


# Element-type kinds that can be widened to float64 without losing
# information about what the caller meant: bool, signed/unsigned integer,
# and floating point.  Complex is deliberately excluded.
_REAL_KINDS = "biuf"


def check_reduced_matsubara(n, zeta=None):
    """Checks that ``n`` is a reduced Matsubara frequency.

    Check that the argument is a reduced Matsubara frequency, which is an
    integer obtained by scaling the freqency `w[n]` as follows::

        beta / np.pi * w[n] == 2 * n + zeta

    Note that this means that instead of a fermionic frequency (``zeta == 1``),
    we expect an odd integer, while for a bosonic frequency (``zeta == 0``),
    we expect an even one.  If ``zeta`` is omitted, any one is fine.

    Raises:
        TypeError: if ``n`` is complex.
        ValueError: if ``n`` is not integral (naming the offending value) or
            has the wrong parity.
    """
    n = np.asarray(n)
    if n.dtype.kind == 'c':
        raise TypeError(
            f"reduced Matsubara frequency must be real, got dtype {n.dtype}")
    if not np.issubdtype(n.dtype, np.integer):
        if n.dtype.kind not in _REAL_KINDS:
            raise TypeError(
                f"reduced Matsubara frequency must be numeric, "
                f"got dtype {n.dtype}")
        nfloat = np.asarray(n, dtype=np.float64)
        if not np.all(np.isfinite(nfloat)):
            raise ValueError(
                "reduced Matsubara frequency must be finite, got "
                f"{nfloat[~np.isfinite(nfloat)][0]!r}")
        n = np.rint(nfloat).astype(np.int64)
        bad = n != nfloat
        if bad.any():
            offending = np.atleast_1d(nfloat)[np.atleast_1d(bad)][0]
            raise ValueError(
                "reduced Matsubara frequency must be an integer, got "
                f"{offending!r} (no truncation is performed)")
    if zeta is not None:
        parity = np.asarray(n) & 1
        if not (parity == zeta).all():
            expected = "odd" if zeta else "even"
            offending = np.atleast_1d(n)[np.atleast_1d(parity != zeta)][0]
            raise ValueError(
                f"reduced Matsubara frequency must be {expected} for "
                f"zeta={zeta}, got {offending!r}")
    return n


def _check_finite(arr, name):
    if arr.size and not np.all(np.isfinite(arr)):
        pos = tuple(int(i) for i in np.argwhere(~np.isfinite(arr))[0])
        raise ValueError(
            f"{name} must be finite, but contains {arr[pos]!r} at index "
            f"{pos[0] if arr.ndim == 1 else pos}")
    return arr


def as_boundary_real(a, name="array", check_finite=True):
    """Normalize ``a`` into a C-contiguous ``float64`` array for the C boundary.

    The returned object is the one whose pointer must be handed to C: a
    pointer taken from the *original* array would be a defect if a copy was
    made here (see ``rules/ffi-boundary.md``, Pointer Provenance).

    Raises:
        TypeError: if ``a`` is complex or of a non-numeric element type.
        ValueError: if ``a`` contains a non-finite value.
    """
    arr = np.asarray(a)
    if arr.dtype.kind == 'c':
        raise TypeError(
            f"{name} must be real-valued, got dtype {arr.dtype}; "
            "the C entry point takes a double pointer")
    if arr.dtype.kind not in _REAL_KINDS:
        raise TypeError(f"{name} has unsupported dtype {arr.dtype}")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if check_finite:
        _check_finite(out, name)
    return out


def as_boundary_complex(a, name="array", check_finite=True):
    """Normalize ``a`` into a C-contiguous ``complex128`` array.

    ``complex64`` is *not* ``complex128``: passing its buffer through a
    ``c_double_complex`` pointer would read twice as many bytes per element
    as were allocated, so the conversion here is explicit and the pointer
    must be taken from the returned object.
    """
    arr = np.asarray(a)
    if arr.dtype.kind not in _REAL_KINDS + "c":
        raise TypeError(f"{name} has unsupported dtype {arr.dtype}")
    out = np.ascontiguousarray(arr, dtype=np.complex128)
    if check_finite:
        _check_finite(out, name)
    return out


def as_boundary_matsubara(n, name="Matsubara indices", zeta=None):
    """Normalize reduced Matsubara indices into a C-contiguous ``int64`` array.

    Validates integrality (and, if ``zeta`` is given, parity) *before* the
    conversion, so a non-integral index raises instead of being truncated.
    """
    checked = check_reduced_matsubara(n, zeta=zeta)
    return np.ascontiguousarray(checked, dtype=np.int64)


def normalize_axis(axis, ndim):
    """Resolve a possibly negative ``axis`` against ``ndim`` and range-check it.

    The C API takes a non-negative target dimension; a negative Python axis
    must be resolved here rather than handed through.
    """
    axis = int(axis)
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise IndexError(
            f"axis {axis} is out of bounds for an array of dimension {ndim} "
            f"(valid: {-ndim} .. {ndim - 1})")
    return resolved


def resolve_function_indices(index, size):
    """Resolve a basis-function index, list of indices, or slice.

    Negative indices are resolved explicitly (Python semantics); an index
    outside ``[-size, size)`` raises :class:`IndexError` naming the requested
    index and the valid range.  No modulo wrap-around is performed.
    """
    if isinstance(index, slice):
        return list(range(*index.indices(size)))

    idx = np.asarray(index)
    if idx.dtype.kind == 'c':
        raise TypeError(
            f"basis-function index must be an integer, got dtype {idx.dtype}")
    if idx.dtype.kind not in _REAL_KINDS:
        raise TypeError(
            f"basis-function index must be an integer, got dtype {idx.dtype}")
    if idx.dtype.kind == 'f':
        rounded = np.rint(idx)
        if not np.array_equal(rounded, idx):
            offending = np.atleast_1d(idx)[np.atleast_1d(rounded != idx)][0]
            raise ValueError(
                f"basis-function index must be an integer, got {offending!r} "
                "(no truncation is performed)")
        idx = rounded.astype(np.int64)

    flat = np.atleast_1d(idx).ravel()
    resolved = []
    for i in flat.tolist():
        j = i + size if i < 0 else i
        if not 0 <= j < size:
            raise IndexError(
                f"basis-function index {i} is out of range for a function set "
                f"of size {size} (valid: {-size} .. {size - 1})")
        resolved.append(int(j))
    return resolved


def check_range(x, xmin, xmax):
    """Checks each element is in range [xmin, xmax]"""
    x = np.asarray(x)
    if not (x >= xmin).all():
        raise ValueError(f"Some x violate lower bound {xmin}")
    if not (x <= xmax).all():
        raise ValueError(f"Some x violate upper bound {xmax}")
    return x


def normalize_tau(statistics, tau, beta):
    """Normalize τ to [0, β] with statistics-dependent periodicity.
    
    Handles boundary conditions based on statistics:
    - Fermions ('F'): Anti-periodic G(τ + β) = -G(τ)
    - Bosons ('B'): Periodic G(τ + β) = G(τ)
    
    This function maps τ values from the range [-β, β] to [0, β] with
    appropriate sign factors, following the periodicity rules.
    
    Arguments:
        statistics (str):
            'F' for Fermionic or 'B' for Bosonic statistics.
        tau (array_like):
            Imaginary time value(s) in range [-β, β].
        beta (float):
            Inverse temperature.
            
    Returns:
        tuple[ndarray, ndarray]:
            (tau_normalized, sign) where:
            - tau_normalized: τ values mapped to [0, β]
            - sign: Sign factor (±1) for periodicity
            
    Raises:
        ValueError: If tau is outside [-β, β] or statistics is invalid.
        
    Special cases:
        - Negative zero (τ = -0.0) is treated as τ = β with appropriate sign
        - For τ in [0, β]: returns (τ, +1)
        - For τ in [-β, 0): returns (τ + β, sign) where sign depends on statistics
        
    .. versionadded:: 1.2
    """
    tau = np.asarray(tau, dtype=np.float64)
    beta = float(beta)
    
    if statistics not in ('F', 'B'):
        raise ValueError("statistics must be 'F' (Fermionic) or 'B' (Bosonic)")
    
    if np.any(tau < -beta) or np.any(tau > beta):
        raise ValueError(f"τ must be in [-β, β] = [{-beta}, {beta}]")
    
    # Handle negative zero: τ = -0.0 → τ = β
    is_neg_zero = (tau == 0.0) & np.signbit(tau)
    
    tau_normalized = np.where(is_neg_zero, beta, tau)
    sign = np.ones_like(tau, dtype=np.float64)
    
    if statistics == 'F':
        # Fermionic: anti-periodic
        sign = np.where(is_neg_zero, -1.0, sign)
    else:  # statistics == 'B'
        # Bosonic: periodic
        sign = np.where(is_neg_zero, 1.0, sign)
    
    # Normalize negative tau to [0, β]
    mask_neg = tau_normalized < 0
    tau_normalized = np.where(mask_neg, tau_normalized + beta, tau_normalized)
    
    if statistics == 'F':
        sign = np.where(mask_neg, -sign, sign)
    
    return tau_normalized, sign


def check_svd_result(svd_result, matrix_shape=None):
    """Checks that argument is a valid SVD triple (u, s, vH)"""
    u, s, vH = map(np.asarray, svd_result)
    m_u, k_u = u.shape
    k_s, = s.shape
    k_v, n_v = vH.shape
    if k_u != k_s or k_s != k_v:
        raise ValueError("shape mismatch between SVD elements:"
                         f"({m_u}, {k_u}) x ({k_s}) x ({k_v}, {n_v})")
    if matrix_shape is not None:
        m, n = matrix_shape
        if m_u != m or n_v != n:
            raise ValueError(f"shape mismatch between SVD ({m_u}, {n_v}) "
                             f"and matrix ({m}, {n})")
    return u, s, vH
