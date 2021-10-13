# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
from warnings import warn
import numpy as np
import scipy.linalg.interpolative as intp_decomp
import scipy.linalg.lapack as sp_lapack

try:
    from xprec import ddouble as _ddouble, finfo
    import xprec.linalg as _xprec_linalg

    MAX_DTYPE = _ddouble
    MAX_EPS = 5e-32
except ImportError:
    _ddouble = None
    _xprec_linalg = None

    MAX_DTYPE = np.double
    MAX_EPS = np.finfo(MAX_DTYPE).eps
    finfo = np.finfo

try:
    _lapack_dgejsv = sp_lapack.dgejsv
except AttributeError:
    _lapack_dgejsv = None


def compute(a_matrix, n_sv_hint=None, strategy='fast'):
    """Compute thin/truncated singular value decomposition

    Computes the thin/truncated singular value decomposition of a matrix `A`
    into `U`, `s`, `V`:

        A == (U * s) @ V.T

    Depending on the strategy, only as few as `n_sv_hint` most significant
    singular values may be returned, but applications should not rely on this
    behvaiour.  The `strategy` parameter can be `fast` (RRQR/t-SVD),
    `default` (full SVD) or `accurate` (Jacobi rotation SVD).
    """
    if n_sv_hint is None:
        n_sv_hint = a_matrix.shape[-1]
    n_sv_hint = min(n_sv_hint, *a_matrix.shape[-2:])

    if _ddouble is not None and a_matrix.dtype == _ddouble:
        u, s, v = _ddouble_svd_trunc(a_matrix)
    elif strategy == 'fast':
        u, s, v = _idsvd(a_matrix, n_sv=n_sv_hint)
    elif strategy == 'default':
        # Usual (simple) SVD
        u, s, vh = np.linalg.svd(a_matrix, full_matrices=False)
        v = vh.swapaxes(-2, -1).conj()
    elif strategy == 'accurate':
        # Most accurate SVD
        if _lapack_dgejsv is None:
            warn("dgejsv (accurate SVD) is not available. Falling back to\n"
                 "default SVD.  Expect slightly lower precision.\n"
                 "Use xprec or scipy >= 1.5 to fix the issue.")
            return compute(a_matrix, n_sv_hint, strategy='default')
        u, s, v = _dgejsv(a_matrix, mode='F')
    else:
        raise ValueError("invalid strategy:" + str(strategy))

    return u, s, v


def _vectorize(svd):
    def _homogenize(a):
        maxshape = np.max([ai.shape for ai in a], axis=0)
        res = np.zeros((len(a), *maxshape), a[0].dtype)
        for i, ai in enumerate(a):
            res[(i, *map(slice, ai.shape))] = ai
        return res

    def _vectorized_svd_func(A, *args, **kwds):
        A = np.asarray(A)
        *rest, m, n = A.shape
        if not rest:
            return svd(A, *args, **kwds)

        # As vectors of potentially different sizes
        U, s, V = zip(*(svd(A[I], *args, **kwds) for I in np.ndindex(*rest)))
        return _homogenize(U), _homogenize(s), _homogenize(V)

    return _vectorized_svd_func


@_vectorize
def _idsvd(a, n_sv):
    # Use interpolative decomposition, since it scales favorably to a full
    # SVD when we are only interested in a small subset of singular values.
    # NOTE: this returns the right singular vectors, not their conjugate!
    intp_decomp.seed(4711)
    return intp_decomp.svd(a, n_sv)


@_vectorize
def _dgejsv(a, mode='A'):
    """Compute SVD using the (more accurate) Jacobi method"""
    # GEJSV can only handle tall matrices
    m, n = a.shape
    if m < n:
        u, s, v = _dgejsv(a.T, mode)
        return v, s, u

    mode = mode.upper()
    joba = dict(zip("CEFGAR", range(6)))[mode]
    s, u, v, _stat, istat, info = sp_lapack.dgejsv(a, joba)
    if info < 0:
        raise ValueError("LAPACK error - invalid parameter")
    if istat[2] != 0:
        warn("a contained denormalized floats - possible loss of accuracy",
             UserWarning, 2)
    if info > 0:
        warn("SVD did not converge", UserWarning, 2)
    return u, s, v


@_vectorize
def _ddouble_svd_trunc(a):
    """Truncated SVD with double double precision"""
    if _xprec_linalg is None:
        raise RuntimeError("require xprec package for this precision")
    u, s, vh = _xprec_linalg.svd_trunc(a)
    return u, s, vh.T


def truncate(u, s, v, rtol=0, lmax=None):
    """Truncate singular value expansion.

    Arguments:

     - `u`, `s`, `v : Thin singular value expansion
     - `rtol` : If given, only singular values satisfying `s[l]/s[0] > rtol`
       are retained.
     - `lmax` : If given, at most the `lmax` most significant singular values
       are retained.
    """
    cut = _singular_values_cut(s, rtol, lmax).max()
    if cut != s.shape[-1]:
        u = u[..., :, :cut]
        s = s[..., :cut]
        v = v[..., :, :cut]
    return u, s, v


def _singular_values_cut(sl, rtol=0, lmax=None, even_odd=None):
    """Return how many singular values match the given criteria"""
    sl = np.asarray(sl)
    if rtol < 0 or rtol > 1:
        raise ValueError("relative tolerance most be between 0 and 1")

    sl = sl[..., :lmax] / sl[..., :1]
    cut = (sl >= rtol).sum(-1)
    if even_odd is not None:
        cut -= (cut % 2 != even_odd)
    return cut
