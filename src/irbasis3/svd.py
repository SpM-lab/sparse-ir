# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
from warnings import warn
import numpy as np
import scipy.linalg.interpolative as intp_decomp

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
    from scipy.linalg.lapack import dgejsv as _lapack_dgejsv
except ImportError:
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
    a_matrix = np.asarray(a_matrix)
    m, n = a_matrix.shape
    if n_sv_hint is None:
        n_sv_hint = min(m, n)
    n_sv_hint = min(m, n, n_sv_hint)

    if _ddouble is not None and a_matrix.dtype == _ddouble:
        u, s, v = _ddouble_svd_trunc(a_matrix)
    elif strategy == 'fast':
        u, s, v = _idsvd(a_matrix, n_sv=n_sv_hint)
    elif strategy == 'default':
        # Usual (simple) SVD
        u, s, vh = np.linalg.svd(a_matrix, full_matrices=False)
        v = vh.T.conj()
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


def _idsvd(a, n_sv):
    # Use interpolative decomposition, since it scales favorably to a full
    # SVD when we are only interested in a small subset of singular values.
    # NOTE: this returns the right singular vectors, not their conjugate!
    intp_decomp.seed(4711)
    return intp_decomp.svd(a, n_sv)


def _dgejsv(a, mode='A'):
    """Compute SVD using the (more accurate) Jacobi method"""
    # GEJSV can only handle tall matrices
    m, n = a.shape
    if m < n:
        u, s, v = _dgejsv(a.T, mode)
        return v, s, u

    mode = mode.upper()
    joba = dict(zip("CEFGAR", range(6)))[mode]
    s, u, v, _stat, istat, info = _lapack_dgejsv(a, joba)
    if info < 0:
        raise ValueError("LAPACK error - invalid parameter")
    if istat[2] != 0:
        warn("a contained denormalized floats - possible loss of accuracy",
             UserWarning, 2)
    if info > 0:
        warn("SVD did not converge", UserWarning, 2)
    return u, s, v


def _ddouble_svd_trunc(a):
    """Truncated SVD with double double precision"""
    if _xprec_linalg is None:
        raise RuntimeError("require xprec package for this precision")
    u, s, vh = _xprec_linalg.svd_trunc(a)
    return u, s, vh.T
