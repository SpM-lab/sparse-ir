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


def check_reduced_matsubara(n, zeta=None):
    """Checks that ``n`` is a reduced Matsubara frequency.

    Check that the argument is a reduced Matsubara frequency, which is an
    integer obtained by scaling the freqency `w[n]` as follows::

        beta / np.pi * w[n] == 2 * n + zeta

    Note that this means that instead of a fermionic frequency (``zeta == 1``),
    we expect an odd integer, while for a bosonic frequency (``zeta == 0``),
    we expect an even one.  If ``zeta`` is omitted, any one is fine.
    """
    n = np.asarray(n)
    if not np.issubdtype(n.dtype, np.integer):
        nfloat = n
        n = nfloat.astype(int)
        if not (n == nfloat).all():
            raise ValueError("reduced frequency n must be integer")
    if zeta is not None:
        if not (n & 1 == zeta).all():
            raise ValueError("n have wrong parity")
    return n


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
