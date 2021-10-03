# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

import irbasis3
from irbasis3 import sampling


def test_decomp():
    rng = np.random.RandomState(4711)
    A = rng.randn(49, 39)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]
    np.testing.assert_allclose(A, np.asarray(Ad), atol=1e-15 * norm_A, rtol=0)

    x = rng.randn(39)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    x = rng.randn(39, 3)
    np.testing.assert_allclose(A @ x, Ad @ x, atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)

    y = rng.randn(49, 2)
    np.testing.assert_allclose(np.linalg.lstsq(A, y, rcond=None)[0],
                               Ad.lstsq(y), atol=1e-14 * norm_A, rtol=0)


def test_axis():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(2, 21, 4, 7)
    ref = np.tensordot(A, x, (-1,1)).transpose((1,0,2,3))
    np.testing.assert_allclose(
            Ad.matmul(x, axis=1), ref,
            atol=1e-13 * norm_A, rtol=0)

def test_axis0():
    rng = np.random.RandomState(4712)
    A = rng.randn(17, 21)

    Ad = sampling.DecomposedMatrix(A)
    norm_A = Ad.s[0] / Ad.s[-1]

    x = rng.randn(21, 2)

    np.testing.assert_allclose(
            Ad.matmul(x, axis=0), A@x,
            atol=1e-13 * norm_A, rtol=0)

    np.testing.assert_allclose(
            Ad.matmul(x), A@x,
            atol=1e-13 * norm_A, rtol=0)


def test_tau_noise():
    K = irbasis3.KernelFFlat(100)
    basis = irbasis3.IRBasis(K, 'F')
    smpl = irbasis3.TauSampling(basis)
    rng = np.random.RandomState(4711)

    rhol = basis.v([-.999, -.01, .5]) @ [0.8, -.2, 0.5]
    Gl = basis.s * rhol
    Gl_magn = np.linalg.norm(Gl)
    Gtau = smpl.evaluate(Gl)

    noise = 1e-5
    Gtau_n = Gtau +  noise * np.linalg.norm(Gtau) * rng.randn(*Gtau.shape)
    Gl_n = smpl.fit(Gtau_n)

    np.testing.assert_allclose(Gl, Gl_n, atol=12 * noise * Gl_magn, rtol=0)


def test_wn_noise():
    K = irbasis3.KernelBFlat(99)
    basis = irbasis3.IRBasis(K, 'B')
    smpl = irbasis3.MatsubaraSampling(basis)
    rng = np.random.RandomState(4711)

    rhol = basis.v([-.999, -.01, .5]) @ [0.8, -.2, 0.5]
    Gl = basis.s * rhol
    Gl_magn = np.linalg.norm(Gl)
    Giw = smpl.evaluate(Gl)

    noise = 1e-5
    Giw_n = Giw +  noise * np.linalg.norm(Giw) * rng.randn(*Giw.shape)
    Gl_n = smpl.fit(Giw_n)
    np.testing.assert_allclose(Gl, Gl_n, atol=12 * noise * Gl_magn, rtol=0)