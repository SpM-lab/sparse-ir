# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT


# Sample codes in the SciPost review paper

def test_sample2():
    # Compute IR basis for fermions and \beta = 100 and \omega_max = 10
    import sparse_ir
    import numpy as np

    lambda_ = 1000
    beta = 100
    wmax = lambda_/beta
    eps = 1e-8 # cut-off value for singular values
    b = sparse_ir.FiniteTempBasis('F', beta, wmax, eps=eps)

    x = y = 0.1
    tau = 0.5 * beta * (x+1)
    omega = wmax * y

    # All singular values
    print("singular values: ", b.s)
    print("U_0(0.1)", b.u[0](tau))
    print("V_0(0.1)", b.v[0](omega))

    print("n-th derivative of U_l(tau) and V_l(omega)")
    for n in range(1,3):
        u_n = b.u.deriv(n)
        v_n = b.v.deriv(n)
        print(" n= ", n, u_n[0](tau))
        print(" n= ", n, v_n[0](omega))

    # Compute u_{ln} as a matrix for the first
    # 10 non-nagative fermionic Matsubara frequencies
    # Fermionic/bosonic frequencies are denoted by odd/even integers.
    hatF_t = b.uhat(2*np.arange(10)+1)
    print(hatF_t.shape)

def test_sample3():
    import sparse_ir
    import numpy as np
    from numpy.fft import fftn, ifftn

    beta = 1e+3
    lambda_ = 1e+5

    wmax = lambda_/beta
    eps = 1e-15
    print("wmax", wmax)

    b = sparse_ir.FiniteTempBasis('F', beta , wmax, eps=eps)
    print("Number of basis functions", b.size)

    # Sparse sampling in tau
    smpl_tau = sparse_ir.TauSampling(b)

    # Sparse sampling in Matsubara frequencies
    smpl_matsu = sparse_ir.MatsubaraSampling(b)

    # Parameters
    nk_lin = 64
    U, kps  = 2.0, np.array([nk_lin, nk_lin])
    nw = smpl_matsu.sampling_points.size
    ntau = smpl_tau.sampling_points.size

    # Generate k mesh and non-interacting band energies
    nk = np.prod(kps)
    kgrid = [2*np.pi*np.arange(kp)/kp for kp in kps]
    k1, k2 = np.meshgrid(*kgrid, indexing='ij')
    ek = -2*(np.cos(k1) + np.cos(k2))
    iw = 1j*np.pi*smpl_matsu.sampling_points/beta

    # G(iw, k): (nw, nk)
    gkf = 1.0 / (iw[:,None] - ek.ravel()[None,:])

    # G(l, k): (L, nk)
    gkl = smpl_matsu.fit(gkf)

    # G(tau, k): (ntau, nk)
    gkt = smpl_tau.evaluate(gkl)

    # G(tau, r): (ntau, nk)
    grt = np.fft.fftn(gkt.reshape(ntau, *kps), axes=(1,2)).\
            reshape(ntau, nk)

    # Sigma(tau, r): (ntau, nk)
    srt = U*U*grt*grt*grt[::-1,:]

    # Sigma(l, r): (L, nk)
    srl = smpl_tau.fit(srt)

    # Sigma(iw, r): (nw, nk)
    srf = smpl_matsu.evaluate(srl)

    # Sigma(l, r): (L, nk)
    srl = smpl_tau.fit(srt)

    # Sigma(iw, r): (nw, nk)
    srf = smpl_matsu.evaluate(srl)

    # Sigma(iw, k): (nw, kps[0], kps[1])
    srf = srf.reshape(nw, *kps)
    skf = ifftn(srf, axes=(1,2))/nk**2
