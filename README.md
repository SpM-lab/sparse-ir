irbasis3 - A library for the intermediate representation of propagators
=======================================================================
This library provides routines for constructing and working with the
intermediate representation of correlation functions.  It provides:

 - on-the-fly computation of basis functions for arbitrary cutoff Λ
 - basis functions and singular values are accurate to full precision
 - routines for sparse sampling

Installation
------------

    pip install irbasis3 xprec

Though not strictly required, we recommend installing the `xprec` alongside
`irbasis3` as it allows to compute the IR basis functions with greater
accuracy.

Quick start
-----------
Here is some python code illustrating the API:

    # Compute IR basis for fermions and β = 10, W <= 4.2
    import irbasis3
    K = irbasis3.KernelFFlat(lambda_=42)
    basis = irbasis3.FiniteTempBasis(K, statistics='F', beta=10)

    # Assume spectrum is a single pole at ω = 2.5, compute G(iw)
    # on the first few Matsubara frequencies
    gl = basis.s * basis.v(2.5)
    giw = gl @ basis.uhat([1, 3, 5, 7])

You may want to start with reading up on the [intermediate representation].
It is tied to the analytic continuation of bosonic/fermionic spectral
functions from (real) frequencies to imaginary time, a transformation mediated
by a kernel `K`.  The kernel depends on a cutoff, which you should choose to be
`lambda_ >= β*W`, where `β` is the inverse temperature and `W` is the bandwidth:

One can now perform a [singular value expansion] on this kernel, which
generates two sets of orthonormal basis functions, one set `v[l](w)` for
real frequency side `w`, and one set `u[l](tau)` for the same obejct in
imaginary (Euclidean) time `tau`, together with a "coupling" strength `s[l]`
between the two sides.

By this construction, the imaginary time basis can be shown to be *optimal* in
terms of compactness.


[intermediate representation]: https://arxiv.org/abs/2106.12685
[singular value expansion]: https://en.wikipedia.org/wiki/Singular_value_decomposition

License
-------
This software is released under the MIT License.  See LICENSE.txt.
