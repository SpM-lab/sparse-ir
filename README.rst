sparse-ir - A library for the intermediate representation of propagators
========================================================================
This library provides routines for constructing and working with the
intermediate representation of correlation functions.  It provides:

 - on-the-fly computation of basis functions for arbitrary cutoff Λ
 - basis functions and singular values are accurate to full precision
 - routines for sparse sampling


Installation
------------
Install via `pip <https://pip.pypa.io/en/stable/getting-started>`_::

    pip install sparse-ir[xprec]

The above line is the recommended way to install `sparse-ir`.  It automatically
installs the `xprec`_ package, which allows to compute the IR basis functions
with greater accuracy.  If you do not want to do this, simply remove the string
``[xprec]`` from the above command.

.. _xprec: https://github.com/tuwien-cms/xprec


Quick start
-----------
Please refer to the `online documentation <https://sparse-ir.readthedocs.io>`_
for more details.

Here is some python code illustrating the API::

    # Compute IR basis for fermions and β = 10, W <= 4.2
    import sparse_ir, numpy
    basis = sparse_ir.FiniteTempBasis(statistics='F', beta=10, wmax=4.2)

    # Assume spectrum is a single pole at ω = 2.5, compute G(iw)
    # on the first few Matsubara frequencies. (Fermionic/bosonic Matsubara
    # frequencies are denoted by odd/even integers.)
    gl = basis.s * basis.v(2.5)
    giw = gl @ basis.uhat([1, 3, 5, 7])

    # Reconstruct same coefficients from sparse sampling on the Matsubara axis:
    smpl_iw = sparse_ir.MatsubaraSampling(basis)
    giw = -1/(1j * numpy.pi/basis.beta * smpl_iw.wn - 2.5)
    gl_rec = smpl_iw.fit(giw)

You may want to start with reading up on the `intermediate representation`_.
It is tied to the analytic continuation of bosonic/fermionic spectral
functions from (real) frequencies to imaginary time, a transformation mediated
by a kernel ``K``.  The kernel depends on a cutoff, which you should choose to
be ``lambda_ >= beta * W``, where ``beta`` is the inverse temperature and ``W``
is the bandwidth.

One can now perform a `singular value expansion`_ on this kernel, which
generates two sets of orthonormal basis functions, one set ``v[l](w)`` for
real frequency side ``w``, and one set ``u[l](tau)`` for the same obejct in
imaginary (Euclidean) time ``tau``, together with a "coupling" strength
``s[l]`` between the two sides.

By this construction, the imaginary time basis can be shown to be *optimal* in
terms of compactness.

.. _intermediate representation: https://arxiv.org/abs/2106.12685
.. _singular value expansion: https://w.wiki/3poQ


License
-------
This software is released under the MIT License.  See LICENSE.txt.
