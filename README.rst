sparse-ir - A library for the intermediate representation of propagators
========================================================================
This library provides routines for constructing and working with the
intermediate representation of correlation functions.  It provides:

 - on-the-fly computation of basis functions for arbitrary cutoff Λ
 - basis functions and singular values are accurate to full precision
 - routines for sparse sampling


Installation
------------
Install via `pip <https://pypi.org/project/sparse-ir>`_::

    pip install sparse-ir[xprec]

The above line is the recommended way to install `sparse-ir`.  It automatically
installs the `xprec <https://github.com/tuwien-cms/xprec>`_ package, which
allows one to compute the IR basis functions with greater accuracy.  If you do
not want to do this, simply remove the string ``[xprec]`` from the above command.

Install via `conda <https://anaconda.org/spm-lab/sparse-ir>`_::

    conda install -c spm-lab sparse-ir xprec

Other than the optional xprec dependency, sparse-ir requires only
`numpy <https://numpy.org/>`_ and `scipy <https://scipy.org/>`_.


Quick start
-----------
Check out our comprehensive tutorial: `<https://spm-lab.github.io/sparse-ir-tutorial>`_!

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

Refer to the `online documentation`_ for more details.

.. _online documentation: https://sparse-ir.readthedocs.io
.. _intermediate representation: https://arxiv.org/abs/2106.12685
.. _singular value expansion: https://w.wiki/3poQ


License and citation
-------------------------------
This software is released under the MIT License.  See LICENSE.txt for details.

If you find the intermediate representation, sparse sampling, or this software
useful in your research, please consider citing the following papers:

 - Hiroshi Shinaoka et al., `Phys. Rev. B 96, 035147`_  (2017)
 - Jia Li et al., `Phys. Rev. B 101, 035144`_ (2020)
 - Markus Wallerberger et al., `arXiv 2206.11762`_ (2022)

If you are discussing sparse sampling in your research specifically, please
also consider citing an independently discovered, closely related approach, the
MINIMAX isometry method (Merzuk Kaltak and Georg Kresse,
`Phys. Rev. B 101, 205145`_, 2020).

.. _Phys. Rev. B 96, 035147: https://doi.org/10.1103/PhysRevB.96.035147
.. _Phys. Rev. B 101, 035144: https://doi.org/10.1103/PhysRevB.101.035144
.. _arXiv 2206.11762: https://doi.org/10.48550/arXiv.2206.11762
.. _Phys. Rev. B 101, 205145: https://doi.org/10.1103/PhysRevB.101.205145
