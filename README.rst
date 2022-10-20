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

To manually install the current development version, you can use the following::

   # Only recommended for developers - no automatic updates!
   git clone https://github.com/SpM-lab/sparse-ir
   pip install -e sparse-ir/[xprec]

Documentation and tutorial
--------------------------
Check out our `comprehensive tutorial`_, where we self-contained
notebooks for several many-body methods - GF(2), GW, Eliashberg equations,
Lichtenstein formula, FLEX, ... - are presented.

Refer to the `API documentation`_ for more details on how to work
with the python library.

There is also a `Julia library`_ and (currently somewhat restricted)
`Fortran library`_ available for the IR basis and sparse sampling.

.. _comprehensive tutorial: https://spm-lab.github.io/sparse-ir-tutorial
.. _API documentation: https://sparse-ir.readthedocs.io
.. _Julia library: https://github.com/SpM-lab/SparseIR.jl
.. _Fortran library: https://github.com/SpM-lab/sparse-ir-fortran

Getting started
---------------
Here is a full second-order perturbation theory solver (GF(2)) in a few
lines of Python code::

    # Construct the IR basis and sparse sampling for fermionic propagators
    import sparse_ir, numpy as np
    basis = sparse_ir.FiniteTempBasis('F', beta=10, wmax=8, eps=1e-6)
    stau = sparse_ir.TauSampling(basis)
    siw = sparse_ir.MatsubaraSampling(basis, positive_only=True)

    # Solve the single impurity Anderson model coupled to a bath with a
    # semicircular states with unit half bandwidth.
    U = 1.2
    def rho0w(w):
        return np.sqrt(1-w.clip(-1,1)**2) * 2/np.pi

    # Compute the IR basis coefficients for the non-interacting propagator
    rho0l = basis.v.overlap(rho0w)
    G0l = -basis.s * rho0l

    # Self-consistency loop: alternate between second-order expression for the
    # self-energy and the Dyson equation until convergence.
    Gl = G0l
    Gl_prev = 0
    while np.linalg.norm(Gl - Gl_prev) > 1e-6:
        Gl_prev = Gl
        Gtau = stau.evaluate(Gl)
        Sigmatau = U**2 * Gtau**3
        Sigmal = stau.fit(Sigmatau)
        Sigmaiw = siw.evaluate(Sigmal)
        G0iw = siw.evaluate(G0l)
        Giw = 1/(1/G0iw - Sigmaiw)
        Gl = siw.fit(Giw)

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

License and citation
--------------------
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
