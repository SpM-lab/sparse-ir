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

    pip install sparse-ir

Install via `conda <https://anaconda.org/spm-lab/sparse-ir>`_::

    conda install -c spm-lab sparse-ir

sparse-ir requires `numpy <https://numpy.org/>`_, `scipy <https://scipy.org/>`_,
and `pylibsparseir <https://pypi.org/project/pylibsparseir>`_ (a thin Python wrapper
for the `libsparseir <https://github.com/SpM-lab/libsparseir>`_ C API).

To manually install the current development version, you can use the following::

   # Only recommended for developers - no automatic updates!
   git clone https://github.com/SpM-lab/sparse-ir
   cd sparse-ir
   uv sync

Note: `uv` is a fast Python package manager. If you don't have it installed,
you can install it with ``pip install uv`` or use ``pip install -e .`` instead.

Building documentation
----------------------
To build the documentation locally, first install the development dependencies::

   uv sync --group doc

Then build the documentation::

   uv run sphinx-build -M html doc _build/html

The documentation will be available in ``_build/html/html/index.html``.

Documentation and tutorial
--------------------------
Check out our `comprehensive tutorial`_, where we self-contained
notebooks for several many-body methods - GF(2), GW, Eliashberg equations,
Lichtenstein formula, FLEX, ... - are presented.

Refer to the `API documentation`_ for more details on how to work
with the python library.

There is also a `Julia library`_ and (currently somewhat restricted)
`C library with Fortran bindings`_ available for the IR basis and sparse sampling.

.. _comprehensive tutorial: https://spm-lab.github.io/sparse-ir-tutorial
.. _API documentation: https://sparse-ir.readthedocs.io
.. _Julia library: https://github.com/SpM-lab/SparseIR.jl
.. _C library with Fortran bindings: https://github.com/SpM-lab/libsparseir

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
 - Markus Wallerberger et al., `SoftwareX 21, 101266`_ (2023)

If you are discussing sparse sampling in your research specifically, please
also consider citing an independently discovered, closely related approach, the
MINIMAX isometry method (Merzuk Kaltak and Georg Kresse,
`Phys. Rev. B 101, 205145`_, 2020).

.. _Phys. Rev. B 96, 035147: https://doi.org/10.1103/PhysRevB.96.035147
.. _Phys. Rev. B 101, 035144: https://doi.org/10.1103/PhysRevB.101.035144
.. _SoftwareX 21, 101266: https://doi.org/10.1016/j.softx.2022.101266
.. _Phys. Rev. B 101, 205145: https://doi.org/10.1103/PhysRevB.101.205145

Development
-----------

Updating pylibsparseir Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When updating the ``pylibsparseir`` dependency version, you must update it in
**both** ``pyproject.toml`` and ``.conda/meta.yaml`` to maintain consistency:

1. **Update pyproject.toml**::

       # Edit dependencies in pyproject.toml
       dependencies = [
           "pylibsparseir>=0.8.0,<0.9.0",  # Update version range
       ]

2. **Update .conda/meta.yaml**::

       # Edit both host and run requirements in .conda/meta.yaml
       requirements:
         host:
           - spm-lab::pylibsparseir >=0.8.0,<0.9.0
         run:
           - spm-lab::pylibsparseir >=0.8.0,<0.9.0

3. **Verify consistency**::

       python check_libsparseir_version_consistency.py

   This should output ``✅ Version specifications are consistent!``

4. **Commit changes**::

       git add pyproject.toml .conda/meta.yaml
       git commit -m "chore: update pylibsparseir dependency to >=0.8.0,<0.9.0"

Version Consistency Check
~~~~~~~~~~~~~~~~~~~~~~~~~~
This repository includes a tool to ensure consistency between different package managers:

- **Version Consistency Check**: Ensures that ``pylibsparseir`` version
  specifications in ``pyproject.toml`` and ``.conda/meta.yaml`` are consistent.

  Run the check manually::

      python check_libsparseir_version_consistency.py

  Or install as a pre-commit hook::

      pip install pre-commit
      pre-commit install

Release Process
~~~~~~~~~~~~~~~
To release a new version (e.g., ``2.0.0a10``):

1. **Create a working branch for version bump**::

       git checkout mainline
       git pull origin mainline
       git checkout -b bump-to-2.0.0a10

2. **Update version in pyproject.toml**::

       # Edit pyproject.toml: version = "2.0.0a10"

3. **Commit and push**::

       git add pyproject.toml
       git commit -m "Bump to v2.0.0a10"
       git push --set-upstream origin bump-to-2.0.0a10

4. **Create Pull Request and merge to mainline**

5. **Create and push tag**::

       git checkout mainline
       git pull origin mainline
       git tag v2.0.0a10
       git push origin v2.0.0a10

6. **Automated builds** (triggered by tag push):

   - PyPI: ``wheel.yml`` workflow builds and uploads to PyPI
   - conda: ``conda.yml`` workflow builds and uploads to SpM-lab channel

Both workflows are automatically triggered when a tag starting with ``v`` is pushed.