Kernels
=======
The IR basis is nothing but the `singular value expansion`_ of a suitable
integral kernel `K` mediating the change from real frequencies to imaginary
times:

.. math::       G(\tau) = - \int_{-\omega_\mathrm{max}}^{\omega_\mathrm{max}}
                d\omega\, K(\tau, \omega) w(\omega) A(\omega),

where :math:`G(\tau) = -\langle T_\tau c(\tau) c^\dagger(0) \rangle` for
:math:`0 < \tau < \beta`,

.. math::       A(\omega) = -\frac{1}{\pi} \mathrm{Im}\, G^\mathrm{R}(\omega)

is the spectral function (for a diagonal component) and w(ω) is a weight
function; the kernel acts on :math:`\rho(\omega) = w(\omega) A(\omega)`.  The
integral is defined on the interval [-ωmax, ωmax], where ωmax (``wmax``) is a
frequency cutoff; Λ = βωmax (``lambda_``) is the cutoff of the kernel.

Different kernels yield different IR basis functions.  The `sparse-ir` library
defines two kernels:

 - :class:`sparse_ir.LogisticKernel`:
   :math:`K(\tau, \omega) = e^{-\tau\omega}/(1 + e^{-\beta\omega})`,
   continuation of *fermionic/bosonic*
   spectral functions with w(ω)=1 for fermions
   and w(ω)=1/tanh(βω/2) for bosons.
 - :class:`sparse_ir.RegularizedBoseKernel` (deprecated; use
   :class:`sparse_ir.LogisticKernel`):
   :math:`K(\tau, \omega) = \omega e^{-\tau\omega}/(1 - e^{-\beta\omega})`,
   continuation of *bosonic* spectral functions with w(ω)=1/ω.  libsparseir
   releases without the fix of SpM-lab/sparse-ir-rs#273 scale the singular
   values of such a basis by :math:`\omega_\mathrm{max}^{-1}` instead of
   :math:`\omega_\mathrm{max}^{+1}`.

By default, :class:`sparse_ir.LogisticKernel` is used.
A kernel can be passed to :class:`sparse_ir.FiniteTempBasis` through its
``kernel`` argument; its ``lambda_`` must equal ``beta * wmax``::

    import sparse_ir
    K = sparse_ir.LogisticKernel(10 * 8.0)
    basis = sparse_ir.FiniteTempBasis('F', beta=10, wmax=8.0, eps=1e-6, kernel=K)

Only these two kernels are supported: :class:`sparse_ir.SVEResult` and
:class:`sparse_ir.FiniteTempBasis` reject any other kernel with a
:class:`TypeError`.  The kernel objects are handles to the C library and are
not callable from Python.  Their formulas below are given in physical
units and in the dimensionless variables x = 2τ/β - 1 and y = ω/ωmax, both
in [-1, 1].

The notation follows the `notation page`_.

.. _singular value expansion: https://w.wiki/3poQ
.. _notation page: https://spm-lab.github.io/sparse-ir-doc/src/notation.html


Predefined kernels
------------------
.. autoclass:: sparse_ir.LogisticKernel
    :members:

.. autoclass:: sparse_ir.RegularizedBoseKernel
    :members:


Base classes
------------
.. autoclass:: sparse_ir.kernel.AbstractKernel
    :members:
