Kernels
=======
The IR basis is nothing but the `singular value expansion`_ of a suitable
integral kernel `K` mediating the change from real frequencies to imaginary
times:

    G(τ) = - ∫ dω  K(τ, ω) w(ω) ρ(ω),

where ρ(ω) = - (1/π) Im G(ω+iδ) and w(ω) is a weight function.
The integral ∫ is defined on the interval [-ωmax, ωmax], where
ωmax (=Λ/β) is a frequency cutoff.

Different kernels yield different IR basis functions.  The `sparse-ir` library
defines two kernels:

 - :class:`sparse_ir.LogisticKernel`: continuation of *fermionic/bosonic*
   spectral functions with w(ω)=1 for fermions
   and w(ω)=1/tanh(ω/ωmax) for bosons.
 - :class:`sparse_ir.RegularizedBoseKernel`: continuation of *bosonic* spectral functions
   with w(ω)=1/ω.

By default, :class:`sparse_ir.LogisticKernel` is used.
Kernels can be fed directly into :class:`sparse_ir.IRBasis` or
:class:`sparse_ir.FiniteTempBasis` to get the intermediate representation.

.. _singular value expansion: https://w.wiki/3poQ


Predefined kernels
------------------
.. autoclass:: sparse_ir.LaplaceKernel
    :members:
    :special-members: __call__

.. autoclass:: sparse_ir.LogisticKernel
    :members:
    :special-members: __call__

.. autoclass:: sparse_ir.RegularizedBoseKernel
    :members:
    :special-members: __call__


Custom kernels
--------------
Adding kernels to ``sparse_ir`` is simple - at the very basic level, the
library expects a kernel ``K`` to be able to provide two things:

 1. the values through ``K(x, y)``
 2. a set of SVE discretization hints through ``K.hints()``

Let us suppose you simply want to include a Gaussian default model on the real
axis instead of the default (flat) one.  We create a new kernel by inheriting
from :class:`sparse_ir.kernel.KernelBase` and then simply wrap around a
fermionic kernel, modifying the values as needed::

    import sparse_ir
    import sparse_ir.kernel

    class KernelFGauss(sparse_ir.kernel.KernelBase):
        def __init__(self, lambda_, std):
            self._inner = sparse_ir.LaplaceKernel(lambda_)
            self.lambda_ = lambda_
            self.std = std

        def __call__(self, x, y, x_plus=None, x_minus=None):
            # First, get value of kernel
            val = self._inner(x, y, x_plus, x_minus)
            # Then multiply with default model
            val *= np.exp(-.5 * (y / self.std)**2)
            return val

        def hints(self, eps):
            return self._inner.hints(eps)

You can feed this kernel now directly to :class:`sparse_ir.IRBasis`::

    K = GaussFKernel(10., 1.)
    basis = sparse_ir.IRBasis(K, 'F')
    print(basis.s)

This should get you started.  For a fully-fledged and robust implementation,
you should:

 1. Make sure that your kernel does not lose precision in computing ``K(x, y)``,
    as this directly affects the quality of the basis.  This is also where the
    arguments ``x_plus`` and ``x_minus`` may become useful.

 2. Optimize your discretization hints.  To do so, it is useful to choose the
    segments in ``x`` and ``y`` close to the asymptotic distribution of the roots
    of the respective singular functions.  The Gauss order can then be
    determined from a convergence analysis.

 3. Check whether your kernel is centrosymmetric, and if so, override the
    ``is_centrosymmetric`` property.  This yields a approximately four-times
    performance boost.  However, check that the symmetrized versions of the
    kernels do not lose precision.


Base classes
------------
.. autoclass:: sparse_ir.kernel.KernelBase
    :members:
    :special-members: __call__

.. autoclass:: sparse_ir.kernel.ReducedKernel
    :members:
    :special-members: __call__

.. autoclass:: sparse_ir.kernel.SVEHintsBase
    :members:
