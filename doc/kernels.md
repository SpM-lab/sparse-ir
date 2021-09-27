Kernels
=======
Defines integral kernels for the analytic continuation problem.  Specifically,
it defines two kernels:

 - `irbasis3.KernelFFlat`: continuation of *fermionic* spectral functions.
 - `irbasis3.KernelFFlat`: continuation of *bosonic* spectral functions.

These can be fed directly into `irbasis3.IRBasis` or `irbasis3.FiniteTempBasis`
to get the intermediate representation.

Predefined kernels
------------------

```{eval-rst}
.. autoclass:: irbasis3.kernel.KernelFFlat
    :members:
    :special-members: __call__

.. autoclass:: irbasis3.kernel.KernelBFlat
    :members:
    :special-members: __call__
```

Custom kernels
--------------
Adding kernels to `irbasis3` is simple - at the very basic levle, the library
expects a kernel `K` to be able to provide two things:

 1. the values through `K(x, y)`
 2. a set of SVE discretization hints through `K.hints()`

Let us suppose you simply want to include a Gaussian default model on the real
axis instead of the default (flat) one.  We create a new kernel by inheriting
from `irbasis.kernel.KernelBase` and then simply wrap around a fermionic
kernel, modifying the values as needed:

    import irbasis3
    import irbasis3.kernel

    class KernelFGauss(irbasis3.kernel.KernelBase):
        def __init__(self, lambda_, std):
            self._inner = irbasis3.KernelFFlat(lambda_)
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

You can feed this kernel now directly to `irbasis.IRBasis`:

    K = GaussFKernel(10., 1.)
    basis = irbasis3.IRBasis(K, 'F')
    print(basis.s)

This should get you started.  For a fully-fledged and robust implementation,
you should:

 1. Make sure that your kernel does not lose precision in computing `K(x, y)`,
    as this directly affects the quality of the basis.  This is also where the
    arguments `x_plus` and `x_minus` may become useful.

 2. Optimize your discretization hints.  To do so, it is useful to choose the
    segments in `x` and `y` close to the asymptotic distribution of the roots
    of the respective singular functions.  The Gauss order can then be
    determined from a convergence analysis.

 3. Check whether your kernel is centrosymmetric, and if so, override the
    `is_centrosymmetric` property.  This yields a approximately four-times
    performance boost.  However, check that the symmetrized versions of the
    kernels do not lose precision.

Base classes
------------

```{eval-rst}
.. autoclass:: irbasis3.kernel.KernelBase
    :members:
    :special-members: __call__

.. autoclass:: irbasis3.kernel.ReducedKernel
    :members:
    :special-members: __call__

.. autoclass:: irbasis3.kernel.SVEHints
    :members:
```
