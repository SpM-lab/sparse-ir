import numpy as np

class _CompositeBasisFunctionBase:
    """Union of several basis functions"""
    def __init__(self, polys):
        self._polys = tuple(polys)
        self._sizes = np.array([p.size for p in self._polys])
        self._cumsum_sizes = np.cumsum(self._sizes)
        self._size = np.sum(self._sizes)

    def __getitem__(self, l):
        """Return part of a set of basis functions"""
        if isinstance(l, int) or issubclass(type(l), np.integer):
            idx_p, l_ = self._internal_pos(l)
            return self._polys[idx_p][l_]
        else:
            raise ValueError(f"Unsupported type of l {type(l)}!")

    def _internal_pos(self, l):
        idx_p = np.searchsorted(self._cumsum_sizes, l, "right")
        l_ = l - self._cumsum_sizes[idx_p-1] if idx_p >= 1 else l
        return idx_p, l_

    @property
    def shape(self): return (self.size,)

    @property
    def size(self): return self._size


class CompositeBasisFunction(_CompositeBasisFunctionBase):
    """Union of several basis functions for the imaginary-time/real-frequency domains"""
    def __init__(self, polys):
        """Initialize CompositeBasisFunction

        Arguments:
        ----------
         - polys: iterable object of basis-function-like instances with ndim=1
        """
        super().__init__(polys)

    def __call__(self, x):
        """Evaluate basis function at position x"""
        return np.vstack((p(x) for p in self._polys))
    
    def value(self, l, x):
        """Return value for l and x."""
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return np.squeeze(np.asarray([self[ll](xx) for ll, xx in zip(l, x)]))

    def overlap(self, f, axis=None, _deg=None):
        r"""Evaluate overlap integral of this basis function with function `f`"""
        return np.vstack((p.overlap(f, axis, _deg) for p in self._polys))

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        return np.vstack([p.deriv(n) for p in self._polys])

    def hat(self, freq, n_asymp=None):
        """Get Fourier transformed object"""
        return CompositeBasisFunctionFT([p.hat(freq, n_asymp) for p in self._polys])

    def roots(self, alpha=2):
        """Find all roots of the basis function """
        return np.unique(np.hstack([p.roots(alpha) for p in self._polys]))
    
    @property
    def ndim(self): return 1

class CompositeBasisFunctionFT(_CompositeBasisFunctionBase):
    """Union of fourier transform of several basis functions for the Matsubara domain"""
    def __init__(self, polys):
        """Initialize CompositeBasisFunctionFT

        Arguments:
        ----------
         - polys: iterable object of basis-function-like instances
        """
        super().__init__(polys)

    def __call__(self, n):
        """Obtain Fourier transform of basis function for given frequencies"""
        return np.vstack((p(n) for p in self._polys))

class CompositeBasis:
    """Union of several basis sets"""
    def __init__(self, bases):
        """Initialize a composite basis

        Args:
            bases): iterable object of FiniteTempBasis instances
        """
        assert np.unique([b.statistics for b in bases]).size == 1, "All bases must have the same statistics!"
        self._size = np.sum((b.size for b in bases))
        self.u = CompositeBasisFunction([b.u for b in bases]) if all(b.u is not None for b in bases) else None
        self.v = CompositeBasisFunction([b.v for b in bases]) if all(b.v is not None for b in bases) else None
        self.uhat = CompositeBasisFunctionFT([b.uhat for b in bases]) if self.u is not None else None

        self.default_tau_sampling_points = np.unique(np.hstack((b.default_tau_sampling_points for b in bases)))
        self.default_matsubara_sampling_points = np.unique(np.hstack((b.default_matsubara_sampling_points for b in bases)))

    @property
    def size(self): return self._size

    @property
    def shape(self): return (self._size,)