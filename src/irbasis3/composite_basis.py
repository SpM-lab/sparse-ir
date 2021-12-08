import numpy as np

from .poly import PiecewiseLegendreFT, PiecewiseLegendrePoly

class _CompositePiecewiseLegendreyBase:
    """Union of several piecewise Legendre polynomials"""
    def __init__(self, polys):
        self._polys = polys
        self._sizes = np.array([p.size for p in self._polys])
        self._cumsum_sizes = np.cumsum(self._sizes)

    def __getitem__(self, l):
        """Return part of a set of piecewise polynomials"""
        if isinstance(l, int) or issubclass(type(l), np.integer):
            idx_p, l_ = self._internal_pos(l)
            return self._polys[idx_p][l_]
        else:
            raise ValueError(f"Unsupported type of l {type(l)}!")

    def _internal_pos(self, l):
        idx_p = np.searchsorted(self._cumsum_sizes, l, "right")
        l_ = l - self._cumsum_sizes[idx_p-1] if idx_p >= 1 else l
        return idx_p, l_

class CompositePiecewiseLegendrePoly(_CompositePiecewiseLegendreyBase):
    """Union of several piecewise Legendre polynomials"""
    def __init__(self, polys):
        """Initialize CompositePieceWiseLegendrePoly

        Arguments:
        ----------
         - polys: list of PiecewiseLegendrePoly instances with ndim=1
        """
        assert isinstance(polys, list)
        assert all([isinstance(x, PiecewiseLegendrePoly) for x in polys]) 
        assert all([x.ndim==1 for x in polys]) 
        super().__init__(polys)

    def __call__(self, x):
        """Evaluate polynomial at position x"""
        return np.vstack((p(x) for p in self._polys))
    
    def value(self, l, x):
        """Return value for l and x."""
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        return np.squeeze(np.asarray([self[ll](xx) for ll, xx in zip(l, x)]))

    def overlap(self, f, axis=None, _deg=None):
        r"""Evaluate overlap integral of this polynomial with function `f`"""
        return np.vstack((p.overlap(f, axis, _deg) for p in self._polys))

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        return np.vstack([p.deriv(n) for p in self._polys])

    def hat(self, freq, n_asymp=None):
        """Get Fourier transformed object"""
        return CompositePiecewiseLegendreFT([PiecewiseLegendreFT(p, freq, n_asymp) for p in self._polys])

    def roots(self, alpha=2):
        """Find all roots of the piecewise polynomial"""
        return np.unique(np.hstack([p.roots(alpha) for p in self._polys]))
    
    @property
    def shape(self):
        return (self.size,)

    @property
    def size(self): return np.sum((p.size for p in self._polys))

    @property
    def ndim(self): return 1

class CompositePiecewiseLegendreFT(_CompositePiecewiseLegendreyBase):
    """Union of fourier transform of several piecewise Legendre polynomials """
    def __init__(self, polys):
        """Initialize CompositePieceWiseLegendreFT

        Arguments:
        ----------
         - polys: list of PiecewiseLegendreFT
        """
        assert isinstance(polys, list)
        assert all([isinstance(x, PiecewiseLegendreFT) for x in polys]) 
        super().__init__(polys)

    def __call__(self, n):
        """Obtain Fourier transform of polynomial for given frequencies"""
        return np.vstack((p(n) for p in self._polys))