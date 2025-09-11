"""
Piecewise polynomial functionality for SparseIR.

This module provides piecewise Legendre polynomial representation and
their Fourier transforms, which serve as core mathematical infrastructure
for IR basis functions.
"""

from ctypes import c_int, POINTER
import numpy as np
import weakref
import threading

from pylibsparseir.core import _lib
from pylibsparseir.core import funcs_eval_single_float64, funcs_eval_single_complex128
from pylibsparseir.core import funcs_get_size, funcs_get_roots
import scipy.integrate as integrate

# Global registry to track pointer usage
_pointer_registry = weakref.WeakSet()
_registry_lock = threading.Lock()

def funcs_get_slice(funcs_ptr, indices):
    status = c_int()
    indices = np.asarray(indices, dtype=np.int32)
    funcs = _lib.spir_funcs_get_slice(funcs_ptr, len(indices), indices.ctypes.data_as(POINTER(c_int)), status)
    if status.value != 0:
        raise RuntimeError(f"Failed to get basis function {indices}: {status.value}")
    return FunctionSet(funcs)

def funcs_ft_get_slice(funcs_ptr, indices):
    status = c_int()
    indices = np.asarray(indices, dtype=np.int32)
    funcs = _lib.spir_funcs_get_slice(funcs_ptr, len(indices), indices.ctypes.data_as(POINTER(c_int)), status)
    if status.value != 0:
        raise RuntimeError(f"Failed to get basis function {indices}: {status.value}")
    return FunctionSetFT(funcs)

class FunctionSet:
    """Wrapper for basis function evaluation."""

    def __init__(self, funcs_ptr):
        self._ptr = funcs_ptr
        self._released = False
        # Register this object for safe cleanup
        with _registry_lock:
            _pointer_registry.add(self)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if self._released:
            raise RuntimeError("Function set has been released")
        if not isinstance(x, np.ndarray):
            o = funcs_eval_single_float64(self._ptr, x)
            if len(o) == 1:
                return o[0]
            else:
                return o
        else:
            o = np.stack([funcs_eval_single_float64(self._ptr, e) for e in x]).T
            if len(o) == 1:
                return o[0]
            else:
                return o

    def __getitem__(self, index):
        """Get a single basis function."""
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        return funcs_get_slice(self._ptr, [index % sz])

    def release(self):
        """Manually release the function set."""
        if not self._released and self._ptr:
            try:
                _lib.spir_funcs_release(self._ptr)
            except:
                pass
            self._released = True
            self._ptr = None

    def __del__(self):
        # Only release if we haven't been released yet
        if not self._released:
            self.release()

class FunctionSetFT:
    """Wrapper for basis function evaluation."""

    def __init__(self, funcs_ptr):
        self._ptr = funcs_ptr
        self._released = False
        # Register this object for safe cleanup
        with _registry_lock:
            _pointer_registry.add(self)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if self._released:
            raise RuntimeError("Function set has been released")
        if not isinstance(x, np.ndarray):
            o = funcs_eval_single_complex128(self._ptr, x)
            if len(o) == 1:
                return o[0]
            else:
                return o
        else:
            o = np.stack([funcs_eval_single_complex128(self._ptr, e) for e in x]).T
            if len(o) == 1:
                return o[0]
            else:
                return o

    def __getitem__(self, index):
        """Get a single basis function."""
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        return funcs_ft_get_slice(self._ptr, [index % sz])

    def release(self):
        """Manually release the function set."""
        if not self._released and self._ptr:
            try:
                _lib.spir_funcs_release(self._ptr)
            except:
                pass
            self._released = True
            self._ptr = None

    def __del__(self):
        # Only release if we haven't been released yet
        if not self._released:
            self.release()

class PiecewiseLegendrePoly:
    """Piecewise Legendre polynomial.

    Models a function on the interval ``[-1, 1]`` as a set of segments on the
    intervals ``S[i] = [a[i], a[i+1]]``, where on each interval the function
    is expanded in scaled Legendre polynomials.
    """

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)


class PiecewiseLegendrePolyVector:
    """Piecewise Legendre polynomial vector."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function."""
        return PiecewiseLegendrePoly(self._funcs[index], self._xmin, self._xmax)

    def overlap(self, f):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).

        WARNING: This is a safe fallback implementation that avoids memory issues
        but may not be as accurate as the full roots-based integration.
        """

        xmin = self._xmin
        xmax = self._xmax
        roots = funcs_get_roots(self._funcs._ptr).tolist()
        roots.sort()

        test_x = (xmin + xmax) / 2
        test_out = self._funcs(test_x)
        output = np.zeros(len(test_out))
        for i in range(len(test_out)):
            for j in range(len(roots) - 1):
                output[i] += integrate.quad(lambda x: self._funcs(x)[i] * f(x), roots[j], roots[j+1], epsabs=1e-10, epsrel=1e-10)[0]
            output[i] += integrate.quad(lambda x: self._funcs(x)[i] * f(x), roots[-1], xmax, epsabs=1e-10, epsrel=1e-10)[0]
            output[i] += integrate.quad(lambda x: self._funcs(x)[i] * f(x), xmin, roots[0], epsabs=1e-10, epsrel=1e-10)[0]
        return output


class PiecewiseLegendrePolyFT:
    """Fourier transform of a piecewise Legendre polynomial.

    For a given frequency index ``n``, the Fourier transform of the Legendre
    function is defined as::

            phat(n) == ∫ dx exp(1j * pi * n * x / (xmax - xmin)) p(x)

    The polynomial is continued either periodically (``freq='even'``), in which
    case ``n`` must be even, or antiperiodically (``freq='odd'``), in which case
    ``n`` must be odd.
    """

    def __init__(self, funcs: FunctionSetFT):
        assert isinstance(funcs, FunctionSetFT), "funcs must be a FunctionSetFT"
        self._funcs = funcs

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

class PiecewiseLegendrePolyFTVector:
    """Fourier transform of a piecewise Legendre polynomial vector."""

    def __init__(self, funcs: FunctionSetFT):
        assert isinstance(funcs, FunctionSetFT), "funcs must be a FunctionSetFT"
        self._funcs = funcs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function."""
        return PiecewiseLegendrePolyFT(self._funcs[index])
