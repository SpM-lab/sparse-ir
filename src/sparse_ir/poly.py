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

    def overlap(self, f, n_points=100):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.
            n_points (int):
                Number of quadrature points per integration segment.

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).
        """
        from scipy.integrate import fixed_quad

        xmin = self._xmin
        xmax = self._xmax
        roots = funcs_get_roots(self._funcs._ptr).tolist()
        roots.sort()

        # Create integration segments
        segments = [xmin] + roots + [xmax]
        segments = sorted(list(set(segments)))  # Remove duplicates and sort

        # Collect all quadrature points and weights
        all_x = []
        all_weights = []
        
        for j in range(len(segments) - 1):
            a, b = segments[j], segments[j+1]
            if abs(b - a) < 1e-14:  # Skip zero-length segments
                continue
            
            # Get Gauss-Legendre quadrature points and weights
            from scipy.special import roots_legendre
            x_quad, w_quad = roots_legendre(n_points)
            # Scale to actual interval
            x_scaled = (b - a) / 2 * x_quad + (a + b) / 2
            w_scaled = w_quad * (b - a) / 2
            
            all_x.extend(x_scaled)
            all_weights.extend(w_scaled)

        # Convert to numpy arrays for batch processing
        all_x = np.array(all_x)
        all_weights = np.array(all_weights)

        # Evaluate function and polynomials at all points
        f_values = f(all_x)  # This should work with array input
        poly_values = self._funcs(all_x)  # Shape: (n_polys, n_points)

        # Compute overlap integrals
        output = np.sum(poly_values * f_values * all_weights, axis=1)

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
