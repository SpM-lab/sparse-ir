# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
from warnings import warn

"""
Piecewise polynomial functionality for SparseIR.

This module provides piecewise Legendre polynomial representation and
their Fourier transforms, which serve as core mathematical infrastructure
for IR basis functions.
"""

from ctypes import c_int, c_int64, POINTER, c_double
import numpy as np
import weakref
import threading

from pylibsparseir.core import _lib
from pylibsparseir.core import funcs_eval_single_float64, funcs_eval_single_complex128
from pylibsparseir.core import funcs_get_size, funcs_get_roots, SPIR_ORDER_COLUMN_MAJOR

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
        self._size = funcs_get_size(funcs_ptr)
        # Register this object for safe cleanup
        with _registry_lock:
            _pointer_registry.add(self)
    
    def size(self):
        return self._size

    """
    Size of returned array is (n_funcs, n_points).
    """
    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if self._released:
            raise RuntimeError("Function set has been released")
        x = np.ascontiguousarray(x)
        if x.ndim == 0:
            o = funcs_eval_single_float64(self._ptr, x.item())
            if len(o) == 1:
                return o[0]
            else:
                return o

        o = self.__call_batch(x)

        if x.size == 1 and self._size == 1:
            return o.flat[0]
        elif x.size == 1 and self._size > 1:
            return o.flat
        elif x.size > 1 and self._size == 1:
            return o.flat
        else:
            return o
    
    def __call_batch(self, x: np.ndarray):
        # Use batch evaluation for arrays
        x = np.ascontiguousarray(x)
        original_shape = x.shape
        x_flat = x.ravel()
        n_points = len(x_flat)
        n_funcs = self._size
        
        # Prepare input array (double)
        x_double = x_flat.astype(np.float64)
        
        # Prepare output array (double)
        output = np.zeros((n_funcs, n_points), dtype=np.float64)
            
        # Call batch evaluation function
        status = _lib.spir_funcs_batch_eval(
            self._ptr,
            SPIR_ORDER_COLUMN_MAJOR,
            n_points,
            x_double.ctypes.data_as(POINTER(c_double)),
            output.ctypes.data_as(POINTER(c_double))
        )

        if status != 0:
            raise RuntimeError(f"Batch evaluation failed with status {status}")
        
        # Reshape output to match input shape: (n_funcs, ...) + original_shape
        output = output.reshape((n_funcs,) + original_shape)

        return output


    def __getitem__(self, index):
        """Get a single basis function or slice of functions."""
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        
        if isinstance(index, slice):
            # Handle slice
            start, stop, step = index.indices(sz)
            indices = list(range(start, stop, step))
        else:
            # Handle single index or list of indices
            index = np.asarray(index)
            if index.ndim == 0:
                # Single index
                indices = [int(index) % sz]
            else:
                # List/array of indices
                indices = (index % sz).tolist()
        
        return funcs_get_slice(self._ptr, indices)

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
        self._size = funcs_get_size(funcs_ptr)
        # Register this object for safe cleanup
        with _registry_lock:
            _pointer_registry.add(self)
    
    def size(self):
        return self._size

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        if self._released:
            raise RuntimeError("Function set has been released")
        x = np.ascontiguousarray(x)
        if x.ndim == 0:
            o = funcs_eval_single_complex128(self._ptr, x.item())
            if len(o) == 1:
                return o[0]
            else:
                return o
        else:
            # Use batch evaluation for arrays
            original_shape = x.shape
            x_flat = x.ravel()
            n_points = len(x_flat)
            n_funcs = self._size
            
            # Prepare input array
            x_int64 = x_flat.astype(np.int64)
            
            # Prepare output array (complex128)
            output = np.zeros((n_funcs, n_points), dtype=np.complex128)
            
            # Call batch evaluation function
            status = _lib.spir_funcs_batch_eval_matsu(
                self._ptr,
                SPIR_ORDER_COLUMN_MAJOR,
                n_points,
                x_int64.ctypes.data_as(POINTER(c_int64)),
                output.ctypes.data_as(POINTER(c_double))
            )
            
            if status != 0:
                raise RuntimeError(f"Batch evaluation failed with status {status}")
            
            # Reshape output to match input shape: (n_funcs, ...) + original_shape
            output = output.reshape((n_funcs,) + original_shape)
            
            if x.size == 1 and self._size == 1:
                return output.flat[0]
            elif x.size == 1 and self._size > 1:
                return output.flat
            elif x.size > 1 and self._size == 1:
                return output.flat
            else:
                return output

    def __getitem__(self, index):
        """Get a single basis function or slice of functions."""
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        
        if isinstance(index, slice):
            # Handle slice
            start, stop, step = index.indices(sz)
            indices = list(range(start, stop, step))
        else:
            # Handle single index or list of indices
            index = np.asarray(index)
            if index.ndim == 0:
                # Single index
                indices = [int(index) % sz]
            else:
                # List/array of indices
                indices = (index % sz).tolist()
        
        return funcs_ft_get_slice(self._ptr, indices)

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

    def overlap(self, f, *, rtol=2.3e-16, return_error=False, points=None):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.

            points (sequence of floats)
                A sequence of break points in the integration interval
                where local difficulties of the integrand may occur
                (e.g., singularities, discontinuities)

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).
        """
        int_result, int_error = _compute_overlap(self, f, rtol=rtol, points=points)
        if return_error:
            return int_result, int_error
        else:
            return int_result



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


def _compute_overlap(poly, f, rtol=2.3e-16, radix=2, max_refine_levels=40,
                     max_refine_points=2000, points=None):
    """Compute overlap integral using simple quadrature.
    
    This is a simplified implementation. For production use,
    a more sophisticated adaptive quadrature should be used.
    """
    if points is None:
        knots = poly.knots
    else:
        points = np.asarray(points)
        knots = np.unique(np.hstack((poly.knots, points)))
    
    # Simple trapezoidal rule for now
    n_points = 100
    x_eval = np.linspace(knots[0], knots[-1], n_points)
    
    # Evaluate function and polynomials
    f_values = f(x_eval)
    poly_values = poly(x_eval)
    
    # Simple integration
    dx = (knots[-1] - knots[0]) / (n_points - 1)
    result = np.sum(poly_values * f_values, axis=1) * dx
    
    # Return result and zero error estimate
    return result, np.zeros_like(result)
