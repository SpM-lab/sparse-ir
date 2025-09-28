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
from pylibsparseir.core import funcs_get_size, funcs_get_knots, SPIR_ORDER_COLUMN_MAJOR
from ._gauss import kronrod_31_15

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
            return o.ravel()
        elif x.size > 1 and self._size == 1:
            return o.ravel()
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
                return output.ravel()
            elif x.size > 1 and self._size == 1:
                return output.ravel()
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

    Parameters:
    -----------
    funcs : FunctionSet
        Function set to evaluate the polynomial
    xmin : float
        Minimum value of the interval
    xmax : float
        Maximum value of the interval
    period : float
        Period of the interval. For periodic functions, this should be the period of the function.
        For non-periodic functions, this should be 0.
    """

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float, period: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax
        self._period = period
        self.shape = (self._funcs.size(),)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def overlap(self, f, xmin: float, xmax: float, *, rtol=2.3e-16, return_error=False, points=None):
        """
        Evaluate overlap integral of this polynomial with function ``f``.
        If ``f` returns a scalar, the result is a scalar.
        If ``f`` returns an array, the result is an array with the same shape.
        """

        # Check if f returns a scalar or an array
        f_result = f(0.5*xmin + 0.5*xmax)
        if hasattr(f_result, 'shape'):
            # NumPy array or similar - check if it's scalar-like
            is_scalar = f_result.shape == ()
        else:
            # Python built-in float type only
            is_scalar = isinstance(f_result, float)
            
        if is_scalar:
            # For scalar functions, compute overlap directly
            int_result, int_error = _compute_overlap(self, f, xmin, xmax, rtol=rtol, points=points)
            if return_error:
                return int_result, int_error
            else:
                return int_result
        else:
            return self.overlap_vector(f, xmin, xmax, rtol=rtol, return_error=return_error, points=points)
            

class PiecewiseLegendrePolyVector:
    """Piecewise Legendre polynomial vector."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float, period: float):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax
        self._period = period
        self.shape = (self._funcs.size(),)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function or slice of functions."""
        if isinstance(index, slice):
            return PiecewiseLegendrePolyVector(self._funcs[index], self._xmin, self._xmax, self._period)
        else:
            return PiecewiseLegendrePoly(self._funcs[index], self._xmin, self._xmax, self._period)

    def overlap(self, f, xmin: float, xmax: float, *, rtol=2.3e-16, return_error=False, points=None):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.
            xmin : float
                Minimum value of the interval
            xmax : float
                Maximum value of the interval
            points (sequence of floats)
                A sequence of break points in the integration interval
                where local difficulties of the integrand may occur
                (e.g., singularities, discontinuities)

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).
        """
        if xmin > xmax:
            raise ValueError("xmin must be less than xmax")
        
        if self._period == 0.0:
            if xmin < self._xmin:
                raise ValueError(f"xmin ({xmin}) must be greater than or equal to the lower bound of the polynomial domain ({self._xmin})")
            if xmax > self._xmax:
                raise ValueError(f"xmax ({xmax}) must be less than or equal to the upper bound of the polynomial domain ({self._xmax})")

        int_result, int_error = _compute_overlap(self, f, xmin, xmax, rtol=rtol, points=points)
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
        """Get a single basis function or slice of functions."""
        if isinstance(index, slice):
            return PiecewiseLegendrePolyFTVector(self._funcs[index])
        else:
            return PiecewiseLegendrePolyFT(self._funcs[index])


def _generate_knots(poly, xmin: float, xmax: float, points=None):
    # Get knots from poly and add integration boundaries
    knots = funcs_get_knots(poly._funcs._ptr)
    knots = np.unique(np.hstack([knots, [xmin, xmax]]))
    
    if points is not None:
        points = np.asarray(points)
        knots = np.unique(np.hstack((knots, points)))
    
    if poly._period != 0.0:
        # Shift points to cover the entire domain
        period = poly._period
        extended_knots = list(knots)
        
        # Extend in positive direction
        i = 1
        while True:
            offset = i * period
            new_knots = knots + offset
            if np.any(new_knots > poly._xmax):
                break
            extended_knots.extend(new_knots)
            i += 1
        
        # Extend in negative direction
        i = 1
        while True:
            offset = -i * period
            new_knots = knots + offset
            if np.any(new_knots < poly._xmin):
                break
            extended_knots.extend(new_knots)
            i += 1
        
        knots = np.unique(np.array(extended_knots))
    
    # Trim knots to the integration interval
    knots = knots[(knots >= xmin) & (knots <= xmax)]
    knots = np.sort(knots)

    return knots


def _compute_overlap(poly, f, xmin: float, xmax: float,
        rtol=2.3e-16, radix=2, max_refine_levels=40,
        max_refine_points=2000, points=None):

    # Get knots from poly and add integration boundaries
    knots = _generate_knots(poly, xmin, xmax, points)
    
    # Use Gauss-Kronrod integration on segments
    base_rule = kronrod_31_15()
    xstart = knots[:-1]
    xstop = knots[1:]

    f_shape = None
    res_value = 0
    res_error = 0
    res_magn = 0
    max_refine_levels = 40
    max_refine_points = 2000
    radix = 2
    
    for _ in range(max_refine_levels):
        if xstart.size > max_refine_points:
            warn("Refinement is too broad, aborting (increase rtol)")
            break

        rule = base_rule.reseat(xstart[:, None], xstop[:, None])

        fx = np.array(list(map(f, rule.x.ravel())))
        if f_shape is None:
            f_shape = fx.shape[1:]
        elif fx.shape[1:] != f_shape:
            raise ValueError("inconsistent shapes")
        fx = fx.reshape(rule.x.shape + (-1,))

        valx = poly(rule.x).reshape(-1, *rule.x.shape, 1) * fx
        int21 = (valx[:, :, :, :] * rule.w[:, :, None]).sum(2)
        int10 = (valx[:, :, rule.vsel, :] * rule.v[:, :, None]).sum(2)
        intdiff = np.abs(int21 - int10)
        intmagn = np.abs(int10)

        magn = res_magn + intmagn.sum(1).max(1)
        relerror = intdiff.max(2) / magn[:, None]

        xconverged = (relerror <= rtol).all(0)
        res_value += int10[:, xconverged].sum(1)
        res_error += intdiff[:, xconverged].sum(1)
        res_magn += intmagn[:, xconverged].sum(1).max(1)
        if xconverged.all():
            break

        xrefine = ~xconverged
        xstart = xstart[xrefine]
        xstop = xstop[xrefine]
        xedge = np.linspace(xstart, xstop, radix + 1, axis=-1)
        xstart = xedge[:, :-1].ravel()
        xstop = xedge[:, 1:].ravel()
    else:
        warn("Integration did not converge after refinement")

    res_shape = poly.shape + f_shape
    return res_value.reshape(res_shape), res_error.reshape(res_shape)
