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

from pylibsparseir.core import _lib, c_double_complex, COMPUTATION_SUCCESS
from pylibsparseir.core import funcs_eval_single_float64, funcs_eval_single_complex128
from pylibsparseir.core import funcs_get_size, funcs_get_knots, SPIR_ORDER_COLUMN_MAJOR
from ._gauss import kronrod_31_15
from . import _util

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

def funcs_clone(funcs_ptr):
    """Clone a function set."""
    cloned = _lib.spir_funcs_clone(funcs_ptr)
    if not cloned:
        raise RuntimeError("Failed to clone function set")
    return FunctionSet(cloned)

def funcs_deriv(funcs_ptr, n):
    """Compute the n-th derivative of a function set."""
    status = c_int()
    deriv_funcs = _lib.spir_funcs_deriv(funcs_ptr, n, status)
    if status.value != 0:
        raise RuntimeError(f"Failed to compute derivative of order {n}: {status.value}")
    if not deriv_funcs:
        raise RuntimeError(f"Failed to compute derivative of order {n}")
    return FunctionSet(deriv_funcs)

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

    def __call__(self, x):
        """Evaluate basis functions at the given point(s).

        The returned shape is ``(n_funcs,) + np.shape(x)``, with singleton
        axes dropped only where they carry no information:

        - a set of ``n_funcs > 1`` functions at a scalar ``x`` gives shape
          ``(n_funcs,)``;
        - a single function (``n_funcs == 1``) gives exactly ``np.shape(x)``,
          i.e. a scalar for scalar ``x``;
        - otherwise the full ``(n_funcs,) + np.shape(x)`` is returned, so a
          multi-dimensional ``x`` keeps its shape.
        """
        if self._released:
            raise RuntimeError("Function set has been released")
        x = np.asarray(x)
        if x.ndim == 0:
            o = np.asarray(funcs_eval_single_float64(self._ptr, float(x)))
            return o[0] if self._size == 1 else o

        o = self.__call_batch(x)
        if self._size == 1:
            # Single function: drop the leading function axis, keep x's shape.
            return o.reshape(x.shape)
        return o

    def __call_batch(self, x: np.ndarray):
        original_shape = np.shape(x)
        n_funcs = self._size

        # Normalize with an explicit dtype and take the pointer from the
        # normalized object: a float32/int array reinterpreted through a
        # c_double pointer would read 8 bytes per 4-byte element.
        x_double = _util.as_boundary_real(np.ravel(x), "evaluation points")
        n_points = x_double.size

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

        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Batch evaluation failed with status {status}")

        # Reshape output to match input shape: (n_funcs, ...) + original_shape
        output = output.reshape((n_funcs,) + original_shape)

        return output


    def __getitem__(self, index):
        """Get a single basis function or slice of functions.

        Negative indices are resolved with Python semantics; an index outside
        ``[-size, size)`` raises :class:`IndexError` rather than being wrapped
        around with a modulo.
        """
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        return funcs_get_slice(self._ptr,
                               _util.resolve_function_indices(index, sz))

    def deriv(self, n=1):
        """Compute the n-th derivative of the basis functions.
        
        Args:
            n (int): Order of the derivative (default: 1)
            
        Returns:
            FunctionSet: New function set representing the n-th derivative
        """
        if self._released:
            raise RuntimeError("Function set has been released")
        if n < 0:
            raise ValueError("Derivative order must be non-negative")
        if n == 0:
            # Return a clone
            return funcs_clone(self._ptr)
            
        return funcs_deriv(self._ptr, n)

    def release(self):
        """Manually release the function set."""
        if not self._released and self._ptr:
            _lib.spir_funcs_release(self._ptr)
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
        """Evaluate the basis functions at reduced Matsubara frequencies.

        The returned shape follows the same convention as
        :py:meth:`FunctionSet.__call__`: ``(n_funcs,) + np.shape(x)``, with
        the leading axis dropped if the set holds a single function and the
        trailing axes dropped if ``x`` is a scalar.

        Raises:
            ValueError: if any element of ``x`` is not an integer.  Reduced
                Matsubara frequencies are integers and are never truncated.
        """
        if self._released:
            raise RuntimeError("Function set has been released")
        # Validate integrality *before* the int64 conversion: ``astype`` would
        # silently turn 1.9 into 1.
        x_checked = _util.check_reduced_matsubara(x)
        original_shape = x_checked.shape
        if x_checked.ndim == 0:
            o = np.asarray(
                funcs_eval_single_complex128(self._ptr, int(x_checked)))
            return o[0] if self._size == 1 else o

        x_int64 = np.ascontiguousarray(np.ravel(x_checked), dtype=np.int64)
        n_points = x_int64.size
        n_funcs = self._size
        output = np.zeros((n_funcs, n_points), dtype=np.complex128)

        status = _lib.spir_funcs_batch_eval_matsu(
            self._ptr,
            SPIR_ORDER_COLUMN_MAJOR,
            n_points,
            x_int64.ctypes.data_as(POINTER(c_int64)),
            output.ctypes.data_as(POINTER(c_double_complex))
        )
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Batch evaluation failed with status {status}")

        output = output.reshape((n_funcs,) + original_shape)
        if n_funcs == 1:
            return output.reshape(original_shape)
        return output

    def __getitem__(self, index):
        """Get a single basis function or slice of functions.

        Negative indices are resolved with Python semantics; an index outside
        ``[-size, size)`` raises :class:`IndexError` rather than being wrapped
        around with a modulo.
        """
        if self._released:
            raise RuntimeError("Function set has been released")
        sz = funcs_get_size(self._ptr)
        return funcs_ft_get_slice(self._ptr,
                                  _util.resolve_function_indices(index, sz))

    def release(self):
        """Manually release the function set."""
        if not self._released and self._ptr:
            _lib.spir_funcs_release(self._ptr)
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
        Period of the interval. For periodic functions, this should be the
        period of the function. For non-periodic functions, this should be 0.
    default_overlap_range : tuple, optional
        Default range for overlap calculations (xmin, xmax)
    """

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float,
                 period: float, default_overlap_range=None):
        if not isinstance(funcs, FunctionSet):
            raise ValueError("funcs must be a FunctionSet")
        if funcs.size() != 1:
            raise ValueError("funcs must have size 1")
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax
        self._period = period
        self.shape = (self._funcs.size(),)

        # Set default overlap range
        if default_overlap_range is not None:
            self._default_overlap_range = default_overlap_range
        else:
            # Default: use existing xmin, xmax
            self._default_overlap_range = (xmin, xmax)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def overlap(self, f, xmin: float = None, xmax: float = None, *, rtol=2.3e-16, return_error=False, points=None):
        """
        Evaluate overlap integral of this polynomial with function ``f``.
        If ``f` returns a scalar, the result is a scalar.
        If ``f`` returns an array, the result is an array with the same shape.

        Parameters:
        -----------
        f : callable
            Function to integrate with
        xmin : float, optional
            Minimum value of the interval. If None, uses default range.
        xmax : float, optional
            Maximum value of the interval. If None, uses default range.
        rtol : float
            Relative tolerance for integration
        return_error : bool
            Whether to return integration error
        points : sequence, optional
            Break points for integration
        """
        # Use default range if not specified
        if xmin is None:
            xmin = self._default_overlap_range[0]
        if xmax is None:
            xmax = self._default_overlap_range[1]

        polyvec = PiecewiseLegendrePolyVector(self._funcs, self._xmin,
                                            self._xmax, self._period)

        int_result, int_error = polyvec.overlap(f, xmin, xmax, rtol=rtol,
                                              return_error=True, points=points)

        int_result = int_result.reshape(int_result.shape[1:])
        int_error = int_error.reshape(int_error.shape[1:])

        if int_result.shape == ():
            int_result = int_result.item()
            int_error = int_error.item()

        if return_error:
            return int_result, int_error
        else:
            return int_result


class PiecewiseLegendrePolyVector:
    """Piecewise Legendre polynomial vector."""

    def __init__(self, funcs: FunctionSet, xmin: float, xmax: float,
                 period: float, default_overlap_range=None):
        self._funcs = funcs
        self._xmin = xmin
        self._xmax = xmax
        self._period = period
        self.shape = (self._funcs.size(),)

        # Set default overlap range
        if default_overlap_range is not None:
            self._default_overlap_range = default_overlap_range
        else:
            # Default: use existing xmin, xmax
            self._default_overlap_range = (xmin, xmax)

    def __call__(self, x):
        """Evaluate basis functions at given points."""
        return self._funcs(x)

    def __getitem__(self, index):
        """Get a single basis function or slice of functions."""
        funcs_slice = self._funcs[index]
        if funcs_slice.size() == 1:
            return PiecewiseLegendrePoly(funcs_slice, self._xmin, self._xmax,
                                       self._period, self._default_overlap_range)
        else:
            return PiecewiseLegendrePolyVector(funcs_slice, self._xmin,
                                             self._xmax, self._period,
                                             self._default_overlap_range)

    def deriv(self, n=1):
        """Compute the n-th derivative of the basis functions.
        
        Args:
            n (int): Order of the derivative (default: 1)
            
        Returns:
            PiecewiseLegendrePolyVector: New polynomial vector representing the n-th derivative
        """
        deriv_funcs = self._funcs.deriv(n)
        return PiecewiseLegendrePolyVector(deriv_funcs, self._xmin, self._xmax,
                                          self._period, self._default_overlap_range)


    def overlap(self, f, xmin: float = None, xmax: float = None, *, rtol=2.3e-16, return_error=False, points=None):
        r"""Evaluate overlap integral of this polynomial with function ``f``.

        Given the function ``f``, evaluate the integral::

            ∫ dx * f(x) * self(x)

        using piecewise Gauss-Legendre quadrature, where ``self`` are the
        polynomials.

        Arguments:
            f (callable):
                function that is called with a point ``x`` and returns ``f(x)``
                at that position.
            xmin : float, optional
                Minimum value of the interval. If None, uses default range.
            xmax : float, optional
                Maximum value of the interval. If None, uses default range.
            points (sequence of floats)
                A sequence of break points in the integration interval
                where local difficulties of the integrand may occur
                (e.g., singularities, discontinuities)

        Return:
            array-like object with shape (poly_dims, f_dims)
            poly_dims are the shape of the polynomial and f_dims are those
            of the function f(x).
        """
        # Use default range if not specified
        if xmin is None:
            xmin = self._default_overlap_range[0]
        if xmax is None:
            xmax = self._default_overlap_range[1]

        if xmin > xmax:
            raise ValueError("xmin must be less than xmax")

        if self._period == 0.0:
            if xmin < self._xmin:
                raise ValueError(f"xmin ({xmin}) must be greater than or equal "
                               f"to the lower bound of the polynomial domain "
                               f"({self._xmin})")
            if xmax > self._xmax:
                raise ValueError(f"xmax ({xmax}) must be less than or equal "
                               f"to the upper bound of the polynomial domain "
                               f"({self._xmax})")

        f_res = f(0.5*xmin + 0.5*xmax)

        f_ = f
        if hasattr(f_res, 'shape'):
            if f_res.dtype != np.float64:
                raise ValueError("f must return a float64 array")
            f_shape = f_res.shape
            f_length = f_res.size
            f_ = lambda x: f(x).ravel()
        elif isinstance(f_res, float) or isinstance(f_res, np.float64):
            if f_res.dtype != np.float64:
                raise ValueError("f must return a float64 scalar")
            f_shape = ()
            f_length = 1
            f_ = lambda x: np.array([f(x)])
        else:
            raise ValueError("f must return a scalar of float64 or an array")

        knots = funcs_get_knots(self._funcs._ptr)
        knots = _cover_domain(knots, self._period, xmin, xmax, self._xmin, self._xmax, points)

        int_result, int_error = _compute_overlap_internal(self, self._funcs.size(), f_, f_length, xmin, xmax, knots, rtol=rtol)

        int_result = int_result.reshape(self.shape + f_shape)
        int_error = int_error.reshape(self.shape + f_shape)

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


def _cover_domain(
    knots, period, xmin: float, xmax: float,
    poly_xmin: float, poly_xmax: float, points=None):

    # Add integration boundaries
    knots = np.unique(np.hstack([knots, [xmin, xmax]]))

    if points is not None:
        points = np.asarray(points)
        knots = np.unique(np.hstack((knots, points)))

    if period != 0.0:
        # Shift points to cover the entire domain
        extended_knots = list(knots)

        # Extend in positive direction
        i = 1
        while True:
            offset = i * period
            new_knots = knots + offset
            if np.any(new_knots > poly_xmax):
                break
            extended_knots.extend(new_knots)
            i += 1

        # Extend in negative direction
        i = 1
        while True:
            offset = -i * period
            new_knots = knots + offset
            if np.any(new_knots < poly_xmin):
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
    knots = funcs_get_knots(poly._funcs._ptr)
    knots = _cover_domain(knots, poly._period, xmin, xmax, poly._xmin, poly._xmax, points)

    f_res = f(0.5*xmin + 0.5*xmax)
    f_ = f
    if hasattr(f_res, 'shape'):
        if f_res.dtype != np.float64:
            raise ValueError("f must return a float64 array")
        f_shape = f_res.shape
        f_length = f_res.size
        f_ = lambda x: f(x).ravel()
    elif isinstance(f_res, float) or isinstance(f_res, np.float64):
        if f_res.dtype != np.float64:
            raise ValueError("f must return a float64 scalar")
        f_shape = ()
        f_length = 1
        f_ = lambda x: np.array([f(x)])
    else:
        raise ValueError("f must return a scalar of float64 or an array")

    result = _compute_overlap_internal(
        poly, f_, f_length, xmin, xmax, knots, rtol, radix, max_refine_levels, max_refine_points)

    return result[0].reshape(poly.shape + f_shape), result[1].reshape(poly.shape + f_shape)


def _compute_overlap_internal(poly, poly_size, f, f_length: int, xmin: float, xmax: float, knots,
        rtol=2.3e-16, radix=2, max_refine_levels=40,
        max_refine_points=2000):

    # Use Gauss-Kronrod integration on segments
    base_rule = kronrod_31_15()
    xstart = knots[:-1]
    xstop = knots[1:]

    res_value = 0
    res_error = 0
    res_magn = 0
    max_refine_levels = 40
    max_refine_points = 2000
    radix = 2

    f_shape = (f_length,)

    for _ in range(max_refine_levels):
        if xstart.size > max_refine_points:
            warn("Refinement is too broad, aborting (increase rtol)")
            break

        rule = base_rule.reseat(xstart[:, None], xstop[:, None])

        fx = np.array(list(map(f, rule.x.ravel())))
        if fx.ndim != 2:
            raise ValueError("f must return a 1D array")
        if fx.shape[1] != f_length:
            raise ValueError("f must return a array with length f_length")
        fx = fx.reshape(rule.x.shape + (f_length,))

        rule_x_flat = rule.x.ravel()
        poly_val = np.array(list(map(poly, rule_x_flat))).T.reshape(-1, *rule.x.shape, 1)
        #poly_val = poly(rule.x).reshape(-1, *rule.x.shape, 1)
        #if poly_val.shape[0] >= 2:
            #print(poly_val.shape)
            ##print(poly_val[0, 0, 0, 0])
            #print(poly_val[1, 0, 0, 0])
        valx = poly_val * fx
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

    res_shape = (poly_size,) + f_shape
    return res_value.reshape(res_shape), res_error.reshape(res_shape)
