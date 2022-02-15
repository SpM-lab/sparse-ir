# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
from warnings import warn


class KernelBase:
    """Integral kernel ``K(x, y)``.

    Abstract base class for an integral kernel, i.e., a real binary function
    ``K(x, y)`` used in a Fredhold integral equation of the first kind:

                        u(x) = ∫ K(x, y) v(y) dy

    where ``x ∈ [xmin, xmax]`` and ``y ∈ [ymin, ymax]``.  For its SVE to exist,
    the kernel must be square-integrable, for its singular values to decay
    exponentially, it must be smooth.

    In general, the kernel is applied to a scaled spectral function rho'(y) as:

                        ∫ K(x, y) rho'(y) dy,

    where rho'(y) = w(y) rho(y).
    """
    def __call__(self, x, y, x_plus=None, x_minus=None):
        """Evaluate kernel at point (x, y)

        For given ``x, y``, return the value of ``K(x, y)``. The arguments may
        be numpy arrays, in which case the function shall be evaluated over
        the broadcast arrays.

        The parameters ``x_plus`` and ``x_minus``, if given, shall contain the
        values of ``x - xmin`` and ``xmax - x``, respectively.  This is useful
        if either difference is to be formed and cancellation expected.
        """
        raise NotImplementedError()

    def sve_hints(self, eps):
        """Provide discretisation hints for the SVE routines.

        Advises the SVE routines of discretisation parameters suitable in
        tranforming the (infinite) SVE into an (finite) SVD problem.

        See: :class:``SVEHintsBase``.
        """
        raise NotImplementedError()

    @property
    def xrange(self):
        """Tuple ``(xmin, xmax)`` delimiting the range of allowed x values"""
        return -1, 1

    @property
    def yrange(self):
        """Tuple ``(ymin, ymax)`` delimiting the range of allowed y values"""
        return -1, 1

    @property
    def is_centrosymmetric(self):
        """Kernel is centrosymmetric.

        Returns true if and only if ``K(x, y) == K(-x, -y)`` for all values of
        ``x`` and ``y``.  This allows the kernel to be block-diagonalized,
        speeding up the singular value expansion by a factor of 4.  Defaults
        to false.
        """
        return False

    def get_symmetrized(self, sign):
        """Return symmetrized kernel ``K(x, y) + sign * K(x, -y)``.

        By default, this returns a simple wrapper over the current instance
        which naively performs the sum.  You may want to override if this
        to avoid cancellation.
        """
        return ReducedKernel(self, sign)

    @property
    def ypower(self):
        """Power with which the y coordinate scales."""
        return 0

    @property
    def conv_radius(self):
        """Convergence radius of the Matsubara basis asymptotic model.

        For improved relative numerical accuracy, the IR basis functions on the
        Matsubara axis ``basis.uhat(n)`` can be evaluated from an asymptotic
        expression for ``abs(n) > conv_radius``.  If ``conv_radius`` is
        ``None``, then the asymptotics are unused (the default).
        """
        return None

    def weight_func(self, statistics: str):
        """Return the weight function for given statistics"""
        if statistics not in 'FB':
            raise ValueError("statistics must be 'F' for fermions or 'B' for bosons")
        return lambda x: np.ones_like(x)


class SVEHintsBase:
    """Discretization hints for singular value expansion of a given kernel."""
    @property
    def segments_x(self):
        """Segments for piecewise polynomials on the ``x`` axis.

        List of segments on the ``x`` axis for the associated piecewise
        polynomial.  Should reflect the approximate position of roots of a
        high-order singular function in ``x``.
        """
        raise NotImplementedError()

    @property
    def segments_y(self):
        """Segments for piecewise polynomials on the ``x`` axis.

        List of segments on the ``y`` axis for the associated piecewise
        polynomial.  Should reflect the approximate position of roots of a
        high-order singular function in ``y``.
        """
        raise NotImplementedError()

    @property
    def ngauss(self):
        """Gauss-Legendre order to use to guarantee accuracy"""
        raise NotImplementedError()

    @property
    def nsvals(self):
        """Upper bound for number of singular values

        Upper bound on the number of singular values above the given
        threshold, i.e., where ``s[l] >= eps * s[0]```
        """
        raise NotImplementedError()


class KernelFFlat(KernelBase):
    """Fermionic analytical continuation kernel.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``, the fermionic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``::

        K(x, y) == exp(-Λ * y * (x + 1)/2) / (1 + exp(-Λ*y))
    """
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, x, y, x_plus=None, x_minus=None):
        x, y = _check_domain(self, x, y)
        u_plus, u_minus, v = _compute_uv(self.lambda_, x, y, x_plus, x_minus)
        return self._compute(u_plus, u_minus, v)

    def sve_hints(self, eps):
        return _SVEHintsFFlat(self, eps)

    def _compute(self, u_plus, u_minus, v):
        # By introducing u_\pm = (1 \pm x)/2 and v = lambda * y, we can write
        # the kernel in the following two ways:
        #
        #    k = exp(-u_+ * v) / (exp(-v) + 1)
        #      = exp(-u_- * -v) / (exp(v) + 1)
        #
        # We need to use the upper equation for v >= 0 and the lower one for
        # v < 0 to avoid overflowing both numerator and denominator
        abs_v = np.abs(v)
        enum = np.exp(-abs_v * np.where(v > 0, u_plus, u_minus))
        denom = 1 + np.exp(-abs_v)
        return enum / denom

    @property
    def is_centrosymmetric(self):
        return True

    def get_symmetrized(self, sign):
        if sign == -1:
            return _KernelFFlatOdd(self, sign)
        return super().get_symmetrized(sign)

    @property
    def conv_radius(self): return 40 * self.lambda_

    def weight_func(self, statistics: str):
        """
        Return the weight function for given statistics.

        This kernel `KFFlat` can be used to represent τ dependence of
        a bosonic correlation function as follows:

            ∫ KBFlat(x, y) ρ(y) dy
                == ∫ exp(-Λ * y * (x + 1)/2) / (1 - exp(-Λ*y)) ρ(y) dy
                == ∫ KFFlat ρ'(y) dy,

        where:

            ρ'(y) == (1/tanh(Λ*y/2)) * ρ(y).
        """
        if statistics not in "FB":
            raise ValueError("invalid value of statistics argument")
        if statistics == "F":
            return lambda y: np.ones_like(y)
        else:
            return lambda y: 1/np.tanh(0.5*y)


class _SVEHintsFFlat(SVEHintsBase):
    def __init__(self, kernel, eps):
        self.kernel = kernel
        self.eps = eps

    @property
    def ngauss(self): return 10 if self.eps >= 1e-8 else 16

    @property
    def segments_x(self):
        nzeros = max(int(np.round(15 * np.log10(self.kernel.lambda_))), 1)
        diffs = 1./np.cosh(.143 * np.arange(nzeros))
        zeros_pos = diffs.cumsum()
        zeros_pos /= zeros_pos[-1]
        return np.concatenate((-zeros_pos[::-1], [0], zeros_pos))

    @property
    def segments_y(self):
        # Zeros around -1 and 1 are distributed asymptotically identical
        leading_diffs = np.array([
            0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030,
            0.06791, 0.06391, 0.05896, 0.05358, 0.04814, 0.04288, 0.03795,
            0.03342, 0.02932, 0.02565, 0.02239, 0.01951, 0.01699])

        nzeros = max(int(np.round(20 * np.log10(self.kernel.lambda_))), 2)
        if nzeros < 20:
            leading_diffs = leading_diffs[:nzeros]
        diffs = .25 / np.exp(.141 * np.arange(nzeros))
        diffs[:leading_diffs.size] = leading_diffs
        zeros = diffs.cumsum()
        zeros = zeros[:-1] / zeros[-1]
        zeros -= 1
        return np.concatenate(([-1], zeros, [0], -zeros[::-1], [1]))

    @property
    def nsvals(self):
        log10_lambda = max(1, np.log10(self.kernel.lambda_))
        return int(np.round((25 + log10_lambda) * log10_lambda))


class KernelBFlat(KernelBase):
    """Bosonic analytical continuation kernel.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``, the fermionic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``::

        K(x, y) == y exp(-Λ * y * (x + 1)/2) / (exp(-Λ * y) - 1)

    Care has to be taken in evaluating this expression around ``y == 0``.
    """
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, x, y, x_plus=None, x_minus=None):
        x, y = _check_domain(self, x, y)
        u_plus, u_minus, v = _compute_uv(self.lambda_, x, y, x_plus, x_minus)
        return self._compute(u_plus, u_minus, v)

    def _compute(self, u_plus, u_minus, v):
        # With "reduced variables" u, v we have:
        #
        #   K = -1/lambda * exp(-u_+ * v) * v / (exp(-v) - 1)
        #     = -1/lambda * exp(-u_- * -v) * (-v) / (exp(v) - 1)
        #
        # where we again need to use the upper equation for v >= 0 and the
        # lower one for v < 0 to avoid overflow.
        abs_v = np.abs(v)
        enum = np.exp(-abs_v * np.where(v >= 0, u_plus, u_minus))
        dtype = v.dtype

        # The expression ``v / (exp(v) - 1)`` is tricky to evaluate: firstly,
        # it has a singularity at v=0, which can be cured by treating that case
        # separately.  Secondly, the denominator loses precision around 0 since
        # exp(v) = 1 + v + ..., which can be avoided using expm1(...)
        not_tiny = abs_v >= 1e-200
        denom = np.ones_like(abs_v)
        np.divide(abs_v, np.expm1(-abs_v, where=not_tiny),
                  out=denom, where=not_tiny)
        return -1/dtype.type(self.lambda_) * enum * denom

    def sve_hints(self, eps):
        return _SVEHintsBFlat(self, eps)

    @property
    def is_centrosymmetric(self):
        return True

    def get_symmetrized(self, sign):
        if sign == -1:
            return _KernelBFlatOdd(self, sign)
        return super().get_symmetrized(sign)

    @property
    def ypower(self): return 1

    @property
    def conv_radius(self): return 40 * self.lambda_

    def weight_func(self, statistics: str):
        """ Return the weight function for given statistics """
        if statistics != "B":
            raise ValueError("Kernel is designed for bosonic functions")
        return lambda y: 1/y


class _SVEHintsBFlat(SVEHintsBase):
    def __init__(self, kernel, eps):
        self.kernel = kernel
        self.eps = eps

    @property
    def ngauss(self): return 10 if self.eps >= 1e-8 else 16

    @property
    def segments_x(self):
        # Somewhat less accurate ...
        nzeros = 15 * np.log10(self.kernel.lambda_)
        diffs = 1./np.cosh(.18 * np.arange(nzeros))
        zeros_pos = diffs.cumsum()
        zeros_pos /= zeros_pos[-1]
        return np.concatenate((-zeros_pos[::-1], [0], zeros_pos))

    @property
    def segments_y(self):
        # Zeros around -1 and 1 are distributed asymptotically identical
        leading_diffs = np.array([
            0.01363, 0.02984, 0.04408, 0.05514, 0.06268, 0.06679, 0.06793,
            0.06669, 0.06373, 0.05963, 0.05488, 0.04987, 0.04487, 0.04005,
            0.03553, 0.03137, 0.02758, 0.02418, 0.02115, 0.01846])

        nzeros = max(int(np.round(20 * np.log10(self.kernel.lambda_))), 20)
        i = np.arange(nzeros)
        diffs = .12/np.exp(.0337 * i * np.log(i+1))
        #diffs[:leading_diffs.size] = leading_diffs
        zeros = diffs.cumsum()
        zeros = zeros[:-1] / zeros[-1]
        zeros -= 1
        return np.concatenate(([-1], zeros, [0], -zeros[::-1], [1]))

    @property
    def nsvals(self):
        log10_lambda = max(1, np.log10(self.kernel.lambda_))
        return int(28 * log10_lambda)


class ReducedKernel(KernelBase):
    """Restriction of centrosymmetric kernel to positive interval.

    For a kernel ``K`` on ``[-1, 1] x [-1, 1]`` that is centrosymmetrix, i.e.,
    ``K(x, y) == K(-x, -y)``, it is straight-forward to show that the left/right
    singular vectors can be chosen as either odd or even functions.

    Consequentially, they are singular functions of a reduced kernel ``K_red``
    on ``[0, 1] x [0, 1]`` that is given as either::

        K_red(x, y) == K(x, y) + sign * K(x, -y)

    This kernel is what this class represents.  The full singular functions can
    be reconstructed by (anti-)symmetrically continuing them to the negative
    axis.
    """
    def __init__(self, inner, sign=1):
        if not inner.is_centrosymmetric:
            raise ValueError("inner kernel must be centrosymmetric")
        if np.abs(sign) != 1:
            raise ValueError("sign must square to one")

        self.inner = inner
        self.sign = sign

    def __call__(self, x, y, x_plus=None, x_minus=None):
        x, y = _check_domain(self, x, y)

        # The reduced kernel is defined only over the interval [0, 1], which
        # means we must add one to get the x_plus for the inner kernels.  We
        # can compute this as 1 + x, since we are away from -1.
        x_plus = 1 + x_plus

        K_plus = self.inner(x, y, x_plus, x_minus)
        K_minus = self.inner(x, -y, x_plus, x_minus)
        return K_plus + K_minus if self.sign == 1 else K_plus - K_minus

    @property
    def xrange(self):
        _, xmax = self.inner.xrange
        return 0, xmax

    @property
    def yrange(self):
        _, ymax = self.inner.yrange
        return 0, ymax

    def sve_hints(self, eps):
        return _SVEHintsReduced(self.inner.sve_hints(eps))

    @property
    def is_centrosymmetric(self):
        """True iff K(x,y) = K(-x, -y)"""
        return False

    def get_symmetrized(self, sign):
        raise RuntimeError("cannot symmetrize twice")

    @property
    def ypower(self): return self.inner.ypower

    @property
    def conv_radius(self): return self.inner.conv_radius


class _SVEHintsReduced(SVEHintsBase):
    def __init__(self, inner_hints):
        self.inner_hints = inner_hints

    @property
    def ngauss(self): return self.inner_hints.ngauss

    @property
    def segments_x(self): return _symm_segments(self.inner_hints.segments_x)

    @property
    def segments_y(self): return _symm_segments(self.inner_hints.segments_y)

    @property
    def nsvals(self): return (self.inner_hints.nsvals + 1) // 2


class _KernelFFlatOdd(ReducedKernel):
    """Fermionic analytical continuation kernel, odd.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``, the fermionic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``::

        K(x, y) == -sinh(Λ/2 * x * y) / cosh(Λ/2 * y)
    """
    def __call__(self, x, y, x_plus=None, x_minus=None):
        result = super().__call__(x, y, x_plus, x_minus)

        # For x * y around 0, antisymmetrization introduces cancellation, which
        # reduces the relative precision.  To combat this, we replace the
        # values with the explicit form
        v_half = self.inner.lambda_/2 * y
        xy_small = x * v_half < 1
        cosh_finite = v_half < 85
        np.divide(-np.sinh(v_half * x, where=xy_small),
                  np.cosh(v_half, where=cosh_finite),
                  out=result, where=np.logical_and(xy_small, cosh_finite))
        return result


class _KernelBFlatOdd(ReducedKernel):
    """Bosonic analytical continuation kernel, odd.

    In dimensionless variables ``x = 2*τ/β - 1``, ``y = β*ω/Λ``, the fermionic
    integral kernel is a function on ``[-1, 1] x [-1, 1]``::

            K(x, y) = -y * sinh(Λ/2 * x * y) / sinh(Λ/2 * y)
    """
    def __call__(self, x, y, x_plus=None, x_minus=None):
        result = super().__call__(x, y, x_plus, x_minus)

        # For x * y around 0, antisymmetrization introduces cancellation, which
        # reduces the relative precision.  To combat this, we replace the
        # values with the explicit form
        v_half = self.inner.lambda_/2 * y
        xv_half = x * v_half
        xy_small = xv_half < 1
        sinh_range = np.logical_and(v_half > 1e-200, v_half < 85)
        np.divide(
            np.multiply(-y, np.sinh(xv_half, where=xy_small), where=xy_small),
            np.sinh(v_half, where=sinh_range),
            out=result, where=np.logical_and(xy_small, sinh_range))
        return result


def matrix_from_gauss(kernel, gauss_x, gauss_y):
    """Compute matrix for kernel from Gauss rule"""
    # (1 +- x) is problematic around x = -1 and x = 1, where the quadrature
    # nodes are clustered most tightly.  Thus we have the need for the
    # matrix method.
    return kernel(gauss_x.x[:,None], gauss_y.x[None,:],
                  gauss_x.x_forward[:,None], gauss_x.x_backward[:,None])


def _check_domain(kernel, x, y):
    """Check that arguments lie within the correct domain"""
    x = np.asarray(x)
    xmin, xmax = kernel.xrange
    if not (x >= xmin).all() or not (x <= xmax).all():
        raise ValueError("x values not in range [{:g},{:g}]".format(xmin, xmax))

    y = np.asarray(y)
    ymin, ymax = kernel.yrange
    if not (y >= ymin).all() or not (y <= ymax).all():
        raise ValueError("y values not in range [{:g},{:g}]".format(ymin, ymax))
    return x, y


def _symm_segments(x):
    x = np.asarray(x)
    if not np.allclose(x, -x[::-1]):
        raise ValueError("segments must be symmetric")
    xpos = x[x.size // 2:]
    if xpos[0] != 0:
        xpos = np.hstack([0, xpos])
    return xpos


def _compute_uv(lambda_, x, y, x_plus=None, x_minus=None):
    if x_plus is None:
        x_plus = 1 + x
    if x_minus is None:
        x_minus = 1 - x
    u_plus = .5 * x_plus
    u_minus = .5 * x_minus
    v = lambda_ * y
    return u_plus, u_minus, v
