# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT

import numpy as np
from warnings import warn


class KernelBase:
    """Integral kernel `K(x, y)`.

    Abstract base class for an integral kernel, i.e., a real binary function
    `K(x, y)` used in a Fredhold integral equation of the first kind:

                        u(x) = ∫ K(x, y) * v(y) * dy

    where `x ∈ [xmin, xmax]` and `y ∈ [ymin, ymax]`.  For its SVE to exist,
    the kernel must be square-integrable, for its singular values to decay
    exponentially, it must be smooth.
    """
    def __init__(self, xmin=-1, xmax=1, ymin=-1, ymax=1):
        """Initialize a kernel over [xmin, xmax] x [ymin, ymax]"""
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __call__(self, x, y, x_plus=None, x_minus=None):
        """Evaluate kernel at point (x, y)

        For given `x, y`, return the value of `K(x, y)`. The arguments may
        be numpy arrays, in which case the function shall be evaluated over
        the broadcast arrays.

        The parameters `x_plus` and `x_minus`, if given, shall contain the
        values of `x - xmin` and `xmax - x`, respectively.  This is useful
        if either difference is to be formed and cancellation expected.
        """
        raise NotImplementedError()

    def hints(self, eps):
        """Provide discretisation hints for the SVE routines.

        Advises the SVE routines of discretisation parameters suitable in
        tranforming the (infinite) SVE into an (finite) SVD problem.
        """
        raise NotImplementedError()

    @property
    def is_centrosymmetric(self):
        """True iff K(x,y) = K(-x, -y) for all (x, y)"""
        raise NotImplementedError()

    def get_symmetrized(self, sign):
        """Return symmetrized kernel `K(x, y) + sign * K(x, -y)`.

        By default, this returns a simple wrapper over the current instance
        which naively performs the sum.  You may want to override if this
        to avoid cancellation.
        """
        return ReducedKernel(self, sign)


class SVEHints:
    """Discretization hints for singular value expansion of a given kernel.

    Attributes:
    -----------
     - `segments_x` : List of segments on the `x` axis for the associated
       piecewise polynomial.  Should reflect the approximate position of
       roots of a high-order singular function in `x`.

     - `segments_y` : Same for `y`.

     - `ngauss` : Gauss-Legendre order to use to guarantee accuracy.

     - `nsvals` : Upper bound on the number of singular values above the
       given threshold
    """
    def __init__(self, segments_x, segments_y, ngauss, nsvals):
        self.segments_x = segments_x
        self.segments_y = segments_y
        self.ngauss = ngauss
        self.nsvals = nsvals


class KernelFFlat(KernelBase):
    """Fermionic analytical continuation kernel.

    In dimensionless variables `x = 2*τ/β - 1`, `y = β*ω/Λ`, the fermionic
    integral kernel is a function on `[-1, 1] x [-1, 1]`:

            K(x, y) = exp(-Λ * y * (x + 1)/2) / (1 + exp(-Λ*y))
    """
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def __call__(self, x, y, x_plus=None, x_minus=None):
        """Evaluate kernel at point (x, y)"""
        x, y = _check_domain(self, x, y)
        u_plus, u_minus, v = _compute_uv(self.lambda_, x, y, x_plus, x_minus)
        return self._compute(u_plus, u_minus, v)

    def _compute(self, u_plus, u_minus, v):
        """Compute kernel in reduced variables"""
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

    def hints(self, eps):
        segments_x = self._segments_x()
        segments_y = self._segments_y()
        nsvals = self._nsvals()
        ngauss = 10
        return SVEHints(segments_x, segments_y, ngauss, nsvals)

    def _segments_x(self):
        nzeros = max(int(np.round(15 * np.log10(self.lambda_))), 1)
        diffs = 1./np.cosh(.143 * np.arange(nzeros))
        zeros_pos = diffs.cumsum()
        zeros_pos /= zeros_pos[-1]
        return np.concatenate((-zeros_pos[::-1], [0], zeros_pos))

    def _segments_y(self):
        # Zeros around -1 and 1 are distributed asymptotically identical
        leading_diffs = np.array([
            0.01523, 0.03314, 0.04848, 0.05987, 0.06703, 0.07028, 0.07030,
            0.06791, 0.06391, 0.05896, 0.05358, 0.04814, 0.04288, 0.03795,
            0.03342, 0.02932, 0.02565, 0.02239, 0.01951, 0.01699])

        nzeros = max(int(np.round(20 * np.log10(self.lambda_))), 2)
        if nzeros < 20:
            leading_diffs = leading_diffs[:nzeros]
        diffs = .25 / np.exp(.141 * np.arange(nzeros))
        diffs[:leading_diffs.size] = leading_diffs
        zeros = diffs.cumsum()
        zeros = zeros[:-1] / zeros[-1]
        zeros -= 1
        return np.concatenate(([-1], zeros, [0], -zeros[::-1], [1]))

    def _nsvals(self):
        log10_lambda = max(1, np.log10(self.lambda_))
        return int(np.round((25 + log10_lambda) * log10_lambda))

    @property
    def is_centrosymmetric(self):
        return True

    def get_symmetrized(self, sign):
        if sign == -1:
            return _KernelFFlatOdd(self, sign)
        return super().get_symmetrized(sign)


class KernelBFlat(KernelBase):
    """Bosonic analytical continuation kernel.

    In dimensionless variables `x = 2*τ/β - 1`, `y = β*ω/Λ`, the fermionic
    integral kernel is a function on `[-1, 1] x [-1, 1]`:

            K(x, y) = y * exp(-Λ * y * (x + 1)/2) / (exp(-Λ*y) - 1)

    Care has to be taken in evaluating this expression around `y == 0`.
    """
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def __call__(self, x, y, x_plus=None, x_minus=None):
        """Evaluate kernel at point (x, y)"""
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

        # The expression `v / (exp(v) - 1)` is tricky to evaluate: firstly, it
        # has a singularity at v=0, which can be cured by treating that case
        # separately.  Secondly, the denominator loses precision around 0 since
        # exp(v) = 1 + v + ..., which can be avoided using expm1(...)
        #tiny = np.finfo(dtype).tiny
        tiny = 1e-200
        with np.errstate(invalid='ignore'):
            denom = np.where(abs_v < tiny, 1, abs_v / np.expm1(-abs_v))

        return -1/dtype.type(self.lambda_) * enum * denom

    def hints(self, eps):
        segments_x = self._segments_x()
        segments_y = self._segments_y()
        nsvals = self._nsvals()
        ngauss = 10
        return SVEHints(segments_x, segments_y, ngauss, nsvals)

    def _segments_x(self):
        # Somewhat less accurate ...
        nzeros = 15 * np.log10(self.lambda_)
        diffs = 1./np.cosh(.18 * np.arange(nzeros))
        zeros_pos = diffs.cumsum()
        zeros_pos /= zeros_pos[-1]
        return np.concatenate((-zeros_pos[::-1], [0], zeros_pos))

    def _segments_y(self):
        # Zeros around -1 and 1 are distributed asymptotically identical
        leading_diffs = np.array([
            0.01363, 0.02984, 0.04408, 0.05514, 0.06268, 0.06679, 0.06793,
            0.06669, 0.06373, 0.05963, 0.05488, 0.04987, 0.04487, 0.04005,
            0.03553, 0.03137, 0.02758, 0.02418, 0.02115, 0.01846])

        nzeros = max(int(np.round(20 * np.log10(self.lambda_))), 20)
        i = np.arange(nzeros)
        diffs = .12/np.exp(.0337 * i * np.log(i+1))
        #diffs[:leading_diffs.size] = leading_diffs
        zeros = diffs.cumsum()
        zeros = zeros[:-1] / zeros[-1]
        zeros -= 1
        return np.concatenate(([-1], zeros, [0], -zeros[::-1], [1]))

    def _nsvals(self):
        log10_lambda = max(1, np.log10(self.lambda_))
        return int(28 * log10_lambda)

    @property
    def is_centrosymmetric(self):
        return True

    def get_symmetrized(self, sign):
        if sign == -1:
            return _KernelBFlatOdd(self, sign)
        return super().get_symmetrized(sign)


class ReducedKernel(KernelBase):
    """Restriction of centrosymmetric kernel to positive interval.

    For a kernel `K` on `[-1, 1] x [-1, 1]` that is centrosymmetrix, i.e.,
    `K(x, y) == K(-x, -y)`, it is straight-forward to show that the left/right
    singular vectors can be chosen as either odd or even functions.

    Consequentially, they are singular functions of a reduced kernel `K_red`
    on `[0, 1] x [0, 1]` that is given as either:

        K_red(x, y) = K(x, y) + sign * K(x, -y)

    This kernel is what this class represents.  The full singular functions can
    be reconstructed by (anti-)symmetrically continuing them to the negative
    axis.
    """
    def __init__(self, inner, sign=1):
        if not inner.is_centrosymmetric:
            raise ValueError("inner kernel must be centrosymmetric")
        if np.abs(sign) != 1:
            raise ValueError("sign must square to one")

        super().__init__(0, inner.xmax, 0, inner.ymax)
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

    def hints(self, eps):
        inner_hints = self.inner.hints(eps)
        segments_x = _symm_segments(inner_hints.segments_x)
        segments_y = _symm_segments(inner_hints.segments_y)
        ngauss = inner_hints.ngauss
        nsvals = (inner_hints.nsvals + 1) // 2
        return SVEHints(segments_x, segments_y, ngauss, nsvals)

    @property
    def is_centrosymmetric(self):
        """True iff K(x,y) = K(-x, -y)"""
        return False

    def get_symmetrized(self, sign):
        raise RuntimeError("cannot symmetrize twice")


class _KernelFFlatOdd(ReducedKernel):
    """Fermionic analytical continuation kernel, odd.

    In dimensionless variables `x = 2*τ/β - 1`, `y = β*ω/Λ`, the fermionic
    integral kernel is a function on `[-1, 1] x [-1, 1]`:

            K(x, y) = -sinh(Λ/2 * x * y) / cosh(Λ/2 * y)
    """
    def __call__(self, x, y, x_plus=None, x_minus=None):
        naive_result = super().__call__(x, y, x_plus, x_minus)

        # For x * y around 0, antisymmetrization introduces cancellation, which
        # reduces the relative precision.  To combat this, we replace the
        # values with the explicit form
        v_half = self.inner.lambda_/2 * y
        with np.errstate(over='ignore', invalid='ignore'):
            antisymm_result = -np.sinh(v_half * x) / np.cosh(v_half)
        return np.where(np.logical_and(x * v_half < 1, np.abs(v_half) < 100),
                        antisymm_result, naive_result)


class _KernelBFlatOdd(ReducedKernel):
    """Bosonic analytical continuation kernel, odd.

    In dimensionless variables `x = 2*τ/β - 1`, `y = β*ω/Λ`, the fermionic
    integral kernel is a function on `[-1, 1] x [-1, 1]`:

            K(x, y) = -y * sinh(Λ/2 * x * y) / sinh(Λ/2 * y)
    """
    def __call__(self, x, y, x_plus=None, x_minus=None):
        naive_result = super().__call__(x, y, x_plus, x_minus)

        # For x * y around 0, antisymmetrization introduces cancellation, which
        # reduces the relative precision.  To combat this, we replace the
        # values with the explicit form
        v_half = self.inner.lambda_/2 * y
        tiny = 1e-200 #np.finfo(naive_result.dtype).tiny
        with np.errstate(over='ignore', invalid='ignore'):
            antisymm_result = -y * np.sinh(v_half * x) / np.sinh(v_half)

        return np.where(np.logical_and(x * v_half < 1, v_half > tiny),
                        antisymm_result, naive_result)


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
    if not (x >= kernel.xmin).all() or not (x <= kernel.xmax).all():
        raise ValueError("x values not in range [{:g},{:g}]"
                         .format(kernel.xmin, kernel.xmax))
    y = np.asarray(y)
    if not (y >= kernel.ymin).all() or not (y <= kernel.ymax).all():
        raise ValueError("y values not in range [{:g},{:g}]"
                         .format(kernel.ymin, kernel.ymax))
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
