# Copyright (C) 2020-2025 Satoshi Terasaki, Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
from . import _util
import numpy as np
from ctypes import c_int, c_bool, byref
from . import abstract
from . import basis
from pylibsparseir.core import basis_get_default_tau_sampling_points_ext, basis_get_default_matsus_ext, _lib
from pylibsparseir.constants import COMPUTATION_SUCCESS

class AugmentedBasis(abstract.AbstractBasis):
    """Augmented basis on the imaginary-time/frequency axis.

    Groups a set of additional functions, ``augmentations``, with a given
    ``basis``.  The augmented functions then form the first basis
    functions, while the rest is provided by the regular basis, i.e.::

        u[l](tau) == augmentations[l](tau) if l < naug else basis.u[l-naug](tau),

    where ``naug = len(augmentations)`` is the number of added basis functions
    through augmentation.  Similarly, in Matsubara frequency,
    ``uhat[l](n) == augmentations[l].hat(n)`` for ``l < naug``, at the reduced
    frequency ``n`` (ν = nπ/β).  ``tau`` may lie anywhere in [-β, β], with the
    extension and endpoint rule of :py:attr:`FiniteTempBasis.u
    <sparse_ir.FiniteTempBasis.u>`.  :py:attr:`significance` and
    :py:attr:`accuracy` are those of the underlying basis.

    Augmentation is useful in constructing bases for vertex-like quantities
    such as self-energies `[1]`_.  It is also useful when constructing a
    two-point kernel that serves as a base for multi-point functions `[2]`_.

    Example:
        For constructing the vertex basis and the augmented basis, one can
        use::

            import sparse_ir, sparse_ir.augment as aug
            basis = sparse_ir.FiniteTempBasis('B', beta=10, wmax=2.0)
            vertex_basis = aug.AugmentedBasis(basis, aug.MatsubaraConst)
            aug_basis = aug.AugmentedBasis(basis, aug.TauConst, aug.TauLinear)

    Warning:
        Bases augmented with `TauConst` and `TauLinear` tend to be poorly
        conditioned.  Care must be taken while fitting and compactness should
        be enforced if possible to regularize the problem.

        While vertex bases, i.e., bases augmented with `MatsubaraConst`, stay
        reasonably well-conditioned, it is still good practice to treat the
        Hartree--Fock term separately rather than including it in the basis,
        if possible.

    See also:
         - :class:`MatsubaraConst` for vertex basis `[1]`_
         - :class:`TauConst`, :class:`TauLinear` for multi-point `[2]`_

    .. _[1]: https://doi.org/10.1103/PhysRevResearch.3.033168
    .. _[2]: https://doi.org/10.1103/PhysRevB.97.205111
    """
    def __init__(self, basis, *augmentations):
        """
        Arguments:
            basis (FiniteTempBasis):
                Basis to augment.
            *augmentations:
                Augmentation classes, such as ``TauConst``, which are created
                for the basis with ``create(basis)``, or instances, which must
                have the ``beta`` of the basis; TauConst and TauLinear
                require a bosonic basis.
        """
        augmentations = tuple(_augmentation_factory(basis, *augmentations))
        self._basis = basis
        self._augmentations = augmentations
        self._naug = len(augmentations)

        self._u = AugmentedTauFunction(self._basis.u, augmentations)
        self._uhat = AugmentedMatsubaraFunction(
                        self._basis.uhat, [aug.hat for aug in augmentations])

    @property
    def basis(self):
        return self._basis

    @property
    def u(self):
        return self._u

    @property
    def uhat(self):
        return self._uhat

    @property
    def statistics(self):
        return self._basis.statistics

    def __getitem__(self, index):
        stop = _util.slice_to_size(index, self.size)
        if stop <= self._naug:
            raise ValueError("Cannot truncate to only augmentation")
        return AugmentedBasis(self._basis[:stop - self._naug],
                              *self._augmentations)

    @property
    def shape(self):
        return self.size,

    @property
    def size(self):
        return self._naug + self._basis.size

    @property
    def significance(self):
        """Significance of the underlying basis"""
        return self._basis.significance

    @property
    def accuracy(self):
        """Accuracy of the underlying basis"""
        return self._basis.accuracy

    @property
    def lambda_(self):
        return self._basis.lambda_

    @property
    def beta(self):
        return self._basis.beta

    @property
    def wmax(self):
        return self._basis.wmax

    def default_tau_sampling_points(self, *, npoints=None, use_positive_taus=True):
        """Get default tau sampling points for augmented basis.

        They are the roots of U_npoints, i.e. the default points of an IR
        basis of size ``npoints``.

        Arguments:
            npoints (int):
                Minimum number of sampling points to return. If None, uses self.size.
            use_positive_taus (bool):
                If True (default), fold the points into [0, β) with
                ``np.mod`` and sort them; they lie in (0, β).
                If False, the points are unfolded: they lie in (-β/2, β/2]
                and come in pairs ±τ, plus β/2 when their number is odd.

                .. versionadded:: 1.2
        """
        if npoints is None:
            npoints = self.size

        # Return the sampling points of the underlying basis, but since we
        # use the size of self, we add two further points.  One then has to
        # hope that these give good sampling points.

        points = basis_get_default_tau_sampling_points_ext(self._basis._ptr, npoints)
        
        if use_positive_taus:
            points = np.mod(points, self.beta)
            points = np.sort(points)
        
        return points

    def default_matsubara_sampling_points(self, *, positive_only=False):
        """Get default Matsubara sampling points for augmented basis.

        This method provides default sampling points for Matsubara frequencies
        when using an augmented basis: reduced Matsubara frequencies n,
        computed from the underlying basis for ``self.size`` functions.  With
        ``positive_only=True`` they are the non-negative half (n >= 0) of the
        full set, as for a plain basis.
        """
        if positive_only:
            # Requesting the positive-only variant from C with the buffer size
            # as the point limit returned a truncated, zero-padded set.
            points = self.default_matsubara_sampling_points()
            return points[points >= 0]
        # Call C function directly with correct 5 arguments
        # The pylibsparseir wrapper basis_get_n_default_matsus_ext has a bug - missing 2nd bool arg
        # C signature: (basis_ptr, _Bool positive_only, _Bool fence, c_int n_points, POINTER(c_int) n_points_returned)
        n_points_returned = c_int()
        fence = False  # fence parameter (second bool)
        status = _lib.spir_basis_get_n_default_matsus_ext(
            self._basis._ptr,
            c_bool(positive_only),
            c_bool(fence),
            c_int(self.size),
            byref(n_points_returned)
        )
        if status != COMPUTATION_SUCCESS:
            raise RuntimeError(f"Failed to get number of default Matsubara points: {status}")

        points = np.zeros(n_points_returned.value, dtype=np.int64)
        basis_get_default_matsus_ext(self._basis._ptr, positive_only, points)
        return points

    @property
    def is_well_conditioned(self):
        """True only for a vertex basis, i.e. a well-conditioned basis
        augmented with MatsubaraConst alone"""
        wbasis = self._basis.is_well_conditioned
        waug = (len(self._augmentations) == 1
                and isinstance(self._augmentations[0], MatsubaraConst))
        return wbasis and waug


class _AugmentedFunction:
    def __init__(self, fbasis, faug):
        #if fbasis.ndim != 1:
        #    raise ValueError("must have vector of functions as fbasis")
        self._fbasis = fbasis
        self._faug = faug
        self._naug = len(faug)

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return self.size,

    @property
    def size(self):
        return self._naug + self._fbasis.size

    def __call__(self, x):
        x = np.asarray(x)
        # The basis part may be a single function (a truncation to naug + 1).
        fbasis_x = np.reshape(self._fbasis(x), (-1,) + x.shape)
        faug_x = [faug_l(x)[None] for faug_l in self._faug]
        f_x = np.concatenate(faug_x + [fbasis_x], axis=0)
        assert f_x.shape[1:] == x.shape
        return f_x

    def __getitem__(self, l):
        if isinstance(l, slice):
            stop = _util.slice_to_size(l, self.size)
            if stop <= self._naug:
                raise ValueError(
                    "cannot truncate to only the augmentation functions")
            # Keep the subclass: it carries xmin/xmax/deriv or zeta.
            return type(self)(self._fbasis[:stop-self._naug], self._faug)
        elif np.ndim(l) != 0:
            raise TypeError("augmented function sets take an integer index or "
                            f"a slice, got {l!r}")
        else:
            # Resolve a negative index against the whole set: without this,
            # u[-1] would return the last *augmentation* instead of the last
            # basis function.
            l, = _util.resolve_function_indices(l, self.size)
            if l < self._naug:
                return self._faug[l]
            else:
                return self._fbasis[l-self._naug]


class AugmentedTauFunction(_AugmentedFunction):
    @property
    def xmin(self):
        return self._fbasis.xmin

    @property
    def xmax(self):
        return self._fbasis.xmax

    def deriv(self, n=1):
        """Get polynomial for the n'th derivative"""
        dbasis = self._fbasis.deriv(n)
        daug = [faug_l.deriv(n) for faug_l in self._faug]
        return AugmentedTauFunction(dbasis, daug)


class AugmentedMatsubaraFunction(_AugmentedFunction):
    @property
    def zeta(self):
        return self._fbasis.zeta


class AbstractAugmentation:
    """Scalar function in imaginary time/frequency.

    This represents a single function in imaginary time and frequency,
    together with some auxiliary methods that make it suitable for augmenting
    a basis: ``aug(tau)`` is the function U(τ) at imaginary time τ in
    [-β, β], and ``aug.hat(n)`` its Fourier transform
    Û(iν) = ∫₀^β dτ e^{iντ} U(τ) at the reduced frequency n, ν = nπ/β.

    See also:
        :class:`AugmentedBasis`
    """
    @classmethod
    def create(cls, basis):
        """Factory method constructing an augmented term for a basis"""
        raise NotImplementedError()

    def __call__(self, tau):
        """Evaluate the function at imaginary time ``tau`` in [-β, β]"""
        raise NotImplementedError()

    def deriv(self, n):
        """Derivative of order ``n`` of the function"""
        raise NotImplementedError()

    def hat(self, n):
        """Evaluate the Fourier transform at reduced frequency ``n`` (ν = nπ/β)"""
        raise NotImplementedError()


class TauConst(AbstractAugmentation):
    """Constant in imaginary time: ``1/sqrt(beta)`` on [0, β], periodic.

    It is normalized on [0, β].  Its Fourier transform is ``sqrt(beta)`` at
    n = 0 and zero at every other reduced frequency.  Defined for bosons
    only; ``statistics='F'`` raises :class:`ValueError`.  It accepts ``tau``
    in [-β, β] and is ``1/sqrt(beta)`` everywhere there, including the
    endpoints ±0.0 and ±β.

    .. versionchanged:: 1.2
        Added statistics parameter and support for [-β, β] range.
    """
    @classmethod
    def create(cls, basis):
        return cls(basis.beta, basis.statistics)

    def __init__(self, beta, statistics='B'):
        """
        Arguments:
            beta (float):
                Inverse temperature.
            statistics (str):
                'B' (default); 'F' raises ValueError.
        """
        if beta <= 0:
            raise ValueError("temperature must be positive")
        if statistics not in ('F', 'B'):
            raise ValueError("statistics must be 'F' or 'B'")
        _check_bosonic_statistics(statistics, "TauConst")
        self._beta = beta
        self._statistics = statistics

    def __call__(self, tau):
        tau_normalized, sign = _util.normalize_tau(self._statistics, tau, self._beta)
        return sign / np.sqrt(self._beta)

    def deriv(self, n=1):
        if n == 0:
            return self
        else:
            return lambda tau: np.zeros(np.shape(tau))

    def hat(self, n):
        zeta = 1 if self._statistics == 'F' else 0
        n = _util.check_reduced_matsubara(n, zeta=zeta)
        return np.sqrt(self._beta) * (n == 0).astype(complex)


class TauLinear(AbstractAugmentation):
    """Linear in imaginary time: ``sqrt(3/beta) * (2*tau/beta - 1)`` on [0, β], periodic.

    It is normalized on [0, β] and antisymmetric around β/2; its Fourier
    transform is ``2*sqrt(3/beta)/(1j*nu)`` with ν = nπ/β, and zero at n = 0.
    Defined for bosons only; ``statistics='F'`` raises :class:`ValueError`.

    It accepts ``tau`` in [-β, β]: negative times follow the periodic
    extension f(τ) = f(τ + β), and the endpoints are one-sided limits, so
    ``+0.0`` and ``-beta`` give f(0⁺) = -sqrt(3/beta), while ``beta`` and
    ``-0.0`` give f(β⁻) = +sqrt(3/beta).

    .. versionchanged:: 1.2
        Added statistics parameter and support for [-β, β] range.
    """
    @classmethod
    def create(cls, basis):
        return cls(basis.beta, basis.statistics)

    def __init__(self, beta, statistics='B'):
        """
        Arguments:
            beta (float):
                Inverse temperature.
            statistics (str):
                'B' (default); 'F' raises ValueError.
        """
        if beta <= 0:
            raise ValueError("temperature must be positive")
        if statistics not in ('F', 'B'):
            raise ValueError("statistics must be 'F' or 'B'")
        _check_bosonic_statistics(statistics, "TauLinear")
        self._beta = beta
        self._statistics = statistics
        self._norm = np.sqrt(3/beta)

    def __call__(self, tau):
        tau_normalized, sign = _util.normalize_tau(self._statistics, tau, self._beta)
        x = 2/self._beta * tau_normalized - 1
        return sign * self._norm * x

    def deriv(self, n=1):
        if n == 0:
            return self
        elif n == 1:
            c = self._norm * 2/self._beta
            return lambda tau: np.full(np.shape(tau), c)
        else:
            return lambda tau: np.zeros(np.shape(tau))

    def hat(self, n):
        zeta = 1 if self._statistics == 'F' else 0
        n = _util.check_reduced_matsubara(n, zeta=zeta)
        w = np.pi / self._beta * np.asarray(n, dtype=np.float64)
        inv_w = np.zeros_like(w)
        np.divide(1.0, w, out=inv_w, where=(w != 0))
        return self._norm * 2/1j * inv_w


class MatsubaraConst(AbstractAugmentation):
    """Constant in Matsubara, undefined in imaginary time.

    This augmentation is constant in Matsubara frequency space,
    ``hat(n) == 1`` for every integer ``n`` (without a factor of √β, and
    without a check of the parity of ``n``), and returns NaN for ``tau`` in
    [-β, β] (outside, :class:`ValueError`). The statistics parameter is
    accepted for type consistency but does not affect the behavior.

    .. versionchanged:: 1.2
        Accepts tau in [-β, β] range (previously [0, β]).
        Added statistics parameter for consistency.
    """
    @classmethod
    def create(cls, basis):
        return cls(basis.beta, basis.statistics)

    def __init__(self, beta, statistics=None):
        """
        Arguments:
            beta (float):
                Inverse temperature.
            statistics (str, optional):
                'F' for Fermionic or 'B' for Bosonic. Accepted for type consistency
                but behavior is identical for both.
        """
        if beta <= 0:
            raise ValueError("temperature must be positive")
        if statistics is not None and statistics not in ('F', 'B'):
            raise ValueError("statistics must be 'F' or 'B'")
        self._beta = beta
        self._statistics = statistics

    def __call__(self, tau):
        tau = _util.check_range(tau, -self._beta, self._beta)
        return np.broadcast_to(np.nan, tau.shape)

    def deriv(self, n=1):
        return self

    def hat(self, n):
        n = _util.check_reduced_matsubara(n)
        return np.broadcast_to(1.0, n.shape)




def _augmentation_factory(basis, *augs):
    for aug in augs:
        if isinstance(aug, AbstractAugmentation):
            _check_augmentation_instance(aug, basis)
            yield aug
        else:
            yield aug.create(basis)


def _check_augmentation_instance(aug, basis):
    """An instance must have been built for the basis it augments."""
    name = type(aug).__name__
    beta = getattr(aug, '_beta', None)
    if beta is not None and not np.isclose(beta, basis.beta, rtol=1e-12, atol=0):
        raise ValueError(f"{name} has beta = {beta}, but the basis has "
                         f"beta = {basis.beta}")
    # MatsubaraConst does not depend on the statistics; TauConst and
    # TauLinear are bosonic.
    if isinstance(aug, (TauConst, TauLinear)) and basis.statistics != 'B':
        raise ValueError(f"{name} is defined for bosons only, got a basis "
                         f"with statistics {basis.statistics!r}")


def _check_bosonic_statistics(statistics, name):
    if statistics == 'B':
        return
    elif statistics == 'F':
        raise ValueError(f"{name} is defined for bosons only, got statistics 'F'")
    else:
        raise ValueError(f"invalid statistics {statistics!r}, expected 'F' or 'B'")