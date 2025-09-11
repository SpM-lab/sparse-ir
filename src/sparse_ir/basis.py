"""
High-level Python classes for FiniteTempBasis
"""
from typing import Optional
import numpy as np
from pylibsparseir.core import basis_new, basis_get_size, basis_get_svals, basis_get_u, basis_get_v, basis_get_uhat, basis_get_default_tau_sampling_points, basis_get_default_omega_sampling_points, basis_get_default_matsubara_sampling_points
from pylibsparseir.constants import SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC
from .kernel import LogisticKernel
from .abstract import AbstractBasis
from .sve import SVEResult
from .poly import PiecewiseLegendrePolyVector, PiecewiseLegendrePolyFTVector, FunctionSet, FunctionSetFT

class FiniteTempBasis(AbstractBasis):
    """Finite temperature basis for intermediate representation."""

    def __init__(self, statistics: str, beta: float, wmax: float, eps: float, sve_result: Optional[SVEResult] = None, max_size: int =-1):
        """
        Initialize finite temperature basis.

        Parameters:
        -----------
        statistics : str
            'F' for fermions, 'B' for bosons
        beta : float
            Inverse temperature
        wmax : float
            Frequency cutoff
        eps : float
            Accuracy threshold
        """
        self._statistics = statistics
        self._beta = beta
        self._wmax = wmax
        self._lambda = beta * wmax

        # Create kernel
        if statistics == 'F' or statistics == 'B':
            self._kernel = LogisticKernel(self._lambda)
        else:
            raise ValueError(f"Invalid statistics: {statistics} expected 'F' or 'B'")

        # Compute SVE
        if sve_result is None:
            self._sve = SVEResult(self._kernel, eps)
        else:
            self._sve = sve_result

        # Create basis
        stats_int = SPIR_STATISTICS_FERMIONIC if statistics == 'F' else SPIR_STATISTICS_BOSONIC
        self._ptr = basis_new(stats_int, self._beta, self._wmax, self._kernel._ptr, self._sve._ptr, max_size)

        u_funcs = FunctionSet(basis_get_u(self._ptr))
        v_funcs = FunctionSet(basis_get_v(self._ptr))
        uhat_funcs = FunctionSetFT(basis_get_uhat(self._ptr))

        self._s = basis_get_svals(self._ptr)
        self._u = PiecewiseLegendrePolyVector(u_funcs, 0, self._beta)
        self._v = PiecewiseLegendrePolyVector(v_funcs, -self._wmax, self._wmax)
        self._uhat = PiecewiseLegendrePolyFTVector(uhat_funcs)

    @property
    def statistics(self):
        """Quantum statistic ('F' for fermionic, 'B' for bosonic)"""
        return self._statistics

    @property
    def beta(self):
        """Inverse temperature"""
        return self._beta

    @property
    def wmax(self):
        """Real frequency cutoff"""
        return self._wmax

    @property
    def lambda_(self):
        """Basis cutoff parameter, Λ = β * wmax"""
        return self._lambda

    @property
    def size(self):
        return self._s.size

    @property
    def s(self):
        """Singular values."""
        if self._s is None:
            self._s = basis_get_svals(self._ptr)
        return self._s

    @property
    def u(self):
        return self._u

    @property
    def v(self):
        return self._v

    @property
    def uhat(self):
        return self._uhat

    @property
    def significance(self):
        """Relative significance of basis functions."""
        return self.s / self.s[0]

    @property
    def accuracy(self):
        """Overall accuracy bound."""
        return self.s[-1] / self.s[0]

    @property
    def shape(self):
        """Shape of the basis function set"""
        return self.s.shape

    def default_tau_sampling_points(self, npoints=None):
        """Get default tau sampling points."""
        return basis_get_default_tau_sampling_points(self._ptr)

    def default_omega_sampling_points(self, npoints=None):
        """
        Get default omega (real frequency) sampling points.

        Returns the extrema of the highest-order basis function in real frequency.
        These points provide near-optimal conditioning for the basis.

        Parameters
        ----------
        npoints : int, optional
            Ignored (for compatibility with sparse-ir API)

        Returns
        -------
        ndarray
            Default omega sampling points
        """
        from pylibsparseir.core import basis_get_default_omega_sampling_points
        return basis_get_default_omega_sampling_points(self._ptr)

    def default_matsubara_sampling_points(self, npoints=None, positive_only=False):
        """Get default Matsubara sampling points."""
        return basis_get_default_matsubara_sampling_points(self._ptr, positive_only)

    def __repr__(self):
        return (f"FiniteTempBasis(statistics='{self.statistics}', "
                f"beta={self.beta}, wmax={self.wmax}, size={self.size})")

    def __getitem__(self, index):
        """Return basis functions/singular values for given index/indices.

        This can be used to truncate the basis to the n most significant
        singular values: basis[:3].
        """
        # TODO: Implement basis truncation when C API supports it
        raise NotImplementedError("Basis truncation not yet implemented in C API")

    @property
    def kernel(self):
        """The kernel used to generate the basis."""
        return self._kernel

    @property
    def sve_result(self):
        """The singular value expansion result."""
        return self._sve

    def rescale(self, new_lambda):
        """Return a basis for different lambda while keeping the same eps.

        Parameters
        ----------
        new_lambda : float
            The new lambda value (must equal new_beta * new_wmax)

        Returns
        -------
        FiniteTempBasis
            A new basis with the rescaled parameters
        """
        # Calculate new beta and wmax that give the desired lambda
        # We keep the ratio beta/wmax constant
        ratio = self.beta / self.wmax
        new_wmax = np.sqrt(new_lambda / ratio)
        new_beta = new_lambda / new_wmax

        # Get epsilon from the current basis accuracy
        eps = self.accuracy

        return FiniteTempBasis(self.statistics, new_beta, new_wmax, eps)


def finite_temp_bases(beta, wmax, eps=None, **kwargs):
    """
    Construct both fermion and boson bases.

    Returns:
    --------
    tuple
        (fermion_basis, boson_basis)
    """
    fermion_basis = FiniteTempBasis('F', beta, wmax, eps, **kwargs)
    boson_basis = FiniteTempBasis('B', beta, wmax, eps, **kwargs)
    return fermion_basis, boson_basis