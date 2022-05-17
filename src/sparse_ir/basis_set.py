from .basis import FiniteTempBasis, finite_temp_bases
from .sampling import TauSampling, MatsubaraSampling


class FiniteTempBasisSet:
    """Class for holding IR bases and sparse-sampling objects.

    An object of this class holds IR bases for fermions and bosons
    and associated sparse-sampling objects.

    Attributes:
        basis_f (FiniteTempBasis):
            Fermion basis
        basis_b (FiniteTempBasis):
            Boson basis
        beta (float):
            Inverse temperature
        wmax (float):
            Cut-off frequency
        eps (float):
            Cut-off value for singular values
        tau (1D ndarray of float):
            Sampling points in the imaginary-time domain
        wn_f (1D ndarray of int):
            Sampling fermionic frequencies
        wn_b (1D ndarray of int):
            Sampling bosonic frequencies
        smpl_tau_f (TauSampling):
            Sparse sampling for tau & fermion
        smpl_tau_b (TauSampling):
            Sparse sampling for tau & boson
        smpl_wn_f (MatsubaraSampling):
            Sparse sampling for Matsubara frequency & fermion
        smpl_wn_b (MatsubaraSampling):
            Sparse sampling for Matsubara frequency & boson
        sve_result (tuple of three ndarray objects):
            Results of SVE
    """
    def __init__(self, beta, wmax, eps=None, sve_result=None):
        """
        Create basis sets for fermion and boson and
        associated sampling objects.
        Fermion and bosonic bases are constructed by SVE of the logistic kernel.
        """
        if sve_result is None:
            # Create bases by sve of the logistic kernel
            self.basis_f, self.basis_b = finite_temp_bases(beta, wmax, eps)
        else:
            # Create bases using the given sve results
            self.basis_f = FiniteTempBasis("F", beta, wmax, eps, sve_result=sve_result)
            self.basis_b = FiniteTempBasis("B", beta, wmax, eps, sve_result=sve_result)

        # Tau sampling
        self.smpl_tau_f = TauSampling(self.basis_f)
        self.smpl_tau_b = TauSampling(self.basis_b)

        # Matsubara sampling
        self.smpl_wn_f = MatsubaraSampling(self.basis_f)
        self.smpl_wn_b = MatsubaraSampling(self.basis_b)

    @property
    def lambda_(self): return self.basis_f.lambda_

    @property
    def beta(self): return self.basis_f.beta

    @property
    def wmax(self): return self.basis_f.wmax

    @property
    def accuracy(self): return self.basis_f.accuracy

    @property
    def sve_result(self): return self.basis_f.sve_result

    @property
    def tau(self): return self.smpl_tau_f.sampling_points

    @property
    def wn_f(self): return self.smpl_wn_f.sampling_points

    @property
    def wn_b(self): return self.smpl_wn_b.sampling_points
