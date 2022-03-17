from .basis import FiniteTempBasis, finite_temp_bases
from .sampling import TauSampling, MatsubaraSampling


class FiniteTempBasisSet:
    """Class for holding IR bases and sparse-sampling objects

    An object of this class holds IR bases for fermion and bosons
    and associated sparse-sampling objects
    """
    def __init__(self, beta, wmax, eps, sve_result=None):
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

        self.eps = eps

        # Tau sampling
        self.smpl_tau_f = TauSampling(self.basis_f)
        self.smpl_tau_b = TauSampling(self.basis_b)

        # Matsubara sampling
        self.smpl_wn_f = MatsubaraSampling(self.basis_f)
        self.smpl_wn_b = MatsubaraSampling(self.basis_b)

    @property
    def beta(self): return self.basis_f.beta
    
    @property
    def wmax(self): return self.basis_f.wmax
    
    @property
    def sve_result(self): return self.basis_f.sve_result

    @property
    def tau(self): return self.smpl_tau_f.sampling_points

    @property
    def wn_f(self): return self.smpl_wn_f.sampling_points

    @property
    def wn_b(self): return self.smpl_wn_b.sampling_points
