"""
Intermediate representation (IR) for many-body propagators
==========================================================
"""
__copyright__ = "2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others"
__license__ = "MIT"
__version__ = "0.95.0"

min_xprec_version = "1.0"

try:
    import xprec as xprec
    from pkg_resources import parse_version
    from warnings import warn
    if parse_version(xprec.__version__) < parse_version(min_xprec_version):
        warn(f"xprec is too old! Please use xprec>={min_xprec_version}.")
except ImportError:
    pass

from .kernel import RegularizedBoseKernel, LogisticKernel
from .sve import compute as compute_sve, SVEResult
from .basis import FiniteTempBasis, finite_temp_bases
from .basis_set import FiniteTempBasisSet
from .sampling import TauSampling, MatsubaraSampling
