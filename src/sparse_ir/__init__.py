"""
Intermediate representation (IR) for many-body propagators
==========================================================
"""
__copyright__ = "Copyright (C) 2020-2021 Markus Wallerberger and others"
__license__ = "MIT"
__version__ = "0.3.2"

min_xprec_version = "1.0"

try:
    import xprec as xprec
    from pkg_resources import parse_version
    from warnings import warn
    if parse_version(xprec.__version__) < parse_version(min_xprec_version):
        warn(f"xprec is too old! Please use xprec>={min_xprec_version}.")
except ImportError:
    pass

from .kernel import KernelFFlat, KernelBFlat
from .sve import compute as compute_sve
from .basis import IRBasis, FiniteTempBasis
from .sampling import TauSampling, MatsubaraSampling
