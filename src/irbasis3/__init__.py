"""
Intermediate representation (IR) for many-body propagators
==========================================================
"""
__copyright__ = "Copyright (C) 2020-2021 Markus Wallerberger and others"
__license__ = "MIT"
__version__ = "3.0-alpha6"
__min_xprec_version__ = "1.1.1"

try:
    import xprec as xprec
    from pkg_resources import parse_version
    from warnings import warn
    if parse_version(xprec.__version__) < parse_version(__min_xprec_version__):
        warn(f"xprec is too old! Please use xprec>={__min_xprec_version__}.")
except ImportError:
    pass

from .kernel import KernelFFlat, KernelBFlat
from .sve import compute as compute_sve
from .basis import IRBasis, FiniteTempBasis
from .sampling import TauSampling, MatsubaraSampling
