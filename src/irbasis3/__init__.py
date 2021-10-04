"""
Intermediate representation (IR) for many-body propagators
==========================================================
"""
__copyright__ = "Copyright (C) 2020-2021 Markus Wallerberger and others"
__license__ = "MIT"

from .kernel import KernelFFlat, KernelBFlat
from .sve import compute as compute_sve
from .basis import IRBasis, FiniteTempBasis
from .sampling import TauSampling, MatsubaraSampling

from . import _version
__version__ = _version.get_versions()['version']
