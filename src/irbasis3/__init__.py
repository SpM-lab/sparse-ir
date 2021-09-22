"""
Intermediate representation (IR) for many-body propagators
==========================================================
"""
__copyright__ = "Copyright (C) 2020-2021 Markus Wallerberger and others"
__license__ = "MIT"
__version__ = "3.0-alpha1"

from .kernel import KernelFFlat, KernelBFlat
from .sve import compute as compute_sve
from .basis import IRBasis, FiniteTempBasis
