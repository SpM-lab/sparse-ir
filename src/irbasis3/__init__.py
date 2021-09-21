"""

Copyright (C) 2020-2021 Markus Wallerberger and others
SPDX-License-Identifier: MIT

"""
__version__ = "3.0-alpha1"

from .kernel import KernelFFlat, KernelBFlat
from .sve import compute as compute_sve
