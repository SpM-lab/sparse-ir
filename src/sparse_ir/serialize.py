# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import textwrap
import sys

from . import __version__
from . import kernel


class TextFormat:
    def write_kernel(self, f, K):
        if isinstance(K, kernel.LogisticKernel):
            f.write("## Kernel: LogisticKernel; lambda={K.lambda_}\n")
        elif isinstance(K, kernel.RegularizedBoseKernel):
            f.write("## Kernel: RegularizedBoseKernel; lambda={K.lambda_}\n")
        else:
            raise ValueError("Unknown kernel")

    def write_basis(self, f, basis):
        f.write("## DATA FOR INTERMEDIATE REPRESENTATION\n")
        f.write("## Description:\n"
                "## \tData for the intermediate representation, which is\n"
                "## \tthe truncated singular value expansion of an integral\n"
                "## \tkernel.  What follows are first the singular values,\n"
                "## \tthen the left singular functions sampled on piecewise\n"
                "## \tLegendre nodes, then the right singular functions.\n")
        self.write_kernel(f, basis.kernel)
        if hasattr(basis, "beta"):
            f.write(f"## InverseTemperature: {basis.beta}\n")
        f.write(f"## Statistics: {basis.statistics}\n")
        f.write(f"## BasisSize: {basis.size}\n")
        f.write(f"## PiecesLeft: {', '.join(map(str, basis.u.knots))}\n")
        f.write(f"## PiecesRight: {', '.join(map(str, basis.v.knots))}\n")
        f.write(f"## LegendreOrder: {basis.u.order}\n")
        f.write(f"## Generator: sparse-ir; version={__version__}\n")

    def read_basis(self, input):
        pass
