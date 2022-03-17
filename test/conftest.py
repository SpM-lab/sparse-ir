# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
#
# This file is available from EVERY test in the directory.  This is why
# we use it to compute the bases ONCE.
import pytest
import sparse_ir


@pytest.fixture(scope="package")
def sve_logistic():
    """SVE of the logistic kernel for Lambda = 42"""
    print("Precomputing SVEs for logistic kernel ...")
    return {
        10:     sparse_ir.compute_sve(sparse_ir.LogisticKernel(10)),
        42:     sparse_ir.compute_sve(sparse_ir.LogisticKernel(42)),
        10_000: sparse_ir.compute_sve(sparse_ir.LogisticKernel(10_000))
        }


@pytest.fixture(scope="package")
def sve_reg_bose():
    """SVE of the logistic kernel for Lambda = 42"""
    print("Precomputing SVEs for regularized Bose kernel ...")
    return {
        10:     sparse_ir.compute_sve(sparse_ir.RegularizedBoseKernel(10)),
        10_000: sparse_ir.compute_sve(sparse_ir.RegularizedBoseKernel(10_000))
        }
