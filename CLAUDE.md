# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sparse-ir** is a Python library providing sparse intermediate representation (IR) methods for many-body physics calculations. This is part of the broader **SpM-lab** ecosystem that includes:

- **sparse-ir**: This Python implementation (pure Python with optional C++ acceleration)
- **SparseIR.jl**: Julia implementation (separate repository) 
- **libsparseir**: High-performance C++ core library (separate repository)
- **pylibsparseir**: Python bindings for libsparseir (dependency of this project)

## Development Commands

### Core Development Workflow

```bash
# Install dependencies
uv sync

# Run basic import test
uv run python -c "import sparse_ir; print('✓ Installation successful!')"

# Run Python tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_basis.py -v

# Run tests and show coverage
uv run pytest --cov=sparse_ir

# Install in development mode with force rebuild
uv sync --reinstall

# Run development environment setup
uv sync --group dev
```

## Architecture Overview

### Core Components

- **src/sparse_ir/__init__.py**: Main module with public API exports
- **src/sparse_ir/basis.py**: Basis functions and finite temperature basis classes
- **src/sparse_ir/sampling.py**: Sampling methods for tau and Matsubara frequencies
- **src/sparse_ir/kernel.py**: Kernel functions for different physics models
- **src/sparse_ir/sve.py**: Singular value expansion implementation
- **src/sparse_ir/dlr.py**: Discrete Lehmann Representation methods
- **src/sparse_ir/augment.py**: Augmentation methods for basis sets
- **src/sparse_ir/poly.py**: Polynomial interpolation utilities
- **src/sparse_ir/basis_set.py**: Basis set management and operations

### Package Structure

This is a pure Python implementation with optional acceleration via the `pylibsparseir` dependency:
1. Primary implementation uses numpy/scipy for numerical operations
2. Can optionally use `pylibsparseir` C++ bindings for performance-critical operations
3. Provides consistent API regardless of backend used
4. Supports both sparse-ir legacy interface and modern sparse_ir interface

### Testing Architecture

- **tests/test_basis.py**: Tests for basis functions and finite temperature basis classes  
- **tests/test_sampling.py**: Tests for tau and Matsubara frequency sampling methods
- **tests/test_core.py**: Tests for core functionality and numerical operations
- **tests/test_kernel.py**: Tests for kernel function implementations
- **tests/test_sve.py**: Tests for singular value expansion methods
- **tests/test_dlr.py**: Tests for Discrete Lehmann Representation functionality
- **tests/test_augment.py**: Tests for basis augmentation methods
- **tests/test_poly.py**: Tests for polynomial interpolation utilities
- **tests/test_basis_set.py**: Tests for basis set management operations
- **tests/test_advanced_features.py**: Tests for advanced features and edge cases
- **tests/test_sve_advanced.py**: Advanced SVE functionality tests
- **tests/test_sampling_advanced.py**: Advanced sampling method tests

Key test patterns:
- Roundtrip accuracy tests (evaluate → fit cycles should be near-perfect)
- Shape validation for all array operations
- Error handling for invalid inputs
- Default vs custom parameter testing

## Mission

Provide a comprehensive Python implementation of sparse intermediate representation methods for many-body physics calculations. The library offers both pure Python implementations and optional C++ acceleration through `pylibsparseir` bindings for optimal performance across different use cases.

## Common Issues

- **ModuleNotFoundError**: Run `uv sync --reinstall` or `uv sync --group dev`
- **Import errors**: Ensure Python >= 3.10 and dependencies are installed via `uv sync`
- **Missing pylibsparseir**: Check that `pylibsparseir>=0.1.0,<0.2.0` is properly installed
- **Test failures**: Run specific test files with `uv run pytest tests/test_<module>.py -v` for detailed output
- **Dependency conflicts**: Clear environment and reinstall with `uv sync --reinstall`

# Important Instruction Reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
