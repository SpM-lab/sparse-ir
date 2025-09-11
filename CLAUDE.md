# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pysparseir** is a Python wrapper for the libsparseir C++ library, providing sparse intermediate representation (IR) methods for many-body physics calculations. This is part of the broader **spm-lab** ecosystem that includes:

- **libsparseir**: High-performance C++ core library with C API (embedded as git submodule)
- **LibSparseIR.jl**: Julia wrapper (separate repository)
- **pysparseir**: This Python wrapper

## Development Commands

### Core Development Workflow

```bash
# Install dependencies and build C++ library
uv sync

# Run basic import test
uv run python -c "import pylibsparseir; print('✓ Installation successful!')"

# Run Python tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_basis.py -v

# Run tests and show coverage
uv run pytest --cov=pylibsparseir

# Install in development mode with force rebuild
uv sync --reinstall

# Quick functionality test
uv run python test_compare.py
```

### C++ Library Build (libsparseir submodule)

```bash
# Build C API only (default)
cd libsparseir && ./build_capi.sh

# Build with Fortran bindings
cd libsparseir && ./build_fortran.sh  

# Build with tests
cd libsparseir && ./build_with_tests.sh

# Manual build with all options
cd libsparseir && mkdir -p build && cd build
cmake .. -DSPARSEIR_BUILD_FORTRAN=ON -DSPARSEIR_BUILD_TESTING=ON
cmake --build . -j
ctest --output-on-failure
cmake --install .
```

## Architecture Overview

### Core Components

- **src/pylibsparseir/core.py**: Main Python wrapper functions that interface with C library
- **src/pylibsparseir/ctypes_wrapper.py**: ctypes type definitions for C interop
- **src/pylibsparseir/constants.py**: Constants and enums
- **setup.py**: Custom CMakeBuild class that builds the C++ submodule during Python package installation

### Library Loading Architecture

The Python package uses a multi-step library discovery process:
1. Searches for platform-specific shared library (`libsparseir.dylib` on macOS, `.so` on Linux, `.dll` on Windows)
2. Searches in: package directory, `../build`, `../../build`
3. Sets up ctypes function prototypes for all C API functions
4. Provides Pythonic wrapper functions with error handling

### Key API Pattern

All wrapper functions follow this pattern:
```python
def function_name(args):
    status = c_int()
    result = _lib.spir_function_name(args, byref(status))
    if status.value != COMPUTATION_SUCCESS:
        raise RuntimeError(f"Function failed: {status.value}")
    return result
```

### Testing Architecture

- **tests/test_basis.py**: Tests for `FiniteTempBasis` class and basis function evaluation
- **tests/test_sampling.py**: Tests for `TauSampling` and `MatsubaraSampling` classes
- **tests/test_core.py**: Tests for low-level C API wrapper functions
- **tests/c_api/**: Comprehensive C API interface tests (corresponding to LibSparseIR.jl/test/C_API/)
  - **core_tests.py**: Core C API functionality tests (kernels, SVE, basis functions)
  - **sampling_tests.py**: C API sampling functionality tests (tau/Matsubara sampling)
  - **dlr_tests.py**: C API DLR (Discrete Lehmann Representation) tests
  - **integration_tests.py**: C API integration workflow tests
  - **comprehensive_tests.py**: Comprehensive C API validation tests
- **test_compare.py**: Comparison test with sparse-ir reference implementation

Key test patterns:
- Roundtrip accuracy tests (evaluate → fit cycles should be near-perfect)
- Shape validation for all array operations
- Error handling for invalid inputs
- Default vs custom parameter testing

## Mission

The goal is to recreate the functionality of the pure Python `sparse-ir` library using the high-performance C++ `libsparseir` backend, similar to how `LibSparseIR.jl` provides a Julia interface to the same C++ library.

## Common Issues

- **CMake Error**: Run `git submodule init && git submodule update`
- **ModuleNotFoundError**: Run `uv sync --reinstall`  
- **Symbol not found**: Verify shared libraries exist in `src/pylibsparseir/`
- **Import errors**: Ensure Python >= 3.12 and numpy is installed