# libsparseir
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SpM-lab/libsparseir)
[![CMake on a single platform](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CI_cmake.yml)
[![Create Tag and Release](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml/badge.svg)](https://github.com/SpM-lab/libsparseir/actions/workflows/CreateTag.yml)

> [!WARNING]
> This C++ project is still under construction. Please use other repositories:
> - https://github.com/SpM-lab/sparse-ir
> - https://github.com/SpM-lab/SparseIR.jl
> - https://github.com/SpM-lab/sparse-ir-fortran

## Description

This C++ library provides routines for constructing and working with the intermediate representation of correlation functions. It provides:

- on-the-fly computation of basis functions for arbitrary cutoff Λ
- basis functions and singular values are accurate to full precision
- routines for sparse sampling

We use [tuwien-cms/libxprec](https://github.com/tuwien-cms/libxprec) as a double-double precision arithmetic library.


## Building and Installation

### Dependencies

- **CMake** (>= 3.10)
- **C++ compiler** with C++11 support
- **Fortran compiler** (optional, for Fortran bindings)

All other dependencies (including libxprec) are automatically downloaded and built during the build process using CMake's FetchContent feature. You do not need to install these manually.

### Using Build Scripts

Three build scripts are provided for easy building and installation:

1. **build_capi.sh**: Builds and installs only the C API
   ```bash
   ./build_capi.sh
   ```

2. **build_fortran.sh**: Builds and installs the C API and Fortran bindings
   ```bash
   ./build_fortran.sh
   ```

3. **build_with_tests.sh**: Builds everything including tests
   ```bash
   ./build_with_tests.sh
   # After testing, you can install with:
   cd build && cmake --install .
   ```

By default, all scripts will install to `$HOME/opt/libsparseir`. You can override this by setting the `CMAKE_INSTALL_PREFIX` environment variable:

```bash
CMAKE_INSTALL_PREFIX=/usr/local ./build_capi.sh
```

### Manual Build

If you prefer to build manually, you can use the following commands:

```bash
mkdir -p build
cd build
# For C API only
cmake .. -DSPARSEIR_BUILD_FORTRAN=OFF -DSPARSEIR_BUILD_TESTING=OFF
# For C API and Fortran bindings
cmake .. -DSPARSEIR_BUILD_FORTRAN=ON -DSPARSEIR_BUILD_TESTING=OFF
# For everything including tests
cmake .. -DSPARSEIR_BUILD_FORTRAN=ON -DSPARSEIR_BUILD_TESTING=ON

# Build
cmake --build .

# Install
cmake --install .
```

### Debug Build

To enable debug logging output, use the `SPARSEIR_DEBUG` CMake option:

```bash
cmake .. -DSPARSEIR_DEBUG=ON
```

This will enable detailed debug logging for various operations in the library, which can be helpful for troubleshooting and development.

### Quick Test Build

For a quick test build with all options enabled:

```sh
rm -rf ./build && cmake -S . -B ./build -DSPARSEIR_BUILD_TESTING=ON && cmake --build ./build -j && ./build/test/libsparseirtests
```

### Testing Fortran Bindings

After building with Fortran bindings enabled, you can run the Fortran test:

```bash
cd build
./test_kernel
```

## BLAS/LAPACK Support

The library can optionally use BLAS/LAPACK for improved performance. By default, BLAS/LAPACK usage is disabled for compatibility reasons to avoid linking issues. You can control this with CMake options:

```bash
# Disable BLAS/LAPACK (default, uses Eigen's internal implementations)
cmake .. -DSPARSEIR_USE_BLAS=OFF -DSPARSEIR_USE_LAPACKE=OFF

# Enable BLAS/LAPACK (may require additional system libraries)
cmake .. -DSPARSEIR_USE_BLAS=ON -DSPARSEIR_USE_LAPACKE=ON
```

Note: When enabling BLAS/LAPACK, ensure that the appropriate libraries (such as OpenBLAS, Intel MKL, or Apple Accelerate) are properly installed and linked.

## Generating documentation with Doxygen

Install `doxygen` and `graphviz`. Then, run the following command:

```bash
bash generate_docs.sh
```

This will create the `docs/html` directory. Open `docs/html/index.html` with your browser to see it.

# Sample code in C

Please refer [`./sample_c/README.md`](./sample_c/README.md) to learn more.

# For developers


## CI/CD

This project uses GitHub Actions for continuous integration and automated releases:

- **CI_cmake.yml**: Runs automated tests on every push and pull request to ensure code quality
- **CreateTag.yml**: Automatically creates tags and releases when version numbers are updated in `include/sparseir/version.h`

### Automated Release Process

The release process is fully automated:

1. Update version numbers in `include/sparseir/version.h` by modifying:
   - `SPARSEIR_VERSION_MAJOR`
   - `SPARSEIR_VERSION_MINOR`
   - `SPARSEIR_VERSION_PATCH`

2. Push changes to the main branch

3. The GitHub Action will automatically:
   - Extract the version from the header file
   - Check if a tag with that version already exists
   - Create a new tag and release if the version is new
   - Generate release notes automatically