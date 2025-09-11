#!/bin/bash
set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Clean build directory to ensure clean state
rm -rf build
mkdir -p build
cd build

# Configure with minimal options (only C-API)
cmake .. \
  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-$HOME/opt/libsparseir} \
  -DSPARSEIR_BUILD_FORTRAN=ON \
  -DSPARSEIR_BUILD_TESTING=OFF \
  -DBUILD_TESTING=OFF

# Build and install
cmake --build . --config Release
cmake --install .

echo "SparseIR C-API library has been built and installed successfully." 
