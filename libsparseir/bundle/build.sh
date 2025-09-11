#!/bin/bash

# Exit on error
set -e

# Version settings
EIGEN3_VERSION="3.4.0"
XPREC_VERSION="0.7.0"

# Create deps directory
mkdir -p deps

# Download and extract Eigen3
if [ ! -d "deps/eigen3" ]; then
    echo "Downloading Eigen3 ${EIGEN3_VERSION}..."
    wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN3_VERSION}/eigen-${EIGEN3_VERSION}.tar.gz -O eigen3.tar.gz
    tar xzf eigen3.tar.gz -C deps
    mv deps/eigen-${EIGEN3_VERSION} deps/eigen3
    rm eigen3.tar.gz
fi

# Download and extract xprec
if [ ! -d "deps/xprec" ]; then
    echo "Downloading xprec ${XPREC_VERSION}..."
    wget -q https://github.com/tuwien-cms/libxprec/archive/refs/tags/v${XPREC_VERSION}.tar.gz -O xprec.tar.gz
    tar xzf xprec.tar.gz -C deps
    mv deps/libxprec-${XPREC_VERSION} deps/xprec
    rm xprec.tar.gz
fi

# Build
echo "Building tar.gz..."
make

echo "Done!" 
