cmake_minimum_required(VERSION 3.10)
project(second_order_perturbation_fort LANGUAGES Fortran C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Fortran standard
set(CMAKE_Fortran_STANDARD 90)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTING "Build tests" OFF)
option(USE_SYSTEM_LIBSPARSEIR "Use system-installed libsparseir" ON)

# Setup libsparseir dependency
if(USE_SYSTEM_LIBSPARSEIR)
    find_package(SparseIR REQUIRED)
    list(GET SPARSEIR_PREFIX 0 SPARSEIR_PREFIX)
else()
    # Fetch and build libsparseir from source
    include(FetchContent)

    FetchContent_Declare(
        libsparseir
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..
    )

    # Set options for libsparseir build
    set(SPARSEIR_BUILD_FORTRAN ON CACHE BOOL "Build Fortran bindings for libsparseir")
    set(SPARSEIR_BUILD_TESTING OFF CACHE BOOL "Disable libsparseir tests")

    FetchContent_MakeAvailable(libsparseir)
endif()

find_library(FFTW3_LIB fftw3)
find_path(FFTW3_INCLUDE_DIR fftw3.f)

# Create a simple Fortran test
add_executable(second_order_perturbation_fort second_order_perturbation_fort.f90)

# Link with libsparseir
target_link_libraries(second_order_perturbation_fort PRIVATE SparseIR::sparseir_fortran ${FFTW3_LIB})

target_include_directories(second_order_perturbation_fort PRIVATE
    ${FFTW3_INCLUDE_DIR}
)