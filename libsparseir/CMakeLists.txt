cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Provides CMAKE_INSTALL_* directories
include(GNUInstallDirs)

# extract version information from header file
if (NOT SPARSEIR_VERSION)
    include("VersionFromHeader")
    version_from_header(SPARSEIR_VERSION
        HEADER "include/sparseir/version.h"
        MACROS SPARSEIR_VERSION_MAJOR SPARSEIR_VERSION_MINOR SPARSEIR_VERSION_PATCH
        )
    message(STATUS "Extracted package version: ${SPARSEIR_VERSION}")

    set(SPARSEIR_VERSION "${SPARSEIR_VERSION}"
        CACHE STRING "version of the sparseir package")
    mark_as_advanced(SPARSEIR_VERSION)
endif()

# Configure project
project(
    SparseIR
    LANGUAGES CXX
    VERSION "${SPARSEIR_VERSION}"
    DESCRIPTION "SparseIR library"
    )

# Set installation directories
set(SPARSEIR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "directory into which to install sparseir include files" FORCE)
set(SPARSEIR_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "directory into which to install sparseir library files" FORCE)

# Add Fortran language support if building Fortran bindings
option(SPARSEIR_BUILD_FORTRAN "Build Fortran bindings" OFF)
if(SPARSEIR_BUILD_FORTRAN)
    enable_language(Fortran)
    message(STATUS "Fortran support enabled for building Fortran bindings")
endif()

# Eigen3
# BLAS/LAPACK options
option(SPARSEIR_USE_BLAS "Enable BLAS support" OFF)
option(SPARSEIR_USE_LAPACKE "Enable LAPACK support" OFF)

# BLAS/LAPACK search and configuration
if(SPARSEIR_USE_BLAS OR SPARSEIR_USE_LAPACKE)
    # Disable Eigen's BLAS/LAPACK usage for compatibility
    add_compile_definitions(EIGEN_DONT_USE_BLAS_LAPACK)
    message(STATUS "BLAS/LAPACK support disabled for compatibility")
endif()

# macOS-specific configuration
if(APPLE)
    # Explicitly add Accelerate Framework
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        list(APPEND BLAS_LIBRARIES ${ACCELERATE_FRAMEWORK})
        list(APPEND LAPACK_LIBRARIES ${ACCELERATE_FRAMEWORK})
    endif()

    # Specify LAPACKE header location
    find_path(LAPACKE_INCLUDE_DIR lapacke.h
        PATHS /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers
        /usr/local/include
        /opt/homebrew/include
    )
    if(LAPACKE_INCLUDE_DIR)
        include_directories(${LAPACKE_INCLUDE_DIR})
    endif()
endif()

set(EIGEN3_REQUIRED_VERSION "3.4.0")
find_package (Eigen3 ${EIGEN3_REQUIRED_VERSION} QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found in system, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG ${EIGEN3_REQUIRED_VERSION}
    )
    FetchContent_MakeAvailable(Eigen3)
    message(STATUS "Eigen3 fetched from GitHub")
else()
    message(STATUS "Eigen3 found in system: ${Eigen3_VERSION}")
endif()

find_package(Threads REQUIRED)

# libxprec - built only as a static dependency, not for installation
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG mainline
)

# Extract the content but don't process it through add_subdirectory
FetchContent_GetProperties(XPrec)
if(NOT xprec_POPULATED)
    FetchContent_Populate(XPrec)
    message(STATUS "XPrec source directory: ${xprec_SOURCE_DIR}")
endif()

# ---------------------------------
# Building

# Enable debug logging
option(SPARSEIR_DEBUG "Enable debug logging output" OFF)
if(SPARSEIR_DEBUG)
    add_definitions(-DDEBUG_SPIR)
    message(STATUS "Debug logging enabled")
endif()

# Set the default component name for installations
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sparseir)

add_library(sparseir SHARED
    src/utils.cpp
    src/linalg.cpp
    src/root.cpp
    src/specfuncs.cpp
    src/svd.cpp
    src/sve.cpp
    src/poly.cpp
    src/kernel.cpp
    src/cinterface.cpp
    )
if(NOT MSVC)
    target_compile_options(sparseir PRIVATE -Wall -Wextra -pedantic)
endif()

target_include_directories(sparseir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${xprec_SOURCE_DIR}/include>
)
set_target_properties(sparseir PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    )

target_link_libraries(sparseir PRIVATE Eigen3::Eigen Threads::Threads)

# Add BLAS/LAPACK linking
if(SPARSEIR_USE_BLAS AND BLAS_FOUND)
    target_link_libraries(sparseir PRIVATE ${BLAS_LIBRARIES})
endif()

if(SPARSEIR_USE_LAPACKE AND LAPACK_FOUND)
    target_link_libraries(sparseir PRIVATE ${LAPACK_LIBRARIES})
endif()

# macOS-specific configuration
if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(sparseir PRIVATE ${ACCELERATE_FRAMEWORK})
endif()

# Use library convention.
add_library(SparseIR::sparseir ALIAS sparseir)

# Build Fortran bindings if enabled
if(SPARSEIR_BUILD_FORTRAN)
    add_subdirectory(fortran)
endif()

# Add include paths from parent scope if available

# -------------------------------------
# Testing

option(SPARSEIR_BUILD_TESTING "Enable creation of SparseIR tests." OFF)

if (SPARSEIR_BUILD_TESTING)
    enable_testing()

    # Download and configure Catch2 for testing
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)

    # Add test directory
    add_subdirectory("test")
endif()

# -------------------------------------
# Installation

# Install only the C API library and headers
install(TARGETS sparseir
    EXPORT sparseirTargets
    LIBRARY DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${SPARSEIR_INSTALL_BINDIR}"
    COMPONENT sparseir
)

# Install only necessary header files (C API)
install(FILES
    include/sparseir/sparseir.h
    include/sparseir/version.h
    include/sparseir/spir_status.h
    DESTINATION "${SPARSEIR_INSTALL_INCLUDEDIR}/sparseir"
    COMPONENT sparseir
)

# Install CMake config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT sparseirTargets
    FILE SparseIRTargets.cmake
    NAMESPACE SparseIR::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sparseirConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)

# Configure and install Fortran config files
#if(SPARSEIR_BUILD_FORTRAN)
    #configure_package_config_file(
        #"${CMAKE_CURRENT_SOURCE_DIR}/cmake/sparseir-fortran-config.cmake.in"
        #"${CMAKE_CURRENT_BINARY_DIR}/sparseir-fortran-config.cmake"
        #INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
    #)
#
    #install(FILES
        #"${CMAKE_CURRENT_BINARY_DIR}/sparseir-fortran-config.cmake"
        #DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
    #)
#endif()

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIR"
)
