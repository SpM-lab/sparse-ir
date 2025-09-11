# Fortran bindings for SparseIR

# Enable Fortran preprocessor
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

# Set Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

add_library(sparseir_fortran SHARED
    sparseir.f90
    sparseir_ext.F90
    _cbinding.inc
    _cbinding_public.inc
    ${TYPE_FILES}
)

# Add include directory
target_include_directories(sparseir_fortran PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Add interface include directory for installed module files
target_include_directories(sparseir_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran>
    $<INSTALL_INTERFACE:include/sparseir>
)

set_target_properties(sparseir_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    #Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(sparseir_fortran PUBLIC sparseir)

# Add alias
add_library(SparseIR::sparseir_fortran ALIAS sparseir_fortran)

# Install Fortran library
install(TARGETS sparseir_fortran
    EXPORT sparseirTargets
    LIBRARY DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${SPARSEIR_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${SPARSEIR_INSTALL_BINDIR}"
    COMPONENT sparseir
)

# Install Fortran module files
message(STATUS "SPARSEIR_INSTALL_LIBDIR: ${SPARSEIR_INSTALL_LIBDIR}")
message(STATUS "SPARSEIR_INSTALL_BINDIR: ${SPARSEIR_INSTALL_BINDIR}")
message(STATUS "SPARSEIR_INSTALL_INCLUDEDIR: ${SPARSEIR_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "Installing Fortran module files to ${SPARSEIR_INSTALL_INCLUDEDIR}/sparseir")
install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran/
    DESTINATION "${SPARSEIR_INSTALL_INCLUDEDIR}/sparseir"
    FILES_MATCHING PATTERN "*.mod"
)