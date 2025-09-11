add_subdirectory(cpp)

if (SPARSEIR_BUILD_FORTRAN)
    add_subdirectory(fortran)
endif()

