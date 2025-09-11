set(CMAKE_CXX_STANDARD 11)

# Set include paths for tests
set(TEST_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${xprec_SOURCE_DIR}/include
)

# List of all test files
file(GLOB TEST_FILES "*.cxx" "*.hpp")

# Create a single combined test executable
add_executable(libsparseirtests ${TEST_FILES})
target_link_libraries(libsparseirtests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(libsparseirtests PRIVATE SparseIR::sparseir)
target_link_libraries(libsparseirtests PRIVATE Eigen3::Eigen Threads::Threads)

# Add include paths
if(DEFINED TEST_INCLUDE_DIRS)
    target_include_directories(libsparseirtests PRIVATE ${TEST_INCLUDE_DIRS})
endif()

# Also explicitly include Eigen3 and XPrec include directories
if(DEFINED EIGEN3_INCLUDE_DIR)
    target_include_directories(libsparseirtests PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(TARGET XPrec::xprec)
    get_target_property(XPREC_INCLUDE_DIRS XPrec::xprec INTERFACE_INCLUDE_DIRECTORIES)
    if(XPREC_INCLUDE_DIRS)
        target_include_directories(libsparseirtests PRIVATE ${XPREC_INCLUDE_DIRS})
    endif()
endif()

add_test(NAME combined_tests COMMAND libsparseirtests -d yes) 