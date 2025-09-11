cmake_minimum_required(VERSION 3.10)
project(SparseIRSamples LANGUAGES C)

# Find SparseIR package
find_package(SparseIR REQUIRED)

# Get all .c files in test_files directory and sort them
file(GLOB TEST_SOURCES "*.c")
list(SORT TEST_SOURCES)

# Create executable for each test file
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE SparseIR::sparseir m)
    list(APPEND TEST_NAMES ${TEST_NAME})
endforeach()

# Create a custom target to run all tests
add_custom_target(test
    DEPENDS ${TEST_NAMES}
    COMMENT "Running all test samples"
)

# Add initial message
add_custom_command(TARGET test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Running all tests..."
    COMMAND ${CMAKE_COMMAND} -E echo "----------------------------------------"
)

# Add commands for each test in sorted order
foreach(TEST_NAME ${TEST_NAMES})
    add_custom_command(TARGET test
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Running ${TEST_NAME}..."
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo "----------------------------------------"
    )
endforeach()

# Add final message
add_custom_command(TARGET test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "All tests completed!"
) 
