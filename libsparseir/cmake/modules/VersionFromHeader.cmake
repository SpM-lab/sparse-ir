# Utility for extracting version from C header
#
# Copyright (C) 2023 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.9)

# Function to extract version string from header file
#
#     version_from_header(
#           version_variable
#           HEADER header_file
#           MACROS macro_name_major [macro_name_minor ...]
#           )
#
# Writes into `version_variable` the version detected from a header file
# named `header_file`. The code looks for preprecessor definitions of the form
#
#     #define MACRO_NAME DIGITS
#
# where MACRO_NAME is each of the arguments of MACROS in sequence. The code
# then strings together the values, separated by "."
function(version_from_header version_arg)
    # parse arguments
    set(options)
    set(one_value_args HEADER)
    set(multi_value_args MACROS)
    cmake_parse_arguments(_ARG "${options}" "${one_value_args}"
                          "${multi_value_args}" ${ARGN})

    # These are required
    if (NOT _ARG_HEADER)
        message(FATAL_ERROR "HEADER is required")
    endif()
    if (NOT _ARG_MACROS)
        message(FATAL_ERROR "MACROS is required")
    endif()

    # read in the file
    file(READ "${_ARG_HEADER}" header)

    # build up version string from macros
    set(version "")
    foreach(macro_name IN LISTS _ARG_MACROS)
        if (header MATCHES "#[ \t]*define[ \t]+${macro_name}[ \t]+([0-9]+)")
            set(chunk "${CMAKE_MATCH_1}")
            if (version STREQUAL "")
                set(version "${chunk}")
            else()
                set(version "${version}.${chunk}")
            endif()
        else()
            message(SEND_ERROR "${macro_name} not #defined in ${_ARG_HEADER}")
            return()
        endif()
    endforeach()

    # export back to parent scope ("pass by reference")
    set("${version_arg}" "${version}" PARENT_SCOPE)
endfunction()
