#[========================================================================[.rst:
LibUtils
--------

Provides utility functions for dealing with library imports.

#]========================================================================]

#[========================================================================[.rst:
.. command:: evp_detect_library_version

Attempts to detect the version of a library from the name of its shared library.

It has the following arguments:

``LIBRARY``
    Path to the library file.
``RESULT_VARIABLE``
    Name of the output variable.

And sets the following result variables:

``${RESULT_VARIABLE}_MAJOR``
    The major version (``X`` in ``X.Y.Z``).
``${RESULT_VARIABLE}_MINOR``
    The minor version (``Y`` in ``X.Y.Z``).
``${RESULT_VARIABLE}_PATCH``
    The subminor version (``Z`` in ``X.Y.Z``).
``${RESULT_VARIABLE}``
    The version string in ``X.Y.Z`` format. If the version could not be
    inferred, this variable is set to ``NOTFOUND``.

#]========================================================================]
function(evp_detect_library_version LIBRARY RESULT_VARIABLE)
    # Get the parent folder of the library file.
    get_filename_component(LIBRARY_DIR "${LIBRARY}" DIRECTORY)
    # Get the name of the library file.
    get_filename_component(LIBRARY_NAME "${LIBRARY}" NAME_WE)

    # - List all files in the same folder ...
    # - ... that are shared libraries of the same name ...
    # - ... and return only the lexicographically last match.
    execute_process(
        COMMAND
            bash "-c" "ls ${LIBRARY_DIR} | grep -E \"^${LIBRARY_NAME}\\.so\\.[0-9\\.]*$\" | tail -1"
        RESULT_VARIABLE     FOUND_LIBSO
        OUTPUT_VARIABLE     LIBSO_NAME
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set("${RESULT_VARIABLE}_MAJOR" "NOTFOUND" PARENT_SCOPE)
    set("${RESULT_VARIABLE}_MINOR" "NOTFOUND" PARENT_SCOPE)
    set("${RESULT_VARIABLE}_PATCH" "NOTFOUND" PARENT_SCOPE)
    set("${RESULT_VARIABLE}" "NOTFOUND" PARENT_SCOPE)

    if(FOUND_LIBSO EQUAL 0)
        # Split the version string (may fail).
        string(REGEX MATCH "\\.so\\.([0-9]*)\\.([0-9]*)\\.([0-9]*)$" _MATCH "${LIBSO_NAME}")
        if (_MATCH)
            set("${RESULT_VARIABLE}_MAJOR" "${CMAKE_MATCH_1}" PARENT_SCOPE)
            set("${RESULT_VARIABLE}_MINOR" "${CMAKE_MATCH_2}" PARENT_SCOPE)
            set("${RESULT_VARIABLE}_PATCH" "${CMAKE_MATCH_3}" PARENT_SCOPE)
            set("${RESULT_VARIABLE}" "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

#[========================================================================[.rst:
.. command:: evp_get_library_undefined_symbols

Gets the undefined symbols in a library.

It has the following arguments:

``LIBRARY``
    Path to the library file.
``RESULT_VARIABLE``
    Name of the output variable.

And sets the following result variables:

``${RESULT_VARIABLE}``
    A list of undefined symbol names as (mangled) identifiers.

#]========================================================================]
function(evp_get_library_undefined_symbols LIBRARY RESULT_VARIABLE)
    set("${RESULT_VARIABLE}" "NOTFOUND" PARENT_SCOPE)

    # Check library type.
    if(LIBRARY MATCHES "^[^\\.]+\\.so")
        set(OPTIONS "-Du")
    elseif(LIBRARY MATCHES "^[^\\.]+\\.a")
        set(OPTIONS "-u")
    else()
        return()
    endif()

    # List all undefined symbols.
    execute_process(
        COMMAND
            bash "-c" "nm ${OPTIONS} ${LIBRARY}"
        RESULT_VARIABLE     RESULT
        OUTPUT_VARIABLE     SYMBOLS
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(RESULT EQUAL 0)
        # Post-process the list
        STRING(REGEX REPLACE "[^\\n]*U ([^\\n]+)\\n" "\\1;" SYMBOLS "${SYMBOLS}")
        set("${RESULT_VARIABLE}" "${SYMBOLS}" PARENT_SCOPE)
    endif()
endfunction()
