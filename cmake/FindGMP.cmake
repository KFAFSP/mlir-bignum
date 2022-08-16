#[========================================================================[.rst:
FindGMP
-------

Finds and provides the GNU multiprecision arithmetic library.

Supports the following signature::

    find_package(GMP
        [version] [EXACT]       # Minimum or EXACT version e.g. 10.4
        [REQUIRED]              # Fail with error if GMP is not found
    )

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``GMP::GMP``
    The GMP library.

Result Variables
^^^^^^^^^^^^^^^^

``GMP_FOUND``
    True if the system has the GMP library.
``GMP_VERSION``
    The version of the GMP library which was found.
``GMP_INCLUDE_DIRS``
    Include directories needed to use GMP.
``GMP_LIBRARIES``
    Libraries needed to link to GMP.

Cache variables
^^^^^^^^^^^^^^^

``GMP_LIBRARY``
    Path to the GMP library.
``GMP_INCLUDE_DIR``
    Path to the GMP include directory.

Hints
^^^^^

``GMP_ROOT``
    Path to a GMP installation or build.
``GMP_USE_STATIC_LIBS``
    If set to ``ON``, only static library files will be accepted, otherwise
    shared libraries are preferred. (Defaults to ``OFF``.)

#]========================================================================]

### Step 1: Find the library and include path. ###

# Allows the user to specify a custom search path.
set(GMP_ROOT "" CACHE PATH "Path to a GMP installation or build.")

if(GMP_USE_STATIC_LIBS)
    # Static-only find hack.
    set(_gmp_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
endif()

find_library(GMP_LIBRARY
    NAMES
        gmp
    HINTS
        "${GMP_ROOT}/lib/"
    DOC "Path to the GMP library."
)
find_path(GMP_INCLUDE_DIR
    NAMES
        gmp.h
    HINTS
        "${GMP_ROOT}/include/"
    DOC "Path to the GMP include directory."
)

if(GMP_USE_STATIC_LIBS)
    # Undo our static-only find hack.
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_gmp_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

mark_as_advanced(
    GMP_ROOT
    GMP_LIBRARY
    GMP_INCLUDE_DIR
)

### Step 2: Examine what we found. ###

include(LibUtils)

if(EXISTS "${GMP_LIBRARY}")
    # A local install or build was found, and must be examined for the version.
    evp_detect_library_version("${GMP_LIBRARY}" GMP_VERSION)
endif()

# Run the standard handler to process all variables.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP
    REQUIRED_VARS
        GMP_INCLUDE_DIR
        GMP_LIBRARY
    VERSION_VAR
        GMP_VERSION
)
if(NOT GMP_FOUND)
    # Optional dependency not fulfilled.
    return()
endif()

### Step 3: Declare targets and macros. ###

# Set legacy result variables.
set(GMP_INCLUDE_DIRS "${GMP_INCLUDE_DIR}")
set(GMP_LIBRARIES "${GMP_LIBRARY}")

# Create imported target if it does not exist.
if(NOT TARGET GMP::GMP)
    add_library(GMP::GMP UNKNOWN IMPORTED)

    set_target_properties(GMP::GMP PROPERTIES
        VERSION                         "${GMP_VERSION}"
        IMPORTED_LOCATION               "${GMP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES   "${GMP_INCLUDE_DIR}"
    )
endif()
