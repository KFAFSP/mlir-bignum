#[========================================================================[.rst:
GitRevision
-----------

Populates some CMake variables with information about the current revision in
the parent Git repository.

Result Variables
^^^^^^^^^^^^^^^^

``GIT_FOUND``
    Whether Git was found on this system.
``GIT_HEAD``
    The commit hash of the HEAD revision.
``GIT_DESCRIBE``
    The description (including tag if present) of the HEAD revision.
``GIT_DIRTY``
    A boolean indicating whether the working tree is dirty.

#]========================================================================]

find_package(Git QUIET)
if(NOT GIT_FOUND)
    # No git present.
    set(GIT_HEAD "-NOTFOUND")
    set(GIT_DESCRIBE "-NOTFOUND")
    set(GIT_DIRTY "-NOTFOUND")
    set(GIT_DESCRIBE_DIRTY "-NOTFOUND")
    return()
endif()

execute_process(
    COMMAND             ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE     _git_HEAD_RESULT
    OUTPUT_VARIABLE     GIT_HEAD
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND             ${GIT_EXECUTABLE} describe --always --long
    WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE     _git_DESCRIBE_RESULT
    OUTPUT_VARIABLE     GIT_DESCRIBE
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND             ${GIT_EXECUTABLE} diff-index --quiet HEAD --
    WORKING_DIRECTORY   ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE     GIT_DIRTY
    OUTPUT_QUIET
    ERROR_QUIET
)

if(_git_HEAD_RESULT OR _git_DESCRIBE_RESULT)
    # Not a git repository or no commits.
    set(GIT_HEAD "NOTFOUND")
    set(GIT_DESCRIBE "NOTFOUND")
    set(GIT_DIRTY "NOTFOUND")
else()
    if(GIT_DIRTY)
        set(GIT_DESCRIBE_DIRTY "${GIT_DESCRIBE}-dirty")
    else()
        set(GIT_DESCRIBE_DIRTY "${GIT_DESCRIBE}")
    endif()
    message("Git revision: ${GIT_DESCRIBE_DIRTY}")
endif()
