################################################################################
# mlir-bignum-tests
#
# The mlir-bignum unit test project.
################################################################################

project(mlir-bignum-tests)

find_package(doctest REQUIRED)
include(doctest)

include(CTest)

add_executable(${PROJECT_NAME}
    Integer.cpp
    LLVM.cpp
    main.cpp
    Native.cpp
    Rational.cpp
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        MLIRBignum
)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        -fno-rtti
)

doctest_discover_tests(${PROJECT_NAME})
