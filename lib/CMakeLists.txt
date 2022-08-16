add_mlir_library(MLIRBignum
        ABI.cpp
        Integer.cpp
        Rational.cpp

    EXCLUDE_FROM_LIBMLIR
    ENABLE_AGGREGATION

    LINK_LIBS PRIVATE
        GMP::GMP

    LINK_LIBS PUBLIC
        MLIRParser
)