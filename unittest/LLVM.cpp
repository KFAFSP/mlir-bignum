#include "mlir/Extras/Bignum/LLVM.h"

#include <doctest/doctest.h>

using namespace llvm;
using namespace mlir::ext;

// clang-format off

TEST_CASE("signum(APSInt)")
{
    SUBCASE("zero")
    {
        CHECK(signum(APSInt()) == 0);
        CHECK(signum(APSInt(APInt(64, 0))) == 0);
    }
    SUBCASE("negative")
    {
        CHECK(signum(APSInt(APInt(64, -1, true), false)) == -1);
        CHECK(signum(APSInt(APInt(64, -2, true), false)) == -1);
        CHECK(signum(APSInt(APInt(64, min_si, true), false)) == -1);
    }
    SUBCASE("positive")
    {
        CHECK(signum(APSInt(APInt(64, 2, true), false)) == 1);
        CHECK(signum(APSInt(APInt(64, 2, true), false)) == 1);
        CHECK(signum(APSInt(APInt(64, max_si, true), false)) == 1);
        CHECK(signum(APSInt(APInt(64, max_ui, true), true)) == 1);
    }
}

TEST_CASE("signum(APInt)")
{
    SUBCASE("zero")
    {
        CHECK(signum(APInt()) == 0);
        CHECK(signum(APInt(64, 0)) == 0);
    }
    SUBCASE("negative")
    {
        CHECK(signum(APInt(64, -1, true)) == -1);
        CHECK(signum(APInt(64, -2, true)) == -1);
        CHECK(signum(APInt(64, min_si, true)) == -1);
    }
    SUBCASE("positive")
    {
        CHECK(signum(APInt(64, 2, true)) == 1);
        CHECK(signum(APInt(64, 2, true)) == 1);
        CHECK(signum(APInt(64, max_si, true)) == 1);
    }
}
