#include "mlir/Extras/Bignum/Native.h"

#include <doctest/doctest.h>

using namespace mlir::ext;

// clang-format off

TEST_CASE("signum(SInt)")
{
    SUBCASE("zero")
    {
        CHECK(signum(si_t(0)) == 0);
    }
    SUBCASE("negative")
    {
        CHECK(signum(si_t(-1)) == -1);
        CHECK(signum(si_t(-2)) == -1);
        CHECK(signum(min_si) == -1);
    }
    SUBCASE("positive")
    {
        CHECK(signum(si_t(1)) == 1);
        CHECK(signum(si_t(2)) == 1);
        CHECK(signum(max_si) == 1);
    }
}

TEST_CASE("signum(UInt)")
{
    SUBCASE("zero")
    {
        CHECK(signum(ui_t(0)) == 0);
    }
    SUBCASE("positive")
    {
        CHECK(signum(ui_t(1)) == 1);
        CHECK(signum(ui_t(2)) == 1);
        CHECK(signum(max_ui) == 1);
        CHECK(signum(ui_t(-1)) == 1);
    }
}

TEST_CASE("magnitude(SInt)")
{
    SUBCASE("zero")
    {
        CHECK(magnitude(si_t(0)) == 0);
    }
    SUBCASE("negative")
    {
        CHECK(magnitude(si_t(-1)) == 1);
        CHECK(magnitude(si_t(-2)) == 2);
        CHECK(magnitude(min_si) == ui_t(-min_si));
    }
    SUBCASE("positive")
    {
        CHECK(magnitude(si_t(1)) == 1);
        CHECK(magnitude(si_t(2)) == 2);
        CHECK(magnitude(max_si) == ui_t(max_si));
    }
}

TEST_CASE("magnitude(UInt)")
{
    SUBCASE("zero")
    {
        CHECK(magnitude(ui_t(0)) == 0);
    }
    SUBCASE("positive")
    {
        CHECK(magnitude(ui_t(1)) == 1);
        CHECK(magnitude(ui_t(2)) == 2);
        CHECK(magnitude(max_ui) == max_ui);
        CHECK(magnitude(ui_t(-1)) == max_ui);
    }
}
