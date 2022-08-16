#include "mlir/Extras/Bignum/Rational.h"

#include <doctest/doctest.h>

using namespace mlir::ext;

// clang-format off

//===----------------------------------------------------------------------===//
// Rational
//===----------------------------------------------------------------------===//

TEST_CASE("Rational::Rational()")
{
    Rational rational{};

    CHECK(rational.isZero());
    CHECK(rational.getNumerator().isZero());
    CHECK(rational.getDenominator().isOne());
}

TEST_CASE("Rational::Rational(Integer, Integer)")
{
    Rational r1(13, 13), r2(-7, 21), r3(-14, -2), r4(31, -97);

    CHECK(r1.getNumerator() == 1);
    CHECK(r1.getDenominator() == 1);
    CHECK(r2.getNumerator() == -1);
    CHECK(r2.getDenominator() == 3);
    CHECK(r3.getNumerator() == 7);
    CHECK(r3.getDenominator() == 1);
    CHECK(r4.getNumerator() == -31);
    CHECK(r4.getDenominator() == 97);
}

TEST_CASE("Rational::exp10(si_t)")
{
    CHECK(Rational::exp10(-4) == Rational(1, 10000));
    CHECK(Rational::exp10(21) == Integer::exp10(21));
}

TEST_CASE("swap(Rational &, Rational &)")
{
    Rational r1(1, 12), r2(21, 4);

    swap(r1, r2);

    CHECK(r1 == Rational(21, 4));
    CHECK(r2 == Rational(1, 12));
}

TEST_CASE("Rational::asInteger()")
{
    SUBCASE("fraction")
    {
        CHECK(!Rational(1, 12).asInteger().has_value());
    }
    SUBCASE("integer")
    {
        CHECK(Rational().asInteger() == 0);
        CHECK(Rational(12, 4).asInteger() == 3);
    }
}

TEST_CASE("Rational::isProper()")
{
    CHECK(!Rational().isProper());
    CHECK(Rational(3, 4).isProper());
    CHECK(!Rational(4, 3).isProper());
    CHECK(Rational(-4, 7).isProper());
    CHECK(!Rational(-7, 4).isProper());
}

TEST_CASE("Rational::operator<=>()")
{
    SUBCASE("Native")
    {
        CHECK(Rational() == 0);
        CHECK(Rational(-4) == -4);
        CHECK(Rational(12, 7) > 1);
        CHECK(Rational(12, -7) > -2);
    }
    SUBCASE("Integer")
    {
        CHECK(Rational() == Integer());
        CHECK(Rational(IntegerRef({1, 2}, 1)) == Integer(IntegerRef({1, 2}, 1)));
        CHECK(Rational(IntegerRef({1, 2}, 1), 4) < Integer(IntegerRef({1, 2}, 1)));
    }
    SUBCASE("Rational")
    {
        CHECK(Rational() == Rational(0));
        CHECK(Rational(21, 3) == Rational(14, 2));
        CHECK(Rational(-6, 8) < Rational(-6, 9));
    }
}

TEST_CASE("Rational::parse(std::string &, unsigned)")
{
    CHECK(Rational::parse("0") == 0);
    CHECK(Rational::parse("21:3") == 7);
    CHECK(Rational::parse("7:-12") == Rational(-7, 12));
}

TEST_CASE("Rational::toString(...)")
{
    const auto stringify = [](const Rational &rat) {
        llvm::SmallString<32> buffer;
        rat.toString(buffer);
        return static_cast<std::string>(buffer);
    };

    CHECK(stringify(Rational()) == "0");
    CHECK(stringify(Rational(-4)) == "-4");
    CHECK(stringify(Rational(7, -12)) == "-7:12");
    CHECK(stringify(Rational(89, 48)) == "89:48");
}

TEST_CASE("Rational::round(RoundingMode)")
{
    //          -6/5 -2/3 -1/2 -1/3  0  1/3 1/2 2/3 6/5
    // Trunc      -1    0    0   0   0   0   0   0   1
    // Floor      -2   -1   -1  -1   0   0   0   0   1
    // Ceil       -1    0    0   0   0   1   1   1   2
    // Nearest    -1   -1   -1   0   0   0   1   1   1

    Rational r1(0), r2(1,3), r3(1,2), r4(2,3), r5(6,5);
    const std::array args =
            {-r5, -r4, -r3, -r2,  r1,  r2,  r3,  r4,  r5};

    SUBCASE("Trunc")
    {
        const std::array expect =
            { -1,   0,   0,   0,   0,   0,   0,   0,   1};
        for (std::size_t i = 0; i < args.size(); ++i)
            CHECK(args[i].round(Rational::RoundingMode::Trunc) == expect[i]);
    }
    SUBCASE("Floor")
    {
        const std::array expect =
            { -2,  -1,  -1,  -1,   0,   0,   0,   0,   1};
        for (std::size_t i = 0; i < args.size(); ++i)
            CHECK(args[i].round(Rational::RoundingMode::Floor) == expect[i]);
    }
    SUBCASE("Ceil")
    {
        const std::array expect =
            { -1,   0,   0,   0,   0,   1,   1,   1,   2};
        for (std::size_t i = 0; i < args.size(); ++i)
            CHECK(args[i].round(Rational::RoundingMode::Ceil) == expect[i]);
    }
    SUBCASE("Nearest")
    {
        const std::array expect =
            { -1,  -1,  -1,   0,   0,   0,   1,   1,   1};
        for (std::size_t i = 0; i < args.size(); ++i)
            CHECK(args[i].round(Rational::RoundingMode::Nearest) == expect[i]);
    }
}

TEST_CASE("Rational::invert()")
{
    CHECK(~Rational() == Rational());
    CHECK(~Rational(7, 21) == 3);
    CHECK(~Rational(-4, 3) == Rational(-3, 4));
}

TEST_CASE("Rational::operator+=()")
{
    Rational r1(-17, 24), r2(41, 16);

    r1 += r2;
    CHECK(r1 == Rational(89, 48));
}

TEST_CASE("Rational::operator*=()")
{
    Rational r1(-4, 3), r2(11, 8);

    r1 *= r2;
    CHECK(r1 == Rational(-11, 6));
}
