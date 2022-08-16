/// Implements the multiprecision rational number type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/Extras/Bignum/Rational.h"

#include <gmp.h>
#include <llvm/Support/Debug.h>

using namespace llvm;
using namespace mlir::ext;

#define DEBUG_TYPE "Rational"

//===----------------------------------------------------------------------===//
// GMP aliasing
//===----------------------------------------------------------------------===//

/// Obtains an mpz_ptr from an ABI compatible Integer.
[[nodiscard]] static inline mpz_ptr get_impl(Integer &impl)
{
    return reinterpret_cast<mpz_ptr>(&impl);
}
/// Obtains an mpz_srcptr from an ABI compatible immutable Integer.
[[nodiscard]] static inline mpz_srcptr get_impl(const Integer &impl)
{
    return reinterpret_cast<mpz_srcptr>(&impl);
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Aligns two fractions to a common denominator.
///
/// @returns    The value @p den1 needs to be multiplied with to get the common
///             denominator.
static Integer
align(Integer &num1, const Integer &den1, Integer &num2, const Integer &den2)
{
    if (den1 == den2) return Integer(static_cast<ui_t>(1));

    Integer t1, t2 = gcd(den1, den2);

    mpz_divexact(get_impl(t1), get_impl(den1), get_impl(t2));
    mpz_divexact(get_impl(t2), get_impl(den2), get_impl(t2));

    mpz_mul(get_impl(num1), get_impl(num1), get_impl(t2));
    mpz_mul(get_impl(num2), get_impl(num2), get_impl(t1));

    return t2;
}

/// Compares two fractions.
static std::strong_ordering compare(
    const Integer &num1,
    const Integer &den1,
    const Integer &num2,
    const Integer &den2)
{
    // Compare numerators if both are integer.
    if (den1.isOne() && den2.isOne()) return num1 <=> num2;

    // Compare cross-multiplied sizes.
    const auto sz1 = num1.size() + den2.size();
    const auto sz2 = num2.size() + den1.size();
    if (sz1 > sz2 + 1) return signum(num1) <=> 0;
    if (sz2 > sz1 + 1) return -signum(num1) <=> 0;

    // Cross-multiply and compare.
    Integer crs1(num1), crs2(num2);
    std::ignore = align(crs1, den1, crs2, den2);
    return crs1 <=> crs2;
}

/// Adds or subtracts two fractions.
template<bool Sub>
static void
add_sub(Integer &num1, Integer &den1, Integer num2, const Integer &den2)
{
    den1 *= align(num1, den1, num2, den2);
    if constexpr (Sub)
        num1 -= num2;
    else
        num1 += num2;
}

/// Multiplies or divides two fractions.
static void mul_div(Integer &num1, Integer &den1, Integer num2, Integer den2)
{
    const auto gcd1 = gcd(num1, den2);
    const auto gcd2 = gcd(num2, den1);

    num1.reduce(gcd1);
    num2.reduce(gcd2);

    num1 *= num2;

    den2.reduce(gcd1);
    den1.reduce(gcd2);

    den1 *= den2;
}

//===----------------------------------------------------------------------===//
// Rational
//===----------------------------------------------------------------------===//

std::strong_ordering Rational::operator<=>(const Rational &rhs) const
{
    return compare(
        getNumerator(),
        getDenominator(),
        rhs.getNumerator(),
        rhs.getDenominator());
}

std::optional<Rational> Rational::parse(std::string str, unsigned base)
{
    // Expect a : to separate numerator and denominator.
    const auto split = str.find(':');
    if (split == std::string::npos) {
        // None was found, expect an integer.
        if (auto maybeInt = Integer::parse(str, base))
            return Rational(std::move(maybeInt).value());
        return std::nullopt;
    }

    // Parse numerator using temporary zero terminator.
    str[split] = '\0';
    auto maybeNum = Integer::parse(str.data(), base);
    str[split] = ':';
    if (!maybeNum) return std::nullopt;

    // Parse denominator using permanent zero terminator.
    auto maybeDen = Integer::parse(str.data() + split + 1, base);
    if (!maybeDen) return std::nullopt;

    return Rational(std::move(maybeNum).value(), std::move(maybeDen).value());
}

void Rational::toString(SmallVectorImpl<char> &buffer, int base) const
{
    assert((base >= 2 && base <= 62) || (base <= -2 && base >= -36));

    getNumerator().toString(buffer, base);
    if (isInteger()) return;

    buffer.push_back(':');
    getDenominator().toString(buffer, base);
}

Integer Rational::round(RoundingMode mode) const
{
    Integer result;
    switch (mode) {
    case RoundingMode::Trunc:
        mpz_tdiv_q(
            get_impl(result),
            get_impl(getNumerator()),
            get_impl(getDenominator()));
        break;
    case RoundingMode::Floor:
        mpz_fdiv_q(
            get_impl(result),
            get_impl(getNumerator()),
            get_impl(getDenominator()));
        break;
    case RoundingMode::Ceil:
        mpz_cdiv_q(
            get_impl(result),
            get_impl(getNumerator()),
            get_impl(getDenominator()));
        break;
    case RoundingMode::Nearest:
        result = getDenominator();
        result /= static_cast<ui_t>(2);
        if (isNegative()) result.negate();
        result += getNumerator();
        mpz_tdiv_q(
            get_impl(result),
            get_impl(result),
            get_impl(getDenominator()));
        break;
    }
    return result;
}

Rational &Rational::operator+=(const Rational &rhs)
{
    add_sub<false>(
        m_numerator,
        m_denominator,
        rhs.getNumerator(),
        rhs.getDenominator());
    return *this;
}

Rational &Rational::operator-=(const Rational &rhs)
{
    add_sub<true>(
        m_numerator,
        m_denominator,
        rhs.getNumerator(),
        rhs.getDenominator());
    return *this;
}

Rational &Rational::operator*=(const Rational &rhs)
{
    mul_div(
        m_numerator,
        m_denominator,
        rhs.getNumerator(),
        rhs.getDenominator());
    return *this;
}

Rational &Rational::operator/=(const Rational &rhs)
{
    assert(!rhs.isZero());

    mul_div(
        m_numerator,
        m_denominator,
        rhs.getDenominator(),
        rhs.getNumerator());
    return *this;
}

void Rational::canonicalize(Integer &num, Integer &den)
{
    assert(!den.isZero());

    // Ensure that the denominator is positive.
    if (den.isNegative()) {
        num.negate();
        den.negate();
    }

    // Reduce numerator and denominator as far as possible.
    const auto div = gcd(num, den);
    num.reduce(div);
    den.reduce(div);
}
