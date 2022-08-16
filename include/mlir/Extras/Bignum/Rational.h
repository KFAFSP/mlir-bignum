#pragma once

#include "mlir/Extras/Bignum/Integer.h"

namespace mlir::ext {

/// Implements a multiprecision rational number type.
///
/// Based on the Integer type, mirroring the semantics of GMP mpq, but unable to
/// be ABI compatible due to our inline storage optimization.
class [[nodiscard]] Rational final {
public:
    using size_type = Integer::size_type;

    //===------------------------------------------------------------------===//
    // Initialization
    //===------------------------------------------------------------------===//

    /// Initializes a Rational to 0.
    /*implicit*/ Rational() : m_numerator(), m_denominator(static_cast<ui_t>(1))
    {}
    /// Initializes a Rational to @p num .
    /*implicit*/ Rational(Integer num)
            : m_numerator(std::move(num)),
              m_denominator(static_cast<ui_t>(1))
    {}
    /// Initializes a Rational from @p num and @p den .
    ///
    /// @pre        `!den.isZero()`
    /*implicit*/ Rational(Integer num, Integer den)
            : m_numerator(std::move(num)),
              m_denominator(std::move(den))
    {
        assert(!m_denominator.isZero());
        canonicalize();
    }

    /// Initializes a Rational of value 10^ @p exp .
    static Rational exp10(si_t exp)
    {
        Rational result(Integer::exp10(std::abs(exp)));
        if (exp < 0) result.invert();
        return result;
    }

    //===------------------------------------------------------------------===//
    // Assignment
    //===------------------------------------------------------------------===//

    /// Assigns @p rhs to the numerator and 1 to the denominator of this
    /// Rational.
    template<class Rhs>
    requires requires(Integer i, Rhs &&rhs)
    {
        i.assign(std::forward<Rhs>(rhs));
    }
    auto assign(Rhs &&rhs)
    {
        m_numerator.assign(std::forward<Rhs>(rhs));
        m_denominator.assign(1);
    }
    /// Assigns @p num and @p den to this Rational.
    ///
    /// @pre        `den != 0`
    template<class Num, class Den>
    requires requires(Integer i, Num &&num, Den &&den)
    {
        i.assign(std::forward<Num>(num));
        i.assign(std::forward<Den>(den));
    }
    void assign(Num &&num, Den &&den)
    {
        m_numerator.assign(std::forward<Num>(num));
        m_denominator.assign(std::forward<Den>(den));

        assert(!m_denominator.isZero());
        canonicalize();
    }

    /// Assigns @p rhs to this Rational.
    // clang-format off
    template<class Rhs>
    auto operator=(Rhs &&rhs)
        -> decltype(assign(std::forward<Rhs>(rhs)), std::declval<Rational &>())
    // clang-format on
    {
        assign(std::forward<Rhs>(rhs));
        return *this;
    }

    //===------------------------------------------------------------------===//
    // Storage lifetime and movement
    //===------------------------------------------------------------------===//

    /// Swaps the values of @p lhs and @p rhs .
    friend void swap(Rational &lhs, Rational &rhs)
    {
        // Swap the contained values.
        swap(lhs.m_numerator, rhs.m_numerator);
        swap(lhs.m_denominator, rhs.m_denominator);
    }

    /// Copies a Rational value.
    /*implicit*/ Rational(const Rational &) = default;
    /// @copydoc operator=(const Rational&)
    Rational &operator=(const Rational &) = default;

    /// Moves a Rational value.
    /*implicit*/ Rational(Rational &&) = default;
    /// @copydoc operator=(Rational&&)
    Rational &operator=(Rational &&) = default;

    //===------------------------------------------------------------------===//
    // Observers
    //===------------------------------------------------------------------===//

    /// Gets the numerator.
    const Integer &getNumerator() const { return m_numerator; }
    /// Gets the denominator.
    const Integer &getDenominator() const { return m_denominator; }

    /// Determines whether this value is negative.
    [[nodiscard]] bool isNegative() const
    {
        return getNumerator().isNegative();
    }
    /// Determines whether this value is zero.
    [[nodiscard]] bool isZero() const { return getNumerator().isZero(); }
    /// Determines whether this value is one.
    [[nodiscard]] bool isOne() const
    {
        return getNumerator() == getDenominator();
    }

    /// Determines whether this value is an integer.
    [[nodiscard]] bool isInteger() const { return getDenominator().isOne(); }
    /// Obtains the contained value as an Integer if it is integer.
    ///
    /// @retval     std::nullopt    The contained value is not an integer.
    /// @retval     Integer         The contained value.
    [[nodiscard]] std::optional<Integer> asInteger() const
    {
        if (!isInteger()) return std::nullopt;
        return getNumerator();
    }
    /// Obtains the contained value as an Integer.
    ///
    /// @pre        `isInteger()`
    const Integer &toInteger() const
    {
        assert(isInteger());
        return getNumerator();
    }
    /// @copydoc Integer::asUnsigned
    [[nodiscard]] std::optional<ui_t> asUnsigned() const
    {
        if (!isInteger()) return std::nullopt;
        return getNumerator().asUnsigned();
    }
    /// @copydoc Integer::toUnsigned
    [[nodiscard]] ui_t toUnsigned() const { return asUnsigned().value(); }
    /// @copydoc Integer::asSigned
    [[nodiscard]] std::optional<si_t> asSigned() const
    {
        if (!isInteger()) return std::nullopt;
        return getNumerator().asSigned();
    }
    /// @copydoc Integer::toSigned
    [[nodiscard]] si_t toSigned() const { return asSigned().value(); }

    /// Determines whether this value is a proper fraction.
    [[nodiscard]] bool isProper() const
    {
        return std::is_gt(getDenominator().compare(
                   getNumerator().size(),
                   getNumerator().data()))
               && !isZero();
    }

    //===------------------------------------------------------------------===//
    // Signum
    //===------------------------------------------------------------------===//

    /// Obtains the signum {-1, 0, 1} of @p value .
    [[nodiscard]] friend auto signum(const Rational &value)
    {
        return signum(value.getNumerator());
    }

    //===------------------------------------------------------------------===//
    // Magnitude
    //===------------------------------------------------------------------===//

    /// @copydoc abs(Rational)
    friend Rational magnitude(Rational value) { return abs(value); }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//

    /// Compares this rational with @p rhs .
    [[nodiscard]] std::strong_ordering operator<=>(Integer rhs) const
    {
        rhs *= getDenominator();
        return getNumerator() <=> rhs;
    }

    /// Compares this rational with @p rhs .
    [[nodiscard]] std::strong_ordering operator<=>(const Rational &rhs) const;

    //===------------------------------------------------------------------===//
    // Equality comparison
    //===------------------------------------------------------------------===//

    /// Determines whether two rational values are equal.
    [[nodiscard]] constexpr auto operator==(const auto &rhs) const
        -> decltype(std::is_eq(*this <=> rhs))
    {
        return std::is_eq(*this <=> rhs);
    }

    //===------------------------------------------------------------------===//
    // Hashing
    //===------------------------------------------------------------------===//

    /// Computes a hash for @p value .
    [[nodiscard]] friend llvm::hash_code hash_value(const Rational &value)
    {
        return llvm::hash_combine(value.getNumerator(), value.getDenominator());
    }

    //===------------------------------------------------------------------===//
    // Container interface
    //===------------------------------------------------------------------===//

    /// Resets this value to 0.
    void clear()
    {
        m_numerator.clear();
        m_denominator.assign(1);
    }

    /// Resets this value to the default-constructed state.
    void reset()
    {
        m_numerator.reset();
        m_denominator.reset();
        m_denominator.assign(1);
    }

    /// Shrinks the associated storage to fit the stored value.
    void shrink_to_fit()
    {
        m_numerator.shrink_to_fit();
        m_denominator.shrink_to_fit();
    }

    /// Ensures that numerator and denominator can fit @p num and @p den limbs
    /// respectively.
    void reserve(size_type num, size_type den = 0)
    {
        m_numerator.reserve(num);
        m_denominator.reserve(den);
    }

    //===------------------------------------------------------------------===//
    // Input
    //===------------------------------------------------------------------===//

    /// Attempts to parse a Rational from a  @p str .
    ///
    /// @pre        `base == 0 || (base >= 2 && base <= 62)`
    static std::optional<Rational> parse(std::string str, unsigned base = 0);

    //===------------------------------------------------------------------===//
    // Output
    //===------------------------------------------------------------------===//

    /// Appends a string representation of this value to @p buffer.
    ///
    /// The @p base must be within -2..-36 (UPPERCASE) or 2..62 (lowercase).
    ///
    /// @pre        `(base >= 2 && base <= 62) || (base <= -2 && base >= -36)`
    void toString(llvm::SmallVectorImpl<char> &buffer, int base = 10) const;

    /// Writes @p value to @p os in base 10.
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const Rational &value)
    {
        llvm::SmallString<32> buffer;
        value.toString(buffer);
        return os << buffer;
    }

    /// @copydoc operator<<(llvm::raw_ostream &, const Rational &)
    friend std::ostream &operator<<(std::ostream &os, const Rational &value)
    {
        llvm::SmallString<32> buffer;
        value.toString(buffer);
        return os << std::string_view(buffer.data(), buffer.size());
    }

    //===------------------------------------------------------------------===//
    // Operations
    //===------------------------------------------------------------------===//

    /// Enumeration of supported rounding modes.
    enum class RoundingMode {
        /// Round towards zero (C standard behavior).
        TowardsZero = 0,
        Trunc = 0,
        /// Round towards the next smaller integer.
        Floor,
        /// Round towards the next greater integer.
        Ceil,
        /// Round towards the nearest integer (up on .5).
        Nearest
    };

    /// Obtains the contained value rounded to an integer according to @p mode .
    Integer round(RoundingMode mode = RoundingMode::Nearest) const;

    /// Swaps numerator and denominator (computing `rec` in-place).
    void invert()
    {
        if (isZero()) return;
        swap(m_numerator, m_denominator);

        // Re-establish canonicalization of sign.
        if (m_denominator.isNegative()) {
            m_numerator.negate();
            m_denominator.negate();
        }
    }
    /// Flips the sign (computing `neg` in-place).
    void negate() { m_numerator.negate(); }
    /// Removes any negative sign (computing `abs` in-place).
    ///
    /// @post       `!isNegative()`
    void dropSign() { m_numerator.dropSign(); }

    /// Adds @p rhs to this value.
    Rational &operator+=(Integer rhs)
    {
        add(m_numerator, m_denominator, rhs);
        return *this;
    }
    /// @copydoc operator+=(Integer)
    Rational &operator+=(const Rational &rhs);
    /// Subtracts @p rhs from this value.
    Rational &operator-=(Integer rhs)
    {
        rhs.negate();
        add(m_numerator, m_denominator, rhs);
        return *this;
    }
    /// @copydoc operator-=(Integer)
    Rational &operator-=(const Rational &rhs);
    /// Multiplies this value with @p rhs .
    Rational &operator*=(Integer rhs)
    {
        multiply(m_numerator, m_denominator, rhs);
        if (rhs.isNegative()) {
            m_numerator.negate();
            m_denominator.negate();
        }
        return *this;
    }
    /// @copydoc operator*=(Integer)
    Rational &operator*=(const Rational &rhs);
    /// Divides this value by @p rhs .
    ///
    /// @pre        `!rhs.isZero()`
    Rational &operator/=(Integer rhs)
    {
        assert(!rhs.isZero());
        multiply(m_denominator, m_numerator, rhs);
        if (rhs.isNegative()) {
            m_numerator.negate();
            m_denominator.negate();
        }
        return *this;
    }
    /// @copydoc operator/=(Integer)
    Rational &operator/=(const Rational &rhs);

    /// Obtains the reciprocal value.
    Rational operator~() const
    {
        Rational copy(*this);
        copy.invert();
        return copy;
    }
    /// Obtains the negated value.
    Rational operator-() const
    {
        Rational copy(*this);
        copy.negate();
        return copy;
    }
    /// Obtains the absolute value.
    Rational friend abs(Rational value)
    {
        value.dropSign();
        return value;
    }

#define BINOP(op)                                                              \
    template<class Rhs>                                                        \
    requires requires                                                          \
    {                                                                          \
        std::declval<Rational &>() op## = std::declval<Rhs>();                 \
    }                                                                          \
    Rational operator op(Rhs &&rhs) const                                      \
    {                                                                          \
        Rational copy(*this);                                                  \
        copy op## = std::forward<Rhs>(rhs);                                    \
        return copy;                                                           \
    }

    /// Obtains the sum of this value with @p rhs .
    BINOP(+)
    /// Obtains the difference of this value and @p rhs .
    BINOP(-)
    /// Obtains the product of this value and @p rhs .
    BINOP(*)
    /// Obtains the quotient of this value and @p rhs .
    ///
    /// @pre        `!rhs.isZero()`
    BINOP(/)

#undef BINOP

private:
    /// Adds @p with to @p num / @p den in-place.
    static void add(Integer &num, Integer &den, Integer &with)
    {
        // gcd(a, b + ac) = gcd(a, b) ->
        //     isCanonical(n, d) -> isCanonical(n + xd, d)
        with *= den;
        num += with;
    }
    /// Multiplies @p num / @p den with @p with in-place.
    static void multiply(Integer &num, Integer &den, Integer &with)
    {
        // gcd(ab, c) = gcd(a, c) * gcd(b, c) ->
        //     isCanonical(n, d) -> gcd(xn, d) = gcd(x, d) * 1
        const auto fac = gcd(den, with);
        if (!fac.isOne()) {
            num.reduce(fac);
            den.reduce(fac);
            with.reduce(fac);
        }
        num *= with;
    }

#ifndef NDEBUG
    /// Determines whether @p num / @p den is a canonical fraction.
    ///
    /// @note       Only used for debugging.
    [[nodiscard]] static bool
    isCanonical(const Integer &num, const Integer &den)
    {
        return !den.isNegative() && gcd(num, den).isOne();
    }
    /// Determines whether this fraction is canonical.
    ///
    /// @note       Only used for debugging.
    [[nodiscard]] bool isCanonical() const
    {
        return isCanonical(getNumerator(), getDenominator());
    }
#endif

    /// Canonicalizes the fraction @p num / @p den .
    ///
    /// @pre        `!den.isZero()`
    static void canonicalize(Integer &num, Integer &den);
    /// Canonicalizes this fraction.
    ///
    /// @note       Since canonicalizing will never change the value,
    ///             it technically has const semantics, but it requires
    ///             modifying the fields.
    ///
    /// @pre        `!getDenominator().isZero()`
    /// @post       `isCanonical()`
    void canonicalize() { canonicalize(m_numerator, m_denominator); }

    Integer m_numerator;
    Integer m_denominator;
};

#define BINOP(op)                                                              \
    template<class Lhs>                                                        \
    requires requires { Rational(std::declval<Lhs>()); }                       \
    inline Rational operator op(Lhs &&lhs, const Rational &rhs)                \
    {                                                                          \
        return Rational(std::forward<Lhs>(lhs)) op## = rhs;                    \
    }

/// Obtains the sum of @p lhs and @p rhs .
BINOP(+)
/// Obtains the difference between @p lhs and @p rhs .
BINOP(-)
/// Obtains the product of @p lhs and @p rhs .
BINOP(*)
/// Obtains the quotient of @p lhs and @p rhs .
///
/// @pre        `!rhs.isZero()`
BINOP(/)

#undef BINOP

} // namespace mlir::ext
