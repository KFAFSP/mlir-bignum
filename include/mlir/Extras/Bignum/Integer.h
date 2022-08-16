/// Declares the multiprecision integer type.
///
/// There are two obvious ways this can be implemented:
///     - Use llvm::APInt, but extend the bitwidth of the operands before every
///       operation to ensure exact results.
///     - Use a multiprecision library, such as GMP.
///
/// Since we want to use these integers for number-theoretic reasons, and we
/// don't allow platform-specific sematics for the operations, the latter seems
/// like the most obvious choice.
///
/// However, GMP integers are really heavy on allocations, especially when only
/// up to 64 active bits are used, which is what we expect almost all values to
/// be. So, an additional trick is used to eliminate allocations by providing
/// in-line storage of 64 bits in the C++ wrapper type. Due to limitations of
/// the GMP interface, this does not remove all 1 limb allocations (only during
/// init), and breaks ABI compatibility for an mpq wrapper type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Extras/Bignum/LimbRange.h"

#include <mlir/IR/DialectImplementation.h>
#include <optional>
#include <string>

namespace mlir::ext {

/// Implements a multiprecision integer type.
///
/// Internally implements a wrapper around GMP mpz with an added inline storage
/// implementation to avoid allocations for 64 bit integers.
class [[nodiscard]] Integer final : public detail::LimbRange {
public:
    //===------------------------------------------------------------------===//
    // Initialization
    //===------------------------------------------------------------------===//

    /// Initializes an Integer of value zero.
    ///
    /// @post       `isInline()`
    /// @post       `capacity() == 1`
    /*implicit*/ Integer() : LimbRange(1, 0, &m_inline), m_inline(0)
    {
        // Ensure GMP is initialized.
        attachGMPAllocator();
        assert(isInline());
    }

    /// Initializes an Integer from @p value .
    ///
    /// @post       `isInline()`
    /// @post       `size() <= 1`
    /*implicit*/ Integer(std::integral auto value) : Integer()
    {
        assign(value);
    }
    /// Initializes an Integer from @p value .
    /*implicit*/ Integer(IntegerRef value) : Integer() { assign(value); }
    /// Initializes an Integer from @p value .
    explicit Integer(const llvm::APInt &value, bool isSigned = true) : Integer()
    {
        assign(value, isSigned);
    }
    /// Initializes an Integer from @p value .
    /*implicit*/ Integer(const llvm::APSInt &value) : Integer()
    {
        assign(value);
    }

    /// Initializes an Integer of value 10^ @p exp .
    static Integer exp10(ui_t exp);

    //===------------------------------------------------------------------===//
    // Assignment
    //===------------------------------------------------------------------===//

    /// Assigns @p move to this Integer.
    void assign(Integer &&move)
    {
        reset();
        swap(*this, move);
    }
    /// Assigns @p value to this Integer.
    ///
    /// @post       `size() <= 1`
    void assign(std::integral auto value)
    {
        // Ensure storage fits value.
        // NOTE: Due to inline storage, capacity never drops below 1.
        assert(capacity() >= 1);

        *data() = magnitude(value);
        m_size_and_sign = signum(value);
    }
    /// Assigns @p value to this Integer.
    void assign(IntegerRef value)
    {
        // Ensure storage fits value.
        reserve(value.size());

        // Copy over all limbs.
        std::copy(value.begin(), value.end(), data());
        m_size_and_sign = value.size_and_sign();
    }
    /// Assigns @p value to this Integer.
    void assign(const llvm::APInt &value, bool isSigned = true)
    {
        // Assign as if value were unsigned.
        assign(IntegerRef(value, false));

        if (isSigned && value.isNegative()) {
            // Obtain the two's complement and adjust the sign.
            llvm::APInt::tcNegate(data(), size());
            negate();
        }
    }
    /// Assigns @p value to this Integer.
    void assign(const llvm::APSInt &value) { assign(value, value.isSigned()); }

    /// Assigns @p rhs to this Integer.
    // clang-format off
    template<class Rhs>
    auto operator=(Rhs &&rhs)
        -> decltype(assign(std::forward<Rhs>(rhs)), std::declval<Integer &>())
    // clang-format on
    {
        assign(std::forward<Rhs>(rhs));
        return *this;
    }

    //===------------------------------------------------------------------===//
    // Storage lifetime and movement
    //===------------------------------------------------------------------===//

    /// Swaps the values and/or storages of @p lhs and @p rhs .
    ///
    /// This will swap the storages associated with @p lhs and @p rhs , but
    /// does not attempt to compact them.
    friend void swap(Integer &lhs, Integer &rhs)
    {
        using std::swap;

        if (lhs.isInline() && rhs.isInline()) {
            // Both storages are inline, so only the contents and signum are
            // swapped.
            swap(lhs.m_size_and_sign, rhs.m_size_and_sign);
            swap(lhs.m_inline, rhs.m_inline);
            return;
        }

        const auto steal = [&](Integer &internal, Integer &external) {
            assert(internal.isInline() && "expected inline storage");
            assert(external.isAllocated() && "expected allocated storage");

            // Steal the external storage.
            internal.m_capacity = external.m_capacity;
            internal.m_data = external.m_data;

            // Transfer the internally stored value.
            external.m_inline = internal.m_inline;

            // Transition external to inline storage.
            external.m_capacity = 1;
            external.m_data = &external.m_inline;

            swap(internal.m_size_and_sign, external.m_size_and_sign);
        };
        if (lhs.isInline()) {
            steal(lhs, rhs);
            return;
        }
        if (rhs.isInline()) {
            steal(rhs, lhs);
            return;
        }

        // Inlined version of mpz_swap.
        swap(lhs.m_capacity, rhs.m_capacity);
        swap(lhs.m_size_and_sign, rhs.m_size_and_sign);
        swap(lhs.m_data, rhs.m_data);
    }

    /// @copydoc Integer(IntegerRef)
    /*implicit*/ Integer(const Integer &copy) : Integer(IntegerRef(copy)) {}
    /// @copydoc Integer(const Integer&)
    Integer &operator=(const Integer &copy)
    {
        assign(copy);
        return *this;
    }

    /// Moves the value and/or storage of @p move .
    /*implicit*/ Integer(Integer &&move) : Integer()
    {
        *this = std::move(move);
    }
    /// @copydoc Integer(Integer&&)
    Integer &operator=(Integer &&move)
    {
        assign(std::move(move));
        return *this;
    }

    /// Releases allocated external storage.
    ~Integer()
    {
        if (isAllocated()) free();
    }

    //===------------------------------------------------------------------===//
    // Container interface
    //===------------------------------------------------------------------===//

    /// Resets this value to the default-constructed state.
    ///
    /// @post       `isInline()`
    /// @post       `capacity() == 1`
    void reset()
    {
        // External storag is free'd.
        if (isAllocated()) free();

        // We immediately go to internal storage.
        m_capacity = 1;
        m_data = &m_inline;
        assert(isInline());

        // We call the underlying clear that resets the size.
        LimbRange::clear();
    }

    /// Shrinks the associated storage to fit the stored value.
    ///
    /// @post       `capacity() == size()`
    void shrink_to_fit()
    {
        const auto limbs = size();
        if (limbs == capacity()) return;

        // Try inlining the storage.
        if (compact()) return;

        // Fallback to GMP reallocation.
        assert(limbs > 0);
        realloc(limbs);
    }

    /// Ensures that this integer can fit up to @p limbs .
    ///
    /// @post       `capacity() >= limbs`
    void reserve(size_type limbs)
    {
        // There is already enough available capacity.
        if (capacity() >= limbs) return;

        if (isInline()) {
            assert(limbs > 1);

            // Move to external storage.
            m_data = allocate(limbs);
            m_capacity = limbs;
            *data() = m_inline;
            return;
        }

        // Fallback to GMP reallocation.
        realloc(limbs);
        assert(capacity() >= limbs);
    }

    /// Resizes this integer to @p limbs , resetting to 0 on data loss.
    ///
    /// @post       `capacity() == limbs`
    ///
    /// @retval     true    Value was resized in-place.
    /// @retval     false   Data was lost and value reset to 0.
    [[nodiscard]] bool resize(size_type limbs)
    {
        switch (limbs) {
        case 0:
        {
            const auto lossless = isZero();
            reset();
            return lossless;
        }
        case 1:
            // Try to compact, or reset to zero.
            if (compact()) return true;
            reset();
            return false;
        default:
            // Fallback to GMP reallocation.
            return realloc(limbs);
        }
    }

    //===------------------------------------------------------------------===//
    // Magnitude
    //===------------------------------------------------------------------===//

    /// @copydoc abs(Integer)
    friend Integer magnitude(Integer value) { return abs(value); }

    //===------------------------------------------------------------------===//
    // Input
    //===------------------------------------------------------------------===//

    /// Attempts to parse an Integer from a zero-terminated @p str .
    ///
    /// @pre        `base == 0 || (base >= 2 && base <= 62)`
    [[nodiscard]] static std::optional<Integer>
    parse(const char* str, unsigned base = 0);
    /// Attempts to parse an Integer from a @p str .
    ///
    /// @pre        `base == 0 || (base >= 2 && base <= 62)`
    [[nodiscard]] static std::optional<Integer>
    parse(const std::string &str, unsigned base = 0)
    {
        return parse(str.c_str(), base);
    }

    //===------------------------------------------------------------------===//
    // Operations
    //===------------------------------------------------------------------===//

    /// Divides this integer by @p rhs if it is known to be divisible.
    ///
    /// @pre        This value is divisible by @p rhs .
    void reduce(IntegerRef rhs);
    /// @copydoc reduce(IntegerRef)
    void reduce(const Integer &rhs) { reduce(static_cast<IntegerRef>(rhs)); }

    /// Adds @p rhs to this value.
    Integer &operator+=(IntegerRef rhs);
    /// @copydoc operator+=(IntegerRef)
    Integer &operator+=(const Integer &rhs)
    {
        return *this += static_cast<IntegerRef>(rhs);
    }
    /// Subtracts @p rhs from this value.
    Integer &operator-=(IntegerRef rhs);
    /// @copydoc operator-=(IntegerRef)
    Integer &operator-=(const Integer &rhs)
    {
        return *this -= static_cast<IntegerRef>(rhs);
    }
    /// Multiplies this value by @p rhs .
    Integer &operator*=(IntegerRef rhs);
    /// @copydoc operator*=(IntegerRef)
    Integer &operator*=(const Integer &rhs)
    {
        return *this *= static_cast<IntegerRef>(rhs);
    }
    /// Divides this value by @p rhs , rounding towards zero.
    ///
    /// @pre        `!rhs.isZero()`
    Integer &operator/=(IntegerRef rhs);
    /// @copydoc operator/=(IntegerRef)
    Integer &operator/=(const Integer &rhs)
    {
        return *this /= static_cast<IntegerRef>(rhs);
    }
    /// Sets this value to the rest after dividing by @p rhs , keeping the sign.
    ///
    /// @pre        `!rhs.isZero()`
    Integer &operator%=(IntegerRef rhs);
    /// @copydoc operator%=(IntegerRef)
    Integer &operator%=(const Integer &rhs)
    {
        return *this %= static_cast<IntegerRef>(rhs);
    }

    /// Obtains the negated value.
    Integer operator-() const
    {
        Integer copy(*this);
        copy.negate();
        return copy;
    }
    /// Obtains the absolute value of @p value .
    Integer friend abs(Integer value)
    {
        value.dropSign();
        return value;
    }
    /// @copydoc abs(Integer)
    Integer friend magnitude(const Integer &value) { return abs(value); }

#define BINOP(op)                                                              \
    template<class Rhs>                                                        \
    requires requires                                                          \
    {                                                                          \
        std::declval<Integer &>() op## = std::declval<Rhs>();                  \
    }                                                                          \
    Integer operator op(Rhs &&rhs) const                                       \
    {                                                                          \
        Integer copy(*this);                                                   \
        copy op## = std::forward<Rhs>(rhs);                                    \
        return copy;                                                           \
    }

    /// Obtains the sum of this value with @p rhs .
    BINOP(+)
    /// Obtains the difference of this value and @p rhs .
    BINOP(-)
    /// Obtains the product of this value and @p rhs .
    BINOP(*)
    /// Obtains the quotient of this value and @p rhs , rounded towards zero.
    ///
    /// @pre        `!rhs.isZero()`
    BINOP(/)
    /// Obtains the same-signed remainder of this value divided by @p rhs .
    ///
    /// @pre        `!rhs.isZero()`
    BINOP(%)

#undef BINOP

private:
    /// Ensures that the custom GMP allocator is attached.
    ///
    /// @post       Custom GMP allocator is attached.
    static void attachGMPAllocator();

    //===------------------------------------------------------------------===//
    // Inline storage implementation
    //===------------------------------------------------------------------===//

    /// Determines whether the storage is inline.
    [[nodiscard]] constexpr bool isInline() const
    {
        return data() == &m_inline;
    }
    /// Determines whether external storage is allocated.
    [[nodiscard]] constexpr bool isAllocated() const
    {
        return !isInline() && capacity() != 0;
    }

    /// Allocates external storage.
    ///
    /// @pre        Custom GMP allocator is attached.
    /// @pre        `words <= gmp::max_size`
    [[nodiscard]] static pointer allocate(size_type words);
    /// Frees the external storage.
    ///
    /// @pre        `isAllocated()`
    ///
    /// @post       `capacity() == 0`
    /// @post       `size_and_sign() == 0`
    void free();
    /// Reallocates the external storage.
    ///
    /// @retval     true    The value remains the same.
    /// @retval     false   Data was lost and the value reset to 0.
    ///
    /// @post       `capacity() == limbs`
    bool realloc(size_type limbs);

    /// Attempts to move into internal storage and returns `isInline()`.
    ///
    /// @note       Since compacting will never change the value, it technically
    ///             has const semantics, but it requires modifying the fields.
    ///
    /// @post       `result == isInline()`
    bool compact()
    {
        /// Storage cannot possibly be inlined (doesn't fit).
        if (size() > 1) return false;

        if (isAllocated()) {
            // Move the value into internal storage.
            m_inline = *data();
            // Free the external storage.
            free();
        }

        // Storage is now internal.
        m_capacity = 1;
        m_data = &m_inline;
        return true;
    }

    // NOTE: This is the inline storage that avoids allocating external limb
    //       storage when initializing values of type Integer that use up to
    //       64 bits of storage.
    // NOTE: This breaks ABI compatibility with __mpq_struct!
    value_type m_inline;
};

#define BINOP(op)                                                              \
    template<class Lhs>                                                        \
    requires requires { std::declval<Integer>() + std::declval<Lhs>(); }       \
    inline Integer operator op(Lhs &&lhs, const Integer &rhs)                  \
    {                                                                          \
        return Integer(std::forward<Lhs>(lhs)) op## = rhs;                     \
    }

/// Obtains the sum of @p lhs and @p rhs .
BINOP(+)
/// Obtains the difference between @p lhs and @p rhs .
BINOP(-)
/// Obtains the product of @p lhs and @p rhs .
BINOP(*)
/// Obtains the quotient of @p lhs and @p rhs , rounded towards zero.
///
/// @pre        `!rhs.isZero()`
BINOP(/)
/// Obtains the remainder of @p lhs divided by @p rhs , keeping the sign.
///
/// @pre        `!rhs.isZero()`
BINOP(%)

#undef BINOP

/// Computes the greatest common divisor of @p lhs and @p rhs .
[[nodiscard]] Integer gcd(IntegerRef lhs, IntegerRef rhs);

} // namespace mlir::ext

/// Provides an MLIR field parser for the unlimited precision Integer type.
template<>
struct mlir::FieldParser<mlir::ext::Integer> {
    FailureOr<ext::Integer> parse(DialectAsmParser &parser) const
    {
        APInt value;
        if (parser.parseInteger(value)) return failure();
        return ext::Integer(value, true);
    }
};
