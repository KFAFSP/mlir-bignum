/// Implements the IntegerTraits template base.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Extras/Bignum/ABI.h"
#include "mlir/Extras/Bignum/LLVM.h"
#include "mlir/Extras/Bignum/Native.h"

#include <algorithm>
#include <cassert>
#include <compare>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <numeric>
#include <optional>
#include <string_view>

namespace mlir::ext::detail {

/// Populates @p result with a string representation of @p data .
///
/// The @p base must be within -2..-36 (UPPERCASE) or 2..62 (lowercase).
///
/// @pre        `(base >= 2 && base <= 62) || (base <= -2 && base >= -36)`
void toStringImpl(
    llvm::SmallVectorImpl<char> &buffer,
    gmp::size_and_sign_t size_and_sign,
    const gmp::limb_t* data,
    int base = 10);

/// Trait base for multiprecision integers.
///
/// This CRTP mixin provides the common interface for the multiprecision integer
/// type @p Derived , given that it provides the `size_and_sign()` and `data()`
/// accessors.
///
/// This trait implementation is storage-agnostic, and so it can not provide
/// methods that could potentially change the size of the limb range.
///
/// A multiprecision integer `i` is:
///     - a non-resizable container of stored limbs.
///     - provides an overload for `signum(i)`.
///     - provides an overload for `llvm::hash_value(i)`.
///     - provides overloads for comparisions with other integers.
///
/// @tparam     Derived     The derived type.
/// @tparam     Mutable     Whether the limb range is mutable.
template<class Derived, bool Mutable>
class IntegerTraits {
    constexpr Derived &_self() { return static_cast<Derived &>(*this); }
    constexpr const Derived &_self() const
    {
        return static_cast<const Derived &>(*this);
    }

    constexpr auto _data() { return _self().data(); }
    constexpr auto _data() const { return _self().data(); }
    constexpr auto _size_and_sign() const { return _self().size_and_sign(); }

public:
    //===------------------------------------------------------------------===//
    // Type aliases
    //===------------------------------------------------------------------===//

    /// Type of the stored limbs.
    using value_type = mlir::ext::gmp::limb_t;
    /// Type of the combined size and sign.
    using size_and_sign_type = mlir::ext::gmp::size_and_sign_t;

    /// Type storing the number of limbs.
    using size_type = std::make_unsigned_t<size_and_sign_type>;

    /// Possibly const-qualified type of the stored limbs.
    using element_type =
        std::conditional_t<Mutable, value_type, const value_type>;

    /// Pointer to stored limbs.
    using pointer = element_type*;
    /// Pointer to immutable stored limbs.
    using const_pointer = const value_type*;
    /// Reference to a stored limb.
    using reference = element_type &;
    /// Reference to an immutable stored limb.
    using const_reference = const element_type &;

    //===------------------------------------------------------------------===//
    // Element access
    //===------------------------------------------------------------------===//

    /// Gets a reference to the least significand limb.
    ///
    /// @pre        `!empty()`
    [[nodiscard]] constexpr reference front()
    {
        assert(!empty());
        return _data()[0];
    }
    /// Gets a reference to the immutable least significand limb.
    ///
    /// @pre        `!empty()`
    [[nodiscard]] constexpr const_reference front() const
    {
        assert(!empty());
        return _data()[0];
    }
    /// Gets a reference to the most significand limb.
    ///
    /// @pre        `!empty()`
    [[nodiscard]] constexpr reference back()
    {
        assert(!empty());
        return _data()[size() - 1];
    }
    /// Gets a reference to the immutable most significand limb.
    ///
    /// @pre        `!empty()`
    [[nodiscard]] constexpr const_reference back() const
    {
        assert(!empty());
        return _data()[size() - 1];
    }

    /// Gets a reference to the limb at @p idx .
    ///
    /// @pre        `idx < size()`
    [[nodiscard]] constexpr reference operator[](size_type idx)
    {
        assert(idx < size());
        return _data()[idx];
    }
    /// Gets a reference to the immutable limb at @p idx .
    ///
    /// @pre        `idx < size()`
    [[nodiscard]] constexpr const_reference operator[](size_type idx) const
    {
        assert(idx < size());
        return _data()[idx];
    }

    //===------------------------------------------------------------------===//
    // Observers
    //===------------------------------------------------------------------===//

    /// Determines whether the integer is zero (no limbs stored).
    [[nodiscard]] constexpr bool empty() const { return _size_and_sign() == 0; }
    /// Obtains the number of limbs.
    [[nodiscard]] constexpr size_type size() const
    {
        return magnitude(_size_and_sign());
    }

    /// Determines whether the sign of this integer is negative.
    [[nodiscard]] constexpr bool isNegative() const
    {
        return _size_and_sign() < 0;
    }
    /// Determines whether this integer is zero.
    [[nodiscard]] constexpr bool isZero() const { return empty(); }
    /// Determines whether this integer is positive one.
    [[nodiscard]] constexpr bool isOne() const
    {
        return _size_and_sign() == 1 && front() == 1;
    }
    /// Determines whether this integer is odd.
    [[nodiscard]] constexpr bool isOdd() const
    {
        return _size_and_sign() != 0 && (front() & static_cast<value_type>(1));
    }

    /// Obtains the contained value as ui_t if it fits.
    ///
    /// @retval     std::nullopt    The contained value is larger than ui_t.
    /// @retval     ui_t            The contained value.
    [[nodiscard]] constexpr std::optional<ui_t> asUnsigned() const
    {
        switch (_size_and_sign()) {
        case 0: return static_cast<ui_t>(0);
        case 1: return front();
        default: return std::nullopt;
        }
    }
    /// Obtains the contained value as ui_t.
    ///
    /// @pre        `asUnsigned().has_value()`
    [[nodiscard]] constexpr ui_t toUnsigned() const
    {
        return asUnsigned().value();
    }
    /// Obtains the contained value as si_t if it fits.
    ///
    /// @retval     std::nullopt    The contained value is larger than si_t.
    /// @retval     si_t            The contained value.
    [[nodiscard]] constexpr std::optional<si_t> asSigned() const
    {
        switch (_size_and_sign()) {
        case 0: return static_cast<si_t>(0);
        case 1:
        {
            if (front() <= magnitude(max_si)) return static_cast<si_t>(front());
            return std::nullopt;
        }
        case -1:
        {
            if (front() <= magnitude(min_si))
                return static_cast<si_t>(-front());
            return std::nullopt;
        }
        default: return std::nullopt;
        }
    }
    /// Obtains the contained value as si_t.
    ///
    /// @pre        `asSigned().has_value()`
    [[nodiscard]] constexpr ui_t toSigned() const { return asSigned().value(); }

    //===------------------------------------------------------------------===//
    // Iterators
    //===------------------------------------------------------------------===//

    /// Iterator over stored limbs.
    using iterator = pointer;
    /// Iterator over immutable stored limbs.
    using const_iterator = const_pointer;
    /// Reverse iterator over stored limbs.
    using reverse_iterator = std::reverse_iterator<iterator>;
    /// Reverse iterator over immutable stored limbs.
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    /// Gets the begin iterator over stored limbs.
    [[nodiscard]] constexpr iterator begin() { return _data(); }
    /// Gets the begin iterator over immutable stored limbs.
    [[nodiscard]] constexpr const_iterator begin() const { return _data(); }
    /// Gets the end iterator over stored limbs.
    [[nodiscard]] constexpr iterator end() { return _data() + size(); }
    /// Gets the end iterator over immutable stored limbs.
    [[nodiscard]] constexpr const_iterator end() const
    {
        return _data() + size();
    }

    /// Gets the reverse begin iterator over stored limbs.
    [[nodiscard]] constexpr reverse_iterator rbegin()
    {
        return std::make_reverse_iterator(end());
    }
    /// Gets the reverse begin iterator over immutable stored limbs.
    [[nodiscard]] constexpr const_reverse_iterator rbegin() const
    {
        return std::make_reverse_iterator(end());
    }
    /// Gets the reverse end iterator over stored limbs.
    [[nodiscard]] constexpr reverse_iterator rend()
    {
        return std::make_reverse_iterator(begin());
    }
    /// Gets the reverse end iterator over immutable stored limbs.
    [[nodiscard]] constexpr const_reverse_iterator rend() const
    {
        return std::make_reverse_iterator(begin());
    }

    //===------------------------------------------------------------------===//
    // Signum
    //===------------------------------------------------------------------===//

    /// Obtains the signum {-1, 0, 1} of @p value .
    [[nodiscard]] constexpr friend auto signum(const IntegerTraits &value)
    {
        return signum(value._size_and_sign());
    }

    //===------------------------------------------------------------------===//
    // Comparison with native integers.
    //===------------------------------------------------------------------===//

    /// Compares the contained value with @p rhs .
    // clang-format off
    template<std::unsigned_integral UInt>
    requires (is_int_subrange_v<UInt, value_type>)
    [[nodiscard]] constexpr std::strong_ordering operator<=>(UInt rhs) const
    // clang-format on
    {
        // Compare sizes and signs.
        const auto signComp =
            _size_and_sign() <=> static_cast<size_and_sign_type>(signum(rhs));
        if (!std::is_eq(signComp) || empty()) return signComp;

        // Compare magnitude.
        return front() <=> static_cast<value_type>(rhs);
    }
    /// Compares the contained value with @p rhs .
    // clang-format off
    template<std::signed_integral SInt>
    requires (is_int_subrange_v<std::make_unsigned_t<SInt>, value_type>)
    [[nodiscard]] constexpr std::strong_ordering operator<=>(SInt rhs) const
    // clang-format on
    {
        // Compare sizes and signs.
        const auto signComp =
            _size_and_sign() <=> static_cast<size_and_sign_type>(signum(rhs));
        if (!std::is_eq(signComp) || empty()) return signComp;

        // Compare magnitude.
        const auto absComp =
            front() <=> static_cast<value_type>(magnitude(rhs));
        // On negative values, result of comparison is flipped.
        return isNegative() ? 0 <=> absComp : absComp;
    }

    //===------------------------------------------------------------------===//
    // Comparison with other multiprecision integers.
    //===------------------------------------------------------------------===//

    /// Compares the contained value with an immutable limb range.
    [[nodiscard]] constexpr std::strong_ordering
    compare(size_and_sign_type size_and_sign, const_pointer data) const
    {
        // Compare size and signs. size() is always minimal!
        const auto signComp = _size_and_sign() <=> _size_and_sign();
        if (!std::is_eq(signComp) || empty()) return signComp;

        // Lexicographically compare the limbs from most to least significant.
        const auto size = magnitude(size_and_sign);
        const auto limbComp = std::lexicographical_compare_three_way(
            rbegin(),
            rend(),
            std::make_reverse_iterator(data + size),
            std::make_reverse_iterator(data));
        // Invert result if negative.
        return isNegative() ? 0 <=> limbComp : limbComp;
    }
    /// Compares the contained value with @p rhs .
    [[nodiscard]] std::strong_ordering
    compare(llvm::APInt rhs, bool isSigned = true) const
    {
        if (rhs.isNegative() && isSigned) rhs.negate();
        return compare(
            signum(rhs, isSigned) * rhs.getActiveWords(),
            rhs.getRawData());
    }

    /// Compares the contained value with @p rhs .
    [[nodiscard]] constexpr std::strong_ordering
    operator<=>(const IntegerTraits &rhs) const
    {
        return compare(rhs._size_and_sign(), rhs._data());
    }
    /// Compares the contained value with @p rhs .
    [[nodiscard]] std::strong_ordering
    operator<=>(const llvm::APSInt &rhs) const
    {
        return compare(rhs, rhs.isSigned());
    }

    //===------------------------------------------------------------------===//
    // Equality comparison
    //===------------------------------------------------------------------===//

    /// Determines whether two integer values are equal.
    [[nodiscard]] constexpr auto operator==(const auto &rhs) const
        -> decltype(std::is_eq(*this <=> rhs))
    {
        return std::is_eq(*this <=> rhs);
    }

    //===------------------------------------------------------------------===//
    // Hashing
    //===------------------------------------------------------------------===//

    /// Computes a hash for @p value .
    [[nodiscard]] friend llvm::hash_code hash_value(const IntegerTraits &value)
    {
        return llvm::hash_combine_range(value.begin(), value.end())
               * signum(value);
    }

    //===------------------------------------------------------------------===//
    // Output
    //===------------------------------------------------------------------===//

    /// Appends a string representation of this value to @p buffer.
    ///
    /// The @p base must be within -2..-36 (UPPERCASE) or 2..62 (lowercase).
    ///
    /// @pre        `(base >= 2 && base <= 62) || (base <= -2 && base >= -36)`
    void toString(llvm::SmallVectorImpl<char> &buffer, int base = 10) const
    {
        toStringImpl(buffer, _size_and_sign(), _data(), base);
    }

    /// Writes @p value to @p os in base 10.
    friend llvm::raw_ostream &
    operator<<(llvm::raw_ostream &os, const IntegerTraits &value)
    {
        llvm::SmallString<32> buffer;
        value.toString(buffer);
        return os << buffer;
    }

    /// @copydoc operator<<(llvm::raw_ostream &, const IntegerTraits &)
    friend std::ostream &
    operator<<(std::ostream &os, const IntegerTraits &value)
    {
        llvm::SmallString<32> buffer;
        value.toString(buffer);
        return os << std::string_view(buffer.data(), buffer.size());
    }

    //===------------------------------------------------------------------===//
    // Bit-twiddling
    //===------------------------------------------------------------------===//

    /// Determines whether the contained value is a power of two.
    [[nodiscard]] friend constexpr bool
    has_single_bit(const IntegerTraits &value)
    {
        if (const auto size = value.size()) {
            return std::has_single_bit(value.back())
                   && std::all_of(
                       std::next(value.rbegin()),
                       value.rend(),
                       [](value_type x) constexpr { return x == 0; });
        }
        return false;
    }

    /// Determines the number of bits required to store the contained value.
    [[nodiscard]] friend constexpr ui_t bit_width(const IntegerTraits &value)
    {
        static_assert(
            std::numeric_limits<value_type>::radix == 2,
            "value_type is not binary!");

        if (const auto size = value.size()) {
            return size * std::numeric_limits<value_type>::digits
                   - std::countl_zero(value.back());
        }
        return 0;
    }

    /// Rounds the contained value down to the next power of two.
    // clang-format off
    template<class = void> requires(Mutable)
    constexpr void bit_floor()
    // clang-format on
    {
        auto it = rbegin();
        const auto end = rend();
        if (it != end) {
            *it = std::bit_floor(*it);
            for (++it; it != end; ++it) *it = 0;
        }
    }

    /// Counts the number of bits set.
    [[nodiscard]] friend constexpr ui_t popcount(const IntegerTraits &value)
    {
        return std::accumulate(
            value.begin(),
            value.end(),
            static_cast<ui_t>(0),
            [](ui_t l, value_type r) constexpr {
                return l + std::popcount(r);
            });
    }
};

} // namespace mlir::ext::detail
