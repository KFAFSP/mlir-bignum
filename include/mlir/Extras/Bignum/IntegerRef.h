/// Implements a non-owning reference to a multiprecision integer value.
///
/// The IntegerRef type can be used in place of a concrete implementation, and
/// simplifies writing comparisons / operations on the different backends, i.e.
/// native / GMP / LLVM.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Extras/Bignum/IntegerTraits.h"

#include <initializer_list>

namespace mlir::ext {

/// Holds a non-owning reference to an immutable multi-precision integer value.
///
/// This class acts as a common argument type that can be used in place of a
/// concrete integer implementation, such as ui_t, llvm::APInt or scil::Integer,
/// as they share the same internal representation in most cases.
class IntegerRef final : public detail::IntegerTraits<IntegerRef, false> {
public:
    //===------------------------------------------------------------------===//
    // Initialization
    //===------------------------------------------------------------------===//

    /// Initializes an IntegerRef to zero.
    ///
    /// @post       `isZero()`
    /*implicit*/ constexpr IntegerRef() = default;

    /// Initializes an IntegerRef to the provided limb storage.
    ///
    /// @pre        `!size_and_sign || data[magnitude(size_and_sign) - 1] != 0`
    explicit constexpr IntegerRef(
        size_and_sign_type size_and_sign,
        pointer data)
            : m_size_and_sign(size_and_sign),
              m_data(data)
    {
        // Ensure that the representation is normalized (no leading zeros).
        assert(!size_and_sign || data[magnitude(size_and_sign) - 1] != 0);
    }

    /// Initializes an IntegerRef to the value of @p ui .
    ///
    /// @warning    Performs lifetime extension of @p ui , caller takes
    ///             responsiblity for lifetime of expression.
    ///
    /// @post       `toUnsigned() == ui`
    explicit constexpr IntegerRef(const ui_t &ui) : IntegerRef(signum(ui), &ui)
    {}
    /// Initializes an IntegerRef using the limbs @p li and @p signum .
    ///
    /// @warning    Performs lifetime extension of @p il , caller takes
    ///             responsibility for lifetime of expression.
    ///
    /// @post       `signum(*this) == signum`
    explicit constexpr IntegerRef(std::initializer_list<ui_t> il, si_t signum)
            : IntegerRef(mlir::ext::signum(signum) * il.size(), il.begin())
    {}

    /// Initializes an IntegerRef to an llvm::APInt.
    ///
    /// Since llvm::APInt instance are stored in two's complement
    /// representation, it is impossible to create a view into a negative
    /// llvm::APInt without altering its bitwise representation!
    ///
    /// @pre        `enable_apint_view`
    /// @pre        `!isSigned || !apint.isNegative()`
    // clang-format off
    template<class = void> requires(enable_apint_view)
    explicit constexpr IntegerRef(
        const llvm::APInt &apint,
        bool isSigned = false)
            : IntegerRef(
                apint.isZero() ? 0 : apint.getActiveWords(),
                apint.getRawData())
    // clang-format on
    {
        assert(
            !isSigned
            || !apint.isNegative() && "negative values cannot be viewed");
    }
    /// Initializes an IntegerRef to an llvm::APSInt.
    ///
    /// See IntegerRef(const llvm::APint &) for limitations.
    ///
    /// @pre        `enable_apint_view`
    /// @pre        `!apsint.isNegative()`
    // clang-format off
    template<class = void> requires(enable_apint_view)
    explicit constexpr IntegerRef(
        const llvm::APSInt &apsint)
            : IntegerRef(apsint, apsint.isSigned())
    // clang-format on
    {}

    //===------------------------------------------------------------------===//
    // IntegerTraits implementation
    //===------------------------------------------------------------------===//

    /// Obtains the product of signum and number of limbs.
    constexpr size_and_sign_type size_and_sign() const
    {
        return m_size_and_sign;
    }
    /// Obtains a pointer to the immutable limbs.
    constexpr pointer data() const { return m_data; }

private:
    size_and_sign_type m_size_and_sign;
    pointer m_data;
};

} // namespace mlir::ext
