/// Implements a mutable reference to a limb range with container semantics.
///
/// This base class provides the container-like interface for the Integer type,
/// and provides operations that mutate the range without requiring allocation.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Extras/Bignum/IntegerRef.h"

namespace mlir::ext::detail {

/// Base class that stores a GMP ABI compatible mutable limb range.
class LimbRange : public IntegerTraits<LimbRange, true> {
public:
    //===------------------------------------------------------------------===//
    // Initialization
    //===------------------------------------------------------------------===//

    /// Initializes a LimbRange to zero (empty).
    ///
    /// @post       `isZero()`
    constexpr LimbRange() = default;

    /// Initializes a LimbRange with pre-allocated storage.
    constexpr LimbRange(
        size_type capacity,
        size_and_sign_type size_and_sign,
        pointer data)
            : m_capacity(capacity),
              m_size_and_sign(size_and_sign),
              m_data(data)
    {}

    //===------------------------------------------------------------------===//
    // Implicit conversion
    //===------------------------------------------------------------------===//

    /// Obtains a non-owning reference to the stored value.
    constexpr operator IntegerRef() const
    {
        return IntegerRef(size_and_sign(), data());
    }

    //===------------------------------------------------------------------===//
    // Disable copy & move (container must overwrite).
    //===------------------------------------------------------------------===//

    /*implicit*/ LimbRange(const LimbRange &) = delete;
    LimbRange &operator=(const LimbRange &) = delete;

    /*implicit*/ LimbRange(LimbRange &&) = delete;
    LimbRange &operator=(LimbRange &&) = delete;

    //===------------------------------------------------------------------===//
    // IntegerTraits implementation
    //===------------------------------------------------------------------===//

    /// Obtains the product of signum and number of limbs.
    size_and_sign_type size_and_sign() const { return m_size_and_sign; }
    /// Obtains a pointer to the limbs.
    pointer data() { return m_data; }
    /// Obtains a pointer to the immutable limbs.
    const_pointer data() const { return m_data; }

    //===------------------------------------------------------------------===//
    // Observers
    //===------------------------------------------------------------------===//

    /// Gets the available capacity in number of limbs.
    constexpr size_type capacity() const { return magnitude(m_capacity); }

    //===------------------------------------------------------------------===//
    // Mutating interface
    //===------------------------------------------------------------------===//

    /// Clears the contained value, setting it to 0.
    ///
    /// @post       `isZero()`
    constexpr void clear() { m_size_and_sign = 0; }

    /// Flips the sign (computing `neg` in-place).
    constexpr void negate() { m_size_and_sign = -m_size_and_sign; }
    /// Removes any negative sign (computing `abs` in-place).
    ///
    /// @post       `!isNegative()`
    constexpr void dropSign() { m_size_and_sign = std::abs(m_size_and_sign); }

protected:
    // NOTE: The exact types, layout and alignment of these fields is required
    //       to be preserved to achieve GMP ABI compatibility!
    gmp::capacity_t m_capacity;
    size_and_sign_type m_size_and_sign;
    mutable pointer m_data;
};

} // namespace mlir::ext::detail
