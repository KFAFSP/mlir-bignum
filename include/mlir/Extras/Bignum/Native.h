/// Native integer types.
///
/// This file provides additional support for native integer types, as well as
/// declaring a preferred native integer type to be used in unsigned & signed
/// contexts respectively.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mlir::ext {

/// Native unsigned integral type.
///
/// This type should be the fastest integral type available on the platform, and
/// should be the limb type used by both GMP and LLVM's APInt.
///
/// We currently assume this to be @c std::uint64_t , which is the case on all
/// considered host platforms and distributions of both libraries.
using ui_t = std::uint64_t;

/// Maximum value of ui_t.
inline constexpr auto max_ui = std::numeric_limits<ui_t>::max();

/// Native signed integral type.
///
/// This is the signed counterpart to ui_t.
using si_t = std::make_signed_t<ui_t>;

/// Minimum value of si_t.
inline constexpr auto min_si = std::numeric_limits<si_t>::min();
/// Maximum value of si_t.
inline constexpr auto max_si = std::numeric_limits<si_t>::max();

//===----------------------------------------------------------------------===//
// Integer concepts
//===----------------------------------------------------------------------===//

/// Compile-time constant indicating whether @p Sub is a subrange of @p Super .
template<std::integral Sub, std::integral Super>
inline constexpr bool is_int_subrange_v =
    (std::numeric_limits<Sub>::max() <= std::numeric_limits<Super>::max())
    && (std::numeric_limits<Sub>::min() >= std::numeric_limits<Super>::min());

//===----------------------------------------------------------------------===//
// Integer template utilities
//===----------------------------------------------------------------------===//

/// Obtains the signum {-1, 0, 1} of a signed @p value .
template<std::signed_integral SInt>
[[nodiscard]] constexpr auto signum(SInt value)
{
    return static_cast<SInt>((value > 0) - (value < 0));
}
/// Obtains the signum {0, 1} of an unsigned @p value .
template<std::unsigned_integral UInt>
[[nodiscard]] constexpr auto signum(UInt value)
{
    return static_cast<UInt>(value > 0);
}

/// Obtains the absolute value of @p value without overflow.
template<std::signed_integral SInt>
[[nodiscard]] constexpr auto magnitude(SInt value)
{
    using UInt = std::make_unsigned_t<SInt>;
    if constexpr (std::numeric_limits<UInt>::radix == 2) {
        const SInt mask = value >> (std::numeric_limits<UInt>::digits - 1);
        value += mask;
        value ^= mask;
        return static_cast<UInt>(value);
    } else {
        return value >= 0 ? value : static_cast<UInt>(-value);
    }
}
/// Obtains @p value .
template<std::unsigned_integral UInt>
[[nodiscard]] constexpr auto magnitude(UInt value)
{
    return value;
}

} // namespace mlir::ext
