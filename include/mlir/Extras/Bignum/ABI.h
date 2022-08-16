/// ABI compatibility header.
///
/// This header declares some concepts related to ABI compatibility with GMP,
/// which is required for this extension to work properly.
///
/// Users of the mlir-bignum library shall not have any GMP headers in their
/// transitive includes, so no public includes of mlir-bignum may include any of
/// them. As a result, we need to define a few GMP types required for ABI
/// compatibility here, which must be matched against the target architecture
/// and library distribution manually.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <limits>

namespace mlir::ext {

/// Declarations of required GMP ABI types.
namespace gmp {

/// Type of a limb.
using limb_t = unsigned long;

/// Type holding the product of the signum and the size of a numeral.
using size_and_sign_t = int;

/// Maximum size of a numeral in limbs.
inline constexpr auto max_size = std::numeric_limits<size_and_sign_t>::max();

/// Type holding the allocated capacity of a numeral.
using capacity_t = int;

} // namespace gmp

} // namespace mlir::ext
