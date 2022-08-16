/// LLVM arbitrary precision number types.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "mlir/Extras/Bignum/ABI.h"
#include "mlir/Extras/Bignum/Native.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>

namespace mlir::ext {

/// Compile-time constant indicating whether the limb types of LLVM's APInt and
/// GMP's mpz are the same.
inline constexpr auto enable_apint_view =
    std::same_as<gmp::limb_t, llvm::APInt::WordType>;

/// Obtains the signum {-1, 0, 1} of @p value .
[[nodiscard]] inline si_t signum(const llvm::APSInt &apsint)
{
    return apsint.isNegative() ? si_t(-1) : si_t(!apsint.isZero());
}
/// Obtains the signum {-1, 0, 1} of @p value .
[[nodiscard]] inline si_t signum(const llvm::APInt &apint, bool isSigned = true)
{
    return apint.isNegative() && isSigned ? si_t(-1) : si_t(!apint.isZero());
}

} // namespace mlir::ext
