/// ABI compatibility checks.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/Extras/Bignum/ABI.h"

#include <gmp.h>
#include <llvm/ADT/APInt.h>

using namespace mlir::ext;

//===----------------------------------------------------------------------===//
// GMP ABI checks.
//===----------------------------------------------------------------------===//

// The correct limb type must have been selected.
static_assert(
    std::same_as<mp_limb_t, gmp::limb_t>,
    "mlir::ext::gmp::limb_t does not match mp_limb_t!");

namespace {

/// Type storing an integer numeral.
struct mpz_struct {
    /// Capacity of the storage pointed to by data in limbs.
    gmp::capacity_t capacity;
    /// Product of signum and size in limbs.
    gmp::size_and_sign_t size_and_sign;
    /// Pointer to the least-significant limb.
    gmp::limb_t* data;
};

} // namespace

// The resulting mpz_struct type must be layout compatible.
static_assert(
    alignof(mpz_struct) == alignof(__mpz_struct)
        && sizeof(mpz_struct) == sizeof(__mpz_struct),
    "mlir::ext::gmp::mpz_struct is not layout compatible to __mpz_struct!");

// The involved types must be trivial.
static_assert(
    std::is_trivial_v<gmp::size_and_sign_t>,
    "mlir::ext::gmp::size_and_sign_t is expected to be a trivial type!");
static_assert(
    std::is_trivial_v<gmp::capacity_t>,
    "mlir::ext::gmp::capacity_t is expected to be a trivial type!");
static_assert(
    std::is_trivial_v<gmp::limb_t>,
    "mlir::ext::gmp::limb_t is expected to be a trivial type!");
