/// Implements the multiprecision integer type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#include "mlir/Extras/Bignum/Integer.h"

#include <gmp.h>
#include <llvm/Support/Debug.h>

using namespace llvm;
using namespace mlir::ext;
using namespace mlir::ext::detail;

#define DEBUG_TYPE "Integer"

// NOTE: Ignore that the value range of size_and_sign_t is known to be less than
//       that of size_t.
#pragma GCC diagnostic ignored "-Wtautological-constant-out-of-range-compare"

//===----------------------------------------------------------------------===//
// Custom GMP allocator
//===----------------------------------------------------------------------===//

/// Given the storage block @p ptr , obtains the control marker pointer.
[[nodiscard]] static void* getControlMarker(void* ptr)
{
    return static_cast<char*>(ptr) - sizeof(void*);
}
/// Given the control marker @p ptr , obtains the storage block pointer.
[[nodiscard]] static void* getStorageBlock(void* ptr)
{
    return static_cast<char*>(ptr) + sizeof(void*);
}
/// Given the control marker @p ptr , determines if it marks inline storage.
[[nodiscard]] static bool isInlineStorage(void* ptr)
{
    return static_cast<void**>(ptr)[0] == getStorageBlock(ptr);
}

/// Maximum number of bytes allocatable by the custom allocator.
inline constexpr auto max_alloc =
    std::numeric_limits<std::size_t>::max() - sizeof(void*);

/// Allocates a storage block of @p size bytes.
[[nodiscard]] static void* customAlloc(std::size_t size)
{
    // NOTE: Since we don't have access to the issuing integer, there is no way
    //       to return internal storage here, and thus this function will always
    //       allocate.
    LLVM_DEBUG(if (size == sizeof(gmp::limb_t)) dbgs()
                   << "Allocating single limb storage?\n";);

    // Ensure there will be space for the control marker.
    assert(size <= max_alloc && "allocation too large");
    size += sizeof(void*);

    // Delegate to the system default allocator.
    auto result = std::malloc(size);
    static_cast<void**>(result)[0] = nullptr;
    return getStorageBlock(result);
}
/// Resizes a storage block @p old of @p old_size bytes to @p new_size bytes.
[[nodiscard]] static void*
customRealloc(void* old, std::size_t old_size, std::size_t new_size)
{
    // NOTE: Since we don't have access to the issuing integer, there is no way
    //       to return internal storage here, and thus this function will always
    //       allocate.
    LLVM_DEBUG(if (new_size == sizeof(gmp::limb_t)) dbgs()
                   << "Allocating single limb storage?\n";);

    // Turn storage block into control marker pointer.
    old = getControlMarker(old);

    if (isInlineStorage(old)) {
        // Do nothing on deallocation.
        if (new_size == 0) return old;

        // Allocate external storage to grow into, and copy into that.
        assert(new_size > old_size);
        auto result = customAlloc(new_size);
        std::memcpy(result, getStorageBlock(old), old_size);
        return result;
    }

    // Ensure there will be space for the control marker.
    assert(new_size < max_alloc && "allocation too large");
    new_size += sizeof(void*);

    // Delegate to the system default allocator.
    auto result = std::realloc(old, new_size);
    assert(static_cast<void**>(result)[0] == nullptr);
    return getStorageBlock(result);
}
/// Deallocates a storage block @p ptr of @p size bytes.
static void customFree(void* ptr, std::size_t)
{
    // NOTE: This function is never called from our interface, since we handle
    //       those differently. However, it can still be called from other
    //       sources, e.g. temporaries and strings.

    // Turn storage block into control marker pointer.
    ptr = getControlMarker(ptr);
    assert(!isInlineStorage(ptr) && "expected external storage");

    // Delegate to the system default allocator.
    std::free(ptr);
}

void Integer::attachGMPAllocator()
{
    static bool attached = false;
    if (!attached) {
        LLVM_DEBUG(dbgs() << "Attaching custom GMP allocator.\n");
        mp_set_memory_functions(customAlloc, customRealloc, customFree);
        attached = true;
    }
}

Integer::pointer Integer::allocate(size_type words)
{
    assert(words <= gmp::max_size && "allocation too large");
    return static_cast<pointer>(customAlloc(words * sizeof(gmp::limb_t)));
}

//===----------------------------------------------------------------------===//
// GMP aliasing
//===----------------------------------------------------------------------===//

/// Initializes an __mpz_struct that can be converted into a mpz_srcptr.
[[nodiscard]] static inline __mpz_struct make_ro_mpz(IntegerRef value)
{
    return {0, value.size_and_sign(), const_cast<gmp::limb_t*>(value.data())};
}
/// Obtains an mpz_ptr from an ABI compatible Integer.
[[nodiscard]] static inline mpz_ptr get_impl(Integer &impl)
{
    return reinterpret_cast<mpz_ptr>(&impl);
}

//===----------------------------------------------------------------------===//
// IntegerTraits
//===----------------------------------------------------------------------===//

void mlir::ext::detail::toStringImpl(
    llvm::SmallVectorImpl<char> &buffer,
    gmp::size_and_sign_t size_and_sign,
    const gmp::limb_t* data,
    int base)
{
    assert((base >= 2 && base <= 62) || (base <= -2 && base >= -36));

    const __mpz_struct value{0, size_and_sign, const_cast<gmp::limb_t*>(data)};

    // Make space at the end of the buffer.
    const auto oldSize = buffer.size();
    buffer.resize_for_overwrite(oldSize + mpz_sizeinbase(&value, base) + 2);

    // Write directly into the buffer.
    mpz_get_str(buffer.data() + oldSize, base, &value);

    // Trim NULL terminator and trailing.
    buffer.truncate(
        std::find(buffer.data() + oldSize, buffer.end(), '\0') - buffer.data());
}

//===----------------------------------------------------------------------===//
// Integer
//===----------------------------------------------------------------------===//

void Integer::free()
{
    assert(isAllocated() && "no storage to free");

    // Delegate to system default allocator.
    std::free(getControlMarker(m_data));
    m_capacity = 0;
}

bool Integer::realloc(size_type limbs)
{
    const auto oldSizeAndSign = size_and_sign();
    mpz_realloc(get_impl(*this), limbs);
    return size_and_sign() == oldSizeAndSign;
}

Integer Integer::exp10(ui_t exp)
{
    Integer result;
    mpz_ui_pow_ui(get_impl(result), 10, exp);
    return result;
}

std::optional<Integer> Integer::parse(const char* cstr, unsigned base)
{
    assert(base == 0 || (base >= 2 && base <= 62));

    Integer result;
    if (mpz_set_str(get_impl(result), cstr, base)) return std::nullopt;
    return std::move(result);
}

void Integer::reduce(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_divexact(get_impl(*this), get_impl(*this), &rhs_ro);
}

Integer &Integer::operator+=(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_add(get_impl(*this), get_impl(*this), &rhs_ro);
    return *this;
}

Integer &Integer::operator-=(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_sub(get_impl(*this), get_impl(*this), &rhs_ro);
    return *this;
}

Integer &Integer::operator*=(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_mul(get_impl(*this), get_impl(*this), &rhs_ro);
    return *this;
}

Integer &Integer::operator/=(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_tdiv_q(get_impl(*this), get_impl(*this), &rhs_ro);
    return *this;
}

Integer &Integer::operator%=(IntegerRef rhs)
{
    const auto rhs_ro = make_ro_mpz(rhs);
    mpz_tdiv_r(get_impl(*this), get_impl(*this), &rhs_ro);
    return *this;
}

Integer mlir::ext::gcd(IntegerRef lhs, IntegerRef rhs)
{
    const auto lhs_ro = make_ro_mpz(lhs);
    const auto rhs_ro = make_ro_mpz(rhs);
    Integer result;
    mpz_gcd(get_impl(result), &lhs_ro, &rhs_ro);
    return result;
}
