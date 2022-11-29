#include "mlir/Extras/Bignum/Integer.h"

#include <array>
#include <doctest/doctest.h>
#include <gmp.h>

using namespace llvm;
using namespace mlir::ext;
using namespace mlir::ext::detail;

// clang-format off

template<std::convertible_to<ui_t>... Limbs>
static constexpr auto make_limbs(Limbs... limbs)
{
    return std::array{static_cast<ui_t>(limbs)...};
}

template<std::size_t N>
static constexpr auto make_iota(auto... is)
{
    if constexpr (N == 0)
        return make_limbs(is...);
    else
        return make_iota<N-1>(N-1, is...);
}

//===----------------------------------------------------------------------===//
// IntegerTraits
//===----------------------------------------------------------------------===//

TEST_CASE("IntegerTraits::front()")
{
    SUBCASE("read")
    {
        CHECK(IntegerRef({2, 3}, -1).front() == 2);
    }
    SUBCASE("write")
    {
        auto limbs = make_limbs(0, 3);
        auto range = LimbRange(2, 2, limbs.data());
        range.front() = 2;
        CHECK(range.front() == 2);
        CHECK(limbs.front() == 2);
    }
}

TEST_CASE("IntegerTraits::back()")
{
    SUBCASE("read")
    {
        CHECK(IntegerRef({2, 3}, -1).back() == 3);
    }
    SUBCASE("write")
    {
        auto limbs = make_limbs(2, 0);
        auto range = LimbRange(2, 2, limbs.data());
        range.back() = 3;
        CHECK(range.back() == 3);
        CHECK(limbs.back() == 3);
    }
}

TEST_CASE("IntegerTraits::operator[]()")
{
    SUBCASE("read")
    {
        auto ref = IntegerRef({2, 3}, -1);
        CHECK(ref[0] == 2);
        CHECK(ref[1] == 3);
    }
    SUBCASE("write")
    {
        auto limbs = make_limbs(2, 3);
        auto range = LimbRange(2, 2, limbs.data());
        range[0] = 4;
        range[1] = 5;
        CHECK(range[0] == 4);
        CHECK(range[1] == 5);
        CHECK(limbs[0] == 4);
        CHECK(limbs[1] == 5);
    }
}

TEST_CASE("IntegerTraits::isNegative()")
{
    CHECK(!IntegerRef({}, 0).isNegative());
    CHECK(!IntegerRef({1}, 1).isNegative());
    CHECK(IntegerRef({1}, -1).isNegative());
}

TEST_CASE("IntegerTraits::isZero()")
{
    CHECK(IntegerRef({}, 0).isZero());
    CHECK(!IntegerRef({1}, 1).isZero());
    CHECK(!IntegerRef({1}, -1).isZero());
}

TEST_CASE("IntegerTraits::isOne()")
{
    CHECK(!IntegerRef({}, 0).isOne());
    CHECK(IntegerRef({1}, 1).isOne());
    CHECK(!IntegerRef({0, 1}, 1).isOne());
    CHECK(!IntegerRef({1}, -1).isOne());
}

TEST_CASE("IntegerTraits::isOdd()")
{
    CHECK(!IntegerRef({}, 0).isOdd());
    CHECK(IntegerRef({1}, 1).isOdd());
    CHECK(!IntegerRef({0, 1}, 1).isOdd());
    CHECK(IntegerRef({1, 1}, 1).isOdd());
}

TEST_CASE("IntegerTraits::asUnsigned()")
{
    SUBCASE("fit")
    {
        CHECK(IntegerRef({}, 0).asUnsigned() == 0);
        CHECK(IntegerRef({2}, 1).asUnsigned() == 2);
    }
    SUBCASE("overflow")
    {
        CHECK(!IntegerRef({0, 1}, 1).asUnsigned().has_value());
    }
}

TEST_CASE("IntegerTraits::asSigned()")
{
    SUBCASE("fit")
    {
        CHECK(IntegerRef({}, 0).asSigned() == 0);
        CHECK(IntegerRef({2}, 1).asSigned() == 2);
        CHECK(IntegerRef({2}, -1).asSigned() == -2);
    }
    SUBCASE("overflow")
    {
        CHECK(!IntegerRef({0, 1}, 1).asSigned().has_value());
        CHECK(!IntegerRef({magnitude(max_si) + 1}, 1).asSigned().has_value());
        CHECK(!IntegerRef({magnitude(min_si) + 1}, -1).asSigned().has_value());
    }
}

TEST_CASE("IntegerTraits::begin()")
{
    auto limbs = make_limbs(2, 3, 4);
    IntegerRef zero(0, limbs.data());
    CHECK(std::equal(zero.begin(), zero.end(), limbs.data(), limbs.data()));
    IntegerRef some(-3, limbs.data());
    CHECK(std::equal(some.begin(), some.end(), limbs.begin(), limbs.end()));
}

TEST_CASE("IntegerTraits::rbegin()")
{
    auto limbs = make_limbs(2, 3, 4);
    IntegerRef zero(0, limbs.data());
    CHECK(std::equal(zero.rbegin(), zero.rend(), limbs.data(), limbs.data()));
    IntegerRef some(-3, limbs.data());
    CHECK(std::equal(some.rbegin(), some.rend(), limbs.rbegin(), limbs.rend()));
}

TEST_CASE("signum(const IntegerTraits &)")
{
    CHECK(signum(IntegerRef({}, 0)) == 0);
    CHECK(signum(IntegerRef({2}, 1)) == 1);
    CHECK(signum(IntegerRef({2}, -1)) == -1);
}

TEST_CASE("IntegerTraits::operator<=>()")
{
    SUBCASE("native")
    {
        CHECK(std::is_eq(IntegerRef({}, 0) <=> 0));
        CHECK(std::is_gt(IntegerRef({1}, 1) <=> 0));
        CHECK(std::is_lt(IntegerRef({1}, -1) <=> 0));

        CHECK(std::is_eq(IntegerRef({15}, 1) <=> 15));
        CHECK(std::is_gt(IntegerRef({15}, 1) <=> 14));
        CHECK(std::is_lt(IntegerRef({15}, -1) <=> -14));
    }
    SUBCASE("IntegerRef")
    {
        CHECK(std::is_eq(IntegerRef({}, 0) <=> IntegerRef({}, 0)));
        CHECK(std::is_gt(IntegerRef({1}, 1) <=> IntegerRef({}, 0)));
        CHECK(std::is_lt(IntegerRef({1}, -1) <=> IntegerRef({}, 0)));

        CHECK(std::is_eq(IntegerRef({15}, 1) <=> IntegerRef({15}, 1)));
        CHECK(std::is_gt(IntegerRef({15}, 1) <=> IntegerRef({14}, 1)));
        CHECK(std::is_lt(IntegerRef({15}, -1) <=> IntegerRef({14}, -1)));

        CHECK(std::is_lt(IntegerRef({1, 2}, -1) <=> IntegerRef({1}, -1)));
        CHECK(std::is_gt(IntegerRef({1, 2}, 1) <=> IntegerRef({1}, 1)));
        CHECK(std::is_lt(IntegerRef({1, 1}, 1) <=> IntegerRef({1, 2}, 1)));
    }
    SUBCASE("APSInt")
    {
        const auto make_apsint = [](ui_t x) {
            return APSInt(APInt(64, x, true), false);
        };

        CHECK(std::is_eq(IntegerRef({}, 0) <=> make_apsint(0)));
        CHECK(std::is_gt(IntegerRef({1}, 1) <=> make_apsint(0)));
        CHECK(std::is_lt(IntegerRef({1}, -1) <=> make_apsint(0)));

        CHECK(std::is_eq(IntegerRef({15}, 1) <=> make_apsint(15)));
        CHECK(std::is_gt(IntegerRef({15}, 1) <=> make_apsint(14)));
        CHECK(std::is_lt(IntegerRef({15}, -1) <=> make_apsint(-14)));

        CHECK(std::is_lt(IntegerRef({1, 2}, -1) <=> make_apsint(-1)));
        CHECK(std::is_gt(IntegerRef({1, 2}, 1) <=> make_apsint(1)));
        CHECK(std::is_lt(
            IntegerRef({1, 1}, 1) <=> APSInt(APInt(128, {1, 2}), false)));
    }
}

TEST_CASE("IntegerTraits::toString(...)")
{
    const auto stringify = [](std::initializer_list<ui_t> il, si_t signum) {
        llvm::SmallString<32> buffer;
        IntegerRef(il, signum).toString(buffer);
        return static_cast<std::string>(buffer);
    };

    CHECK(stringify({}, 0) == "0");
    CHECK(stringify({1}, 1) == "1");
    CHECK(stringify({1}, -1) == "-1");
    CHECK(stringify({1, 2}, 1) == "36893488147419103233");
}

TEST_CASE("IntegerTraits::has_single_bit()")
{
    CHECK(!has_single_bit(IntegerRef({}, 0)));
    CHECK(has_single_bit(IntegerRef({1}, 1)));
    CHECK(!has_single_bit(IntegerRef({1, 2}, 1)));
    CHECK(has_single_bit(IntegerRef({0, 2}, -1)));
}

TEST_CASE("bit_width(const IntegerTraits &)")
{
    CHECK(bit_width(IntegerRef({}, 0)) == 0);
    CHECK(bit_width(IntegerRef({1}, 1)) == 1);
    CHECK(bit_width(IntegerRef({1, 2}, 1)) == 66);
    CHECK(bit_width(IntegerRef({0, 2}, -1)) == 66);
}

TEST_CASE("IntegerTraits::bit_floor()")
{
    auto limbs = make_limbs(2, 3);
    LimbRange(2, 2, limbs.data()).bit_floor();
    CHECK(limbs[0] == 0);
    CHECK(limbs[1] == 2);
}

TEST_CASE("popcount(const IntegerTraits &)")
{
    CHECK(popcount(IntegerRef({}, 0)) == 0);
    CHECK(popcount(IntegerRef({1}, 1)) == 1);
    CHECK(popcount(IntegerRef({1, 2}, -1)) == 2);
}

//===----------------------------------------------------------------------===//
// IntegerRef
//===----------------------------------------------------------------------===//

TEST_CASE("IntegerRef::IntegerRef(const ui_t &)")
{
    CHECK(IntegerRef(ui_t(4)) == 4);
}

TEST_CASE("IntegerRef::IntegerRef(const llvm::APInt &, bool)")
{
    CHECK(IntegerRef(APInt(64, 0, true), true) == 0);
    CHECK(IntegerRef(APInt(64, 1, true), true) == 1);
    CHECK(IntegerRef(APInt(64, -1, true), false) == max_ui);
}

//===----------------------------------------------------------------------===//
// Integer
//===----------------------------------------------------------------------===//

TEST_CASE("Integer::Integer()")
{
    struct mpz_struct {
        int c;
        int s;
        void* d;
    };

    Integer integer{};
    auto alias = reinterpret_cast<mpz_struct*>(&integer);

    CHECK(alias->c == Integer::inline_capacity);
    CHECK(alias->s == 0);
    CHECK(alias->d == integer.data());
}

const Integer zero;
constexpr auto small_limbs = make_iota<Integer::inline_capacity>();
const IntegerRef small_ref(small_limbs.size(), small_limbs.data());
constexpr auto big_limbs = make_iota<Integer::inline_capacity+1>();
const IntegerRef big_ref(big_limbs.size(), big_limbs.data());

TEST_CASE("Integer::exp10(ui_t)")
{
    CHECK(Integer::exp10(0) == 1);
    CHECK(Integer::exp10(1) == 10);
    CHECK(Integer::exp10(20) > max_ui);
}

TEST_CASE("Integer::assign(std::integral auto)")
{
    Integer integer{};

    integer.assign(4U);
    CHECK(integer == 4U);

    integer.assign(-3L);
    CHECK(integer == -3L);
}

TEST_CASE("Integer::assign(IntegerRef value)")
{
    Integer integer{};

    integer.assign(small_ref);
    CHECK(integer == small_ref);

    integer.assign(big_ref);
    CHECK(integer == big_ref);
}

TEST_CASE("Integer::assign(const llvm::APInt &, bool)")
{
    Integer integer{};

    integer.assign(llvm::APInt(64U, -1, true), true);
    CHECK(integer == -1);

    integer.assign(llvm::APInt(64U, -1, false), false);
    CHECK(integer == max_ui);
}

TEST_CASE("swap(Integer &, Integer &)")
{
    Integer int1(small_ref), int2(big_ref);
    swap(int1, int2);

    CHECK(int1 == big_ref);
    CHECK(int2 == small_ref);
}

TEST_CASE("Integer::Integer(Integer &&)")
{
    Integer int1(big_ref);
    Integer int2(std::move(int1));

    CHECK(int1.isZero());
    CHECK(int2 == big_ref);
}

TEST_CASE("Integer::clear()")
{
    Integer integer(big_ref);

    integer.clear();
    CHECK(integer.isZero());
    CHECK(integer.capacity() >= 2);
}

TEST_CASE("Integer::reset()")
{
    Integer integer(big_ref);

    integer.reset();
    CHECK(integer.isZero());
    CHECK(integer.capacity() == Integer::inline_capacity);
}

TEST_CASE("Integer::shrink_to_fit()")
{
    Integer integer(big_ref);

    integer.assign(4);
    integer.shrink_to_fit();
    CHECK(integer.capacity() == std::max(Integer::inline_capacity, integer.size()));
}

TEST_CASE("Integer::reserve(size_type)")
{
    Integer integer{};

    integer.reserve(5);
    CHECK(integer.capacity() >= 5);
}

TEST_CASE("Integer::resize(size_type)")
{
    Integer integer(big_ref);

    CHECK(!integer.resize(1));
    CHECK(integer.isZero());

    integer.assign(1);
    CHECK(integer.resize(1));
    CHECK(integer == 1);
}

TEST_CASE("Integer::parse(const char*, unsigned)")
{
    CHECK(Integer::parse("-1234") == -1234);
    CHECK(Integer::parse("0xBEEF") == 0xBEEF);
    CHECK(Integer::parse("0B001001") == 0B001001);
    CHECK(Integer::parse("077") == 077);
}

TEST_CASE("Integer::reduce(IntegerRef)")
{
    Integer integer(39*27*6*3);

    integer.reduce(39);
    CHECK(integer == 27*6*3);

    integer.reduce(27);
    CHECK(integer == 6*3);

    integer.reduce(3);
    CHECK(integer == 6);
}

TEST_CASE("Integer::operator+=(IntegerRef)")
{
    Integer integer(IntegerRef({1, 2}, 1));
    integer += IntegerRef({2, 1}, 1);

    CHECK(integer == IntegerRef({3, 3}, 1));
}

TEST_CASE("Integer::operator-=(IntegerRef)")
{
    Integer integer(IntegerRef({2, 1}, 1));
    integer -= IntegerRef({1, 2}, 1);

    CHECK(integer == IntegerRef({max_ui}, -1));
}

TEST_CASE("Integer::operator*=(IntegerRef)")
{
    Integer integer(IntegerRef({1, 2}, 1));
    integer *= IntegerRef({2}, -1);

    CHECK(integer == IntegerRef({2, 4}, -1));
}

TEST_CASE("Integer::operator/=(IntegerRef)")
{
    Integer integer(17);
    integer /= 2;

    CHECK(integer == 8);

    integer = -17;
    integer /= 2;
    CHECK(integer == -8);
}

TEST_CASE("Integer::operator%=(IntegerRef)")
{
    Integer integer(17);
    integer %= 3;

    CHECK(integer == 2);
}

TEST_CASE("gcd(IntegerRef, IntegerRef)")
{
    CHECK(gcd(IntegerRef(21), IntegerRef(49)) == 7);
}
