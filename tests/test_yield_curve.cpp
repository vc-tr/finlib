#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/curves/yield_curve.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Flat yield curve", "[curves]") {
    Vec tenors = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
    Vec rates(6, 0.05);  // flat 5%

    YieldCurve curve(tenors, rates);

    SECTION("zero rate is flat") {
        REQUIRE_THAT(curve.zero_rate(0.5), WithinAbs(0.05, 1e-10));
        REQUIRE_THAT(curve.zero_rate(3.0), WithinAbs(0.05, 0.01));
    }

    SECTION("discount factor matches exp(-rT)") {
        REQUIRE_THAT(curve.discount(1.0), WithinAbs(std::exp(-0.05), 1e-6));
        REQUIRE_THAT(curve.discount(2.0), WithinAbs(std::exp(-0.10), 1e-6));
    }

    SECTION("forward rate equals spot rate for flat curve") {
        REQUIRE_THAT(curve.forward_rate(1.0, 2.0), WithinAbs(0.05, 0.01));
    }
}

TEST_CASE("Upward-sloping yield curve", "[curves]") {
    Vec tenors = {0.25, 0.5, 1.0, 2.0, 5.0, 10.0};
    Vec rates = {0.02, 0.025, 0.03, 0.035, 0.04, 0.045};

    YieldCurve curve(tenors, rates);

    SECTION("short rates are lower") {
        REQUIRE(curve.zero_rate(0.5) < curve.zero_rate(5.0));
    }

    SECTION("discount factors decrease with maturity") {
        REQUIRE(curve.discount(1.0) > curve.discount(5.0));
        REQUIRE(curve.discount(5.0) > curve.discount(10.0));
    }

    SECTION("forward rate is higher than spot rate") {
        Real fwd = curve.forward_rate(1.0, 2.0);
        Real spot = curve.zero_rate(1.0);
        REQUIRE(fwd > spot);
    }
}

TEST_CASE("Bootstrap from deposits", "[curves]") {
    std::vector<DepositRate> deposits = {
        {0.25, 0.02},  // 3M deposit at 2%
        {0.5, 0.025},  // 6M at 2.5%
        {1.0, 0.03}    // 1Y at 3%
    };

    std::vector<SwapRate> swaps = {
        {2.0, 0.035, 2.0},  // 2Y swap at 3.5%, semi-annual
        {5.0, 0.04, 2.0},   // 5Y swap at 4%
    };

    auto curve = bootstrap_curve(deposits, swaps);

    // Deposits should roundtrip
    for (const auto& dep : deposits) {
        Real df = curve.discount(dep.tenor);
        Real implied_simple = (1.0 / df - 1.0) / dep.tenor;
        REQUIRE_THAT(implied_simple, WithinAbs(dep.rate, 0.005));
    }

    // Curve should be upward sloping
    REQUIRE(curve.zero_rate(0.25) < curve.zero_rate(5.0));
}

TEST_CASE("Discount at t=0 is 1", "[curves]") {
    Vec tenors = {0.5, 1.0, 2.0};
    Vec rates = {0.03, 0.04, 0.05};
    YieldCurve curve(tenors, rates);

    REQUIRE_THAT(curve.discount(0.0), WithinAbs(1.0, 1e-12));
}
