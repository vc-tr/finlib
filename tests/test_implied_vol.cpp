#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/volatility/implied_vol.hpp"
#include "qe/volatility/vol_surface.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Implied vol roundtrip: price -> vol -> price", "[volatility]") {
    Real S = 100.0, K = 100.0, r = 0.05, T = 1.0;
    Real true_vol = 0.25;

    Real price = bs_call(S, K, r, true_vol, T);
    auto result = implied_vol(price, S, K, r, T, OptionType::Call);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.vol, WithinAbs(true_vol, 1e-6));
}

TEST_CASE("Implied vol for various moneyness levels", "[volatility]") {
    Real S = 100.0, r = 0.05, T = 1.0, true_vol = 0.2;

    for (Real K : {80.0, 90.0, 100.0, 110.0, 120.0}) {
        Real price = bs_call(S, K, r, true_vol, T);
        auto result = implied_vol(price, S, K, r, T, OptionType::Call);

        REQUIRE(result.converged);
        REQUIRE_THAT(result.vol, WithinAbs(true_vol, 1e-4));
    }
}

TEST_CASE("Implied vol for puts", "[volatility]") {
    Real S = 100.0, K = 95.0, r = 0.05, T = 0.5, true_vol = 0.3;

    Real price = bs_put(S, K, r, true_vol, T);
    auto result = implied_vol(price, S, K, r, T, OptionType::Put);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.vol, WithinAbs(true_vol, 1e-4));
}

TEST_CASE("Implied vol converges quickly for ATM", "[volatility]") {
    Real price = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    auto result = implied_vol(price, 100.0, 100.0, 0.05, 1.0, OptionType::Call);

    REQUIRE(result.converged);
    REQUIRE(result.iterations <= 10);  // Newton should be fast for ATM
}

TEST_CASE("Vol surface bilinear interpolation", "[volatility]") {
    Vec strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    Vec maturities = {0.25, 0.5, 1.0};

    // Flat vol surface at 20%
    Mat vols = {
        {0.25, 0.22, 0.20, 0.22, 0.25},  // 3M smile
        {0.24, 0.21, 0.20, 0.21, 0.24},  // 6M smile (flatter)
        {0.23, 0.21, 0.20, 0.21, 0.23},  // 1Y smile (flattest)
    };

    VolSurface surface(strikes, maturities, vols);

    SECTION("exact grid points") {
        REQUIRE_THAT(surface.vol(100.0, 0.25), WithinAbs(0.20, 1e-10));
        REQUIRE_THAT(surface.vol(80.0, 1.0), WithinAbs(0.23, 1e-10));
    }

    SECTION("interpolated point") {
        Real v = surface.vol(95.0, 0.75);
        REQUIRE(v > 0.19);
        REQUIRE(v < 0.22);
    }

    SECTION("smile extraction") {
        Vec smile = surface.smile(0.5);
        REQUIRE(smile.size() == 5);
        // ATM should be lowest (smile shape)
        REQUIRE(smile[2] <= smile[0]);
        REQUIRE(smile[2] <= smile[4]);
    }
}
