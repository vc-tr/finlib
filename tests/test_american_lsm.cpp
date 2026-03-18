#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/instruments/american.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("American put price >= European put price", "[american]") {
    AmericanPricer::Config config = {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Put,
        .num_paths = 200000,
        .num_steps = 50,
        .poly_degree = 3,
        .seed = 42
    };

    AmericanPricer pricer(config);
    auto result = pricer.price();

    Real euro_put = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);

    // American put must be worth at least as much as European put
    REQUIRE(result.price >= euro_put - 0.5);  // allow MC noise
    REQUIRE(result.early_exercise_premium > -0.5);
}

TEST_CASE("American put has positive early exercise premium", "[american]") {
    // Deep ITM put: early exercise premium should be significant
    AmericanPricer::Config config = {
        .spot = 80.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Put,
        .num_paths = 200000,
        .num_steps = 50,
        .poly_degree = 3,
        .seed = 42
    };

    AmericanPricer pricer(config);
    auto result = pricer.price();

    // Deep ITM put should have meaningful early exercise premium
    REQUIRE(result.early_exercise_premium > 0.0);
    REQUIRE(result.price > result.european_price);
}

TEST_CASE("American call with no dividends equals European call", "[american]") {
    // Without dividends, early exercise of a call is never optimal
    AmericanPricer::Config config = {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Call,
        .num_paths = 200000,
        .num_steps = 50,
        .poly_degree = 3,
        .seed = 42
    };

    AmericanPricer pricer(config);
    auto result = pricer.price();

    Real euro_call = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // Should be approximately equal (no early exercise premium for calls)
    REQUIRE_THAT(result.price, WithinAbs(euro_call, 0.5));
}

TEST_CASE("American put price is reasonable for ATM", "[american]") {
    AmericanPricer::Config config = {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.3,
        .maturity = 1.0,
        .type = OptionType::Put,
        .num_paths = 200000,
        .num_steps = 50,
        .poly_degree = 3,
        .seed = 123
    };

    AmericanPricer pricer(config);
    auto result = pricer.price();

    // Price should be between European put and intrinsic value + some premium
    Real euro_put = bs_put(100.0, 100.0, 0.05, 0.3, 1.0);
    REQUIRE(result.price >= euro_put - 0.5);
    REQUIRE(result.price < 30.0);  // sanity check
}

TEST_CASE("American put price increases with volatility", "[american]") {
    auto price_with_vol = [](Real sigma) {
        AmericanPricer::Config config = {
            .spot = 100.0,
            .strike = 100.0,
            .rate = 0.05,
            .sigma = sigma,
            .maturity = 1.0,
            .type = OptionType::Put,
            .num_paths = 200000,
            .num_steps = 50,
            .poly_degree = 3,
            .seed = 42
        };
        return AmericanPricer(config).price().price;
    };

    Real p_low = price_with_vol(0.15);
    Real p_high = price_with_vol(0.35);

    REQUIRE(p_high > p_low);
}

TEST_CASE("LSM result has valid statistics", "[american]") {
    AmericanPricer::Config config = {
        .spot = 100.0,
        .strike = 100.0,
        .rate = 0.05,
        .sigma = 0.2,
        .maturity = 1.0,
        .type = OptionType::Put,
        .num_paths = 50000,
        .num_steps = 50,
        .poly_degree = 3,
        .seed = 42
    };

    AmericanPricer pricer(config);
    auto result = pricer.price();

    REQUIRE(result.price > 0.0);
    REQUIRE(result.std_error > 0.0);
    REQUIRE(result.std_error < 1.0);  // SE should be small
    REQUIRE(result.elapsed_ms > 0.0);
    REQUIRE(result.european_price > 0.0);
}
