#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/instruments/exotic.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

static const ExoticPricer::Config base_config = {
    .spot = 100.0,
    .rate = 0.05,
    .sigma = 0.2,
    .maturity = 1.0,
    .num_paths = 200000,
    .num_steps = 252,
    .seed = 42
};

// ============================================================================
// Barrier Options
// ============================================================================

TEST_CASE("Down-and-out call: price < vanilla call", "[exotic][barrier]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_barrier(100.0, OptionType::Call,
                                        80.0, BarrierType::DownAndOut);
    Real vanilla = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // Barrier option is always worth less (some paths get knocked out)
    REQUIRE(result.price < vanilla);
    REQUIRE(result.price > 0.0);
    REQUIRE(result.knock_pct >= 0.0);
    REQUIRE(result.knock_pct <= 1.0);
}

TEST_CASE("In + Out = Vanilla (barrier parity)", "[exotic][barrier]") {
    ExoticPricer pricer(base_config);
    Real barrier = 80.0;

    auto out = pricer.price_barrier(100.0, OptionType::Call,
                                     barrier, BarrierType::DownAndOut);
    auto in = pricer.price_barrier(100.0, OptionType::Call,
                                    barrier, BarrierType::DownAndIn);

    Real vanilla = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // In + Out should approximately equal vanilla
    REQUIRE_THAT(in.price + out.price, WithinAbs(vanilla, 0.5));
}

TEST_CASE("Up-and-out put: knocked out by high prices", "[exotic][barrier]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_barrier(100.0, OptionType::Put,
                                        130.0, BarrierType::UpAndOut);
    Real vanilla_put = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE(result.price < vanilla_put);
    REQUIRE(result.price >= 0.0);
}

// ============================================================================
// Asian Options
// ============================================================================

TEST_CASE("Geometric Asian call matches closed form", "[exotic][asian]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_asian(100.0, OptionType::Call, AverageType::Geometric);
    Real analytical = ExoticPricer::geometric_asian_call(
        100.0, 100.0, 0.05, 0.2, 1.0, 252);

    // MC should match closed-form within a few std errors
    REQUIRE_THAT(result.price, WithinAbs(analytical, 0.5));
}

TEST_CASE("Asian call is cheaper than vanilla call", "[exotic][asian]") {
    ExoticPricer pricer(base_config);

    auto arith = pricer.price_asian(100.0, OptionType::Call, AverageType::Arithmetic);
    auto geom = pricer.price_asian(100.0, OptionType::Call, AverageType::Geometric);
    Real vanilla = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // Averaging reduces volatility → cheaper option
    REQUIRE(arith.price < vanilla);
    REQUIRE(geom.price < vanilla);

    // Arithmetic average >= Geometric average (AM-GM inequality)
    REQUIRE(arith.price >= geom.price - 0.5);  // allow MC noise
}

TEST_CASE("Asian put has positive price", "[exotic][asian]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_asian(100.0, OptionType::Put, AverageType::Arithmetic);
    REQUIRE(result.price > 0.0);
}

// ============================================================================
// Lookback Options
// ============================================================================

TEST_CASE("Lookback call is more expensive than vanilla call", "[exotic][lookback]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_lookback(OptionType::Call);
    Real vanilla = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // Lookback always has better strike → more valuable
    REQUIRE(result.price > vanilla);
}

TEST_CASE("Lookback put is more expensive than vanilla put", "[exotic][lookback]") {
    ExoticPricer pricer(base_config);

    auto result = pricer.price_lookback(OptionType::Put);
    Real vanilla = bs_put(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE(result.price > vanilla);
}

TEST_CASE("Lookback prices are positive", "[exotic][lookback]") {
    ExoticPricer pricer(base_config);

    auto call = pricer.price_lookback(OptionType::Call);
    auto put = pricer.price_lookback(OptionType::Put);

    // Lookback call: S(T) - min(S) >= 0 always
    REQUIRE(call.price > 0.0);
    // Lookback put: max(S) - S(T) >= 0 always
    REQUIRE(put.price > 0.0);
}

// ============================================================================
// Basket Options
// ============================================================================

TEST_CASE("Basket call with single asset matches vanilla", "[exotic][basket]") {
    ExoticPricer::Config cfg = base_config;
    cfg.num_steps = 1;  // European-style
    ExoticPricer pricer(cfg);

    ExoticPricer::BasketConfig basket;
    basket.spots = {100.0};
    basket.sigmas = {0.2};
    basket.weights = {1.0};
    basket.correlation = {{1.0}};

    auto result = pricer.price_basket(basket, 100.0, OptionType::Call);
    Real vanilla = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    REQUIRE_THAT(result.price, WithinAbs(vanilla, 0.5));
}

TEST_CASE("Basket call with correlated assets", "[exotic][basket]") {
    ExoticPricer::Config cfg = base_config;
    cfg.num_steps = 1;
    ExoticPricer pricer(cfg);

    ExoticPricer::BasketConfig basket;
    basket.spots = {100.0, 100.0};
    basket.sigmas = {0.2, 0.3};
    basket.weights = {0.5, 0.5};
    basket.correlation = {{1.0, 0.5}, {0.5, 1.0}};

    auto result = pricer.price_basket(basket, 100.0, OptionType::Call);

    // Basket call should be cheaper than max of individual calls (diversification)
    Real c1 = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);
    Real c2 = bs_call(100.0, 100.0, 0.05, 0.3, 1.0);

    REQUIRE(result.price > 0.0);
    REQUIRE(result.price < std::max(c1, c2));
}

TEST_CASE("Basket call: higher correlation → higher price", "[exotic][basket]") {
    ExoticPricer::Config cfg = base_config;
    cfg.num_steps = 1;
    cfg.num_paths = 300000;
    ExoticPricer pricer(cfg);

    auto price_with_corr = [&](Real rho) {
        ExoticPricer::BasketConfig basket;
        basket.spots = {100.0, 100.0};
        basket.sigmas = {0.2, 0.2};
        basket.weights = {0.5, 0.5};
        basket.correlation = {{1.0, rho}, {rho, 1.0}};
        return pricer.price_basket(basket, 100.0, OptionType::Call);
    };

    auto low_corr = price_with_corr(0.1);
    auto high_corr = price_with_corr(0.9);

    // Higher correlation → less diversification → higher basket vol → higher price
    REQUIRE(high_corr.price > low_corr.price);
}
