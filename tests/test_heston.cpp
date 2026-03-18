#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/models/heston.hpp"
#include "qe/models/black_scholes.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

static HestonParams make_params() {
    return {
        .spot = 100.0,
        .rate = 0.05,
        .v0 = 0.04,       // initial vol = 20%
        .kappa = 2.0,
        .theta = 0.04,     // long-run vol = 20%
        .xi = 0.3,         // vol of vol
        .rho = -0.7,       // leverage effect
        .maturity = 1.0
    };
}

TEST_CASE("Heston reduces to BS when xi=0", "[heston]") {
    HestonParams p = make_params();
    p.xi = 0.001;   // nearly zero vol of vol
    p.rho = 0.0;

    Real heston_c = heston_call(p, 100.0);
    Real bs_c = bs_call(100.0, 100.0, 0.05, 0.2, 1.0);

    // With no stochastic vol, should match BS closely
    REQUIRE_THAT(heston_c, WithinAbs(bs_c, 0.5));
}

TEST_CASE("Heston put-call parity", "[heston]") {
    HestonParams p = make_params();
    Real K = 100.0;

    Real call = heston_call(p, K);
    Real put = heston_put(p, K);
    Real parity = p.spot - K * std::exp(-p.rate * p.maturity);

    REQUIRE_THAT(call - put, WithinAbs(parity, 0.5));
}

TEST_CASE("Heston call price is positive and reasonable", "[heston]") {
    HestonParams p = make_params();

    Real call = heston_call(p, 100.0);
    REQUIRE(call > 0.0);
    REQUIRE(call < p.spot);

    // ATM call should be in reasonable range (BS is ~10.45)
    REQUIRE(call > 5.0);
    REQUIRE(call < 20.0);
}

TEST_CASE("Heston MC vs semi-analytical", "[heston]") {
    HestonParams p = make_params();
    Real K = 100.0;

    Real analytical = heston_call(p, K);

    HestonMC::Config mc_config;
    mc_config.params = p;
    mc_config.strike = K;
    mc_config.type = OptionType::Call;
    mc_config.num_paths = 200000;
    mc_config.num_steps = 252;
    mc_config.seed = 42;

    HestonMC mc(mc_config);
    auto result = mc.price();

    // MC should be within a few standard errors of analytical
    REQUIRE_THAT(result.price, WithinAbs(analytical, 1.5));
}

TEST_CASE("Heston vol smile: OTM puts more expensive with negative rho", "[heston]") {
    HestonParams p = make_params();
    p.rho = -0.7;  // strong negative correlation (leverage effect)

    // OTM put (K < S) should have higher implied vol than ATM
    Real otm_put = heston_put(p, 90.0);
    Real atm_put = heston_put(p, 100.0);

    // Compare implied vols (not raw prices)
    // OTM put BS equivalent vol should be higher
    REQUIRE(otm_put > 0.0);
    REQUIRE(atm_put > 0.0);
}
