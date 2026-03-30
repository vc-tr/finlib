#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/greeks/finite_difference.hpp"
#include "qe/greeks/pathwise.hpp"
#include "qe/greeks/likelihood_ratio.hpp"
#include "qe/models/black_scholes.hpp"
#include "qe/instruments/payoff.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

static const MCEngine::Config base_config = {
    .spot = 100.0,
    .rate = 0.05,
    .sigma = 0.2,
    .maturity = 1.0,
    .num_paths = 500000,
    .num_steps = 1,
    .seed = 42
};

static const Real K = 100.0;

// Analytical reference values
static Real an_delta() { return bs_delta(100.0, K, 0.05, 0.2, 1.0, OptionType::Call); }
static Real an_vega() { return bs_vega(100.0, K, 0.05, 0.2, 1.0); }
static Real an_rho() { return bs_rho(100.0, K, 0.05, 0.2, 1.0, OptionType::Call); }

// ============================================================================
// Finite Difference Greeks
// ============================================================================

TEST_CASE("FD delta matches analytical", "[greeks][fd]") {
    VanillaPayoff payoff(K, OptionType::Call);
    auto greeks = mc_fd_greeks(base_config, payoff);
    REQUIRE_THAT(greeks.delta, WithinAbs(an_delta(), 0.05));
}

TEST_CASE("FD vega matches analytical", "[greeks][fd]") {
    VanillaPayoff payoff(K, OptionType::Call);
    auto greeks = mc_fd_greeks(base_config, payoff);
    REQUIRE_THAT(greeks.vega, WithinAbs(an_vega(), 2.0));
}

TEST_CASE("FD theta is negative for call", "[greeks][fd]") {
    VanillaPayoff payoff(K, OptionType::Call);
    auto greeks = mc_fd_greeks(base_config, payoff);
    REQUIRE(greeks.theta < 0.0);
}

// ============================================================================
// Pathwise (IPA) Greeks
// ============================================================================

TEST_CASE("IPA delta matches analytical", "[greeks][ipa]") {
    auto greeks = mc_ipa_greeks(base_config, K, OptionType::Call);
    REQUIRE_THAT(greeks.delta, WithinAbs(an_delta(), 0.02));
}

TEST_CASE("IPA vega matches analytical", "[greeks][ipa]") {
    auto greeks = mc_ipa_greeks(base_config, K, OptionType::Call);
    REQUIRE_THAT(greeks.vega, WithinAbs(an_vega(), 1.5));
}

TEST_CASE("IPA rho matches analytical", "[greeks][ipa]") {
    auto greeks = mc_ipa_greeks(base_config, K, OptionType::Call);
    REQUIRE_THAT(greeks.rho, WithinAbs(an_rho(), 2.0));
}

// ============================================================================
// Likelihood Ratio Greeks
// ============================================================================

TEST_CASE("LR delta matches analytical", "[greeks][lr]") {
    VanillaPayoff payoff(K, OptionType::Call);
    auto greeks = mc_lr_greeks(base_config, payoff);
    REQUIRE_THAT(greeks.delta, WithinAbs(an_delta(), 0.05));
}

TEST_CASE("LR works for digital options (unlike IPA)", "[greeks][lr]") {
    // Digital payoff: discontinuous, so IPA doesn't work but LR does
    DigitalPayoff payoff(K, OptionType::Call);
    auto greeks = mc_lr_greeks(base_config, payoff);

    // Digital delta should be positive (more likely ITM as spot rises)
    REQUIRE(greeks.delta > 0.0);
}

// ============================================================================
// Comparison: all three methods on the same option
// ============================================================================

TEST_CASE("All three methods agree on call delta", "[greeks]") {
    VanillaPayoff payoff(K, OptionType::Call);

    auto fd = mc_fd_greeks(base_config, payoff);
    auto ipa = mc_ipa_greeks(base_config, K, OptionType::Call);
    VanillaPayoff payoff2(K, OptionType::Call);
    auto lr = mc_lr_greeks(base_config, payoff2);

    Real ref = an_delta();

    // All should be within 0.05 of analytical
    REQUIRE_THAT(fd.delta, WithinAbs(ref, 0.05));
    REQUIRE_THAT(ipa.delta, WithinAbs(ref, 0.05));
    REQUIRE_THAT(lr.delta, WithinAbs(ref, 0.05));
}

TEST_CASE("Put delta is negative via all methods", "[greeks]") {
    VanillaPayoff payoff(K, OptionType::Put);

    auto fd = mc_fd_greeks(base_config, payoff);
    auto ipa = mc_ipa_greeks(base_config, K, OptionType::Put);
    auto lr = mc_lr_greeks(base_config, payoff);

    REQUIRE(fd.delta < 0.0);
    REQUIRE(ipa.delta < 0.0);
    REQUIRE(lr.delta < 0.0);
}
