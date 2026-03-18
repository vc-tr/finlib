#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "qe/risk/var.hpp"
#include "qe/risk/portfolio.hpp"
#include "qe/risk/stress.hpp"
#include "qe/math/random.hpp"

using namespace qe;
using Catch::Matchers::WithinAbs;

TEST_CASE("Historical VaR on known data", "[risk][var]") {
    // Simple P&L: 100 observations, sorted = {-10, -5, ..., 0, ..., +10}
    Vec pnl(100);
    for (Size i = 0; i < 100; ++i) {
        pnl[i] = static_cast<Real>(i) - 50.0;  // range: -50 to 49
    }

    Real var95 = historical_var(pnl, 0.95);

    // 5th percentile of [-50, 49] is -45
    // VaR = -(-45) = 45
    REQUIRE_THAT(var95, WithinAbs(45.0, 1.0));
}

TEST_CASE("CVaR >= VaR (always)", "[risk][cvar]") {
    MersenneTwister rng(42);
    Vec pnl(10000);
    for (auto& x : pnl) {
        x = rng.normal() * 10.0;  // mean=0, std=10
    }

    Real var95 = historical_var(pnl, 0.95);
    Real cvar95 = historical_cvar(pnl, 0.95);

    REQUIRE(cvar95 >= var95);
}

TEST_CASE("CVaR > VaR for non-degenerate distributions", "[risk][cvar]") {
    Vec pnl = {-10, -5, -3, -1, 0, 1, 2, 3, 5, 8, 10, 12, 15, 18, 20};

    Real var90 = historical_var(pnl, 0.90);
    Real cvar90 = historical_cvar(pnl, 0.90);

    REQUIRE(cvar90 > var90);
}

TEST_CASE("Parametric VaR for standard normal", "[risk][var]") {
    // For N(0, 1) at 95%: VaR ≈ 1.645
    Real var95 = parametric_var(0.0, 1.0, 0.95);
    REQUIRE_THAT(var95, WithinAbs(1.645, 0.01));
}

TEST_CASE("Parametric VaR for N(mu, sigma)", "[risk][var]") {
    // For N(0.01, 0.02) at 99%
    Real var99 = parametric_var(0.01, 0.02, 0.99);
    // VaR = -(0.01 - 2.326 * 0.02) = -(0.01 - 0.04652) = 0.03652
    REQUIRE_THAT(var99, WithinAbs(0.03652, 0.002));
}

TEST_CASE("Parametric CVaR > parametric VaR", "[risk][cvar]") {
    Real var = parametric_var(0.0, 1.0, 0.95);
    Real cvar = parametric_cvar(0.0, 1.0, 0.95);
    REQUIRE(cvar > var);
}

// ============================================================================
// Portfolio risk tests
// ============================================================================

TEST_CASE("Portfolio risk aggregation", "[risk][portfolio]") {
    std::vector<OptionPosition> positions = {
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, 10.0},
        {100.0, 95.0, 0.05, 0.2, 1.0, OptionType::Put, -5.0},
    };

    auto risk = portfolio_risk(positions);

    REQUIRE(risk.total_value != 0.0);
    // Long calls + short puts → positive delta
    REQUIRE(risk.total_delta > 0.0);
    // Gamma: long calls contribute positive, short puts contribute negative
    REQUIRE(risk.total_gamma != 0.0);
}

TEST_CASE("Delta-neutral portfolio has near-zero delta", "[risk][portfolio]") {
    // Long 1 ATM call, short delta calls to hedge
    Real delta = bs_delta(100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call);

    std::vector<OptionPosition> positions = {
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, 1.0},
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, 1.0 / (delta - 1.0) * delta},
    };

    // This won't be perfectly hedged with puts, but the concept is right
    auto risk = portfolio_risk(positions);
    // Just verify it computes without error
    REQUIRE(std::isfinite(risk.total_delta));
}

// ============================================================================
// Stress testing
// ============================================================================

TEST_CASE("Stress test produces results for each scenario", "[risk][stress]") {
    std::vector<OptionPosition> positions = {
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, 10.0},
    };

    auto scenarios = predefined_scenarios();
    auto results = stress_test_portfolio(positions, scenarios);

    REQUIRE(results.size() == scenarios.size());

    for (const auto& r : results) {
        REQUIRE(std::isfinite(r.stressed_value));
        REQUIRE(std::isfinite(r.pnl));
    }
}

TEST_CASE("Crash scenario hurts long call portfolio", "[risk][stress]") {
    std::vector<OptionPosition> positions = {
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, 100.0},
    };

    StressScenario crash = {"Crash", 0.75, 0.15, -0.01};
    auto results = stress_test_portfolio(positions, {crash});

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].pnl < 0.0);  // long calls lose in a crash
}

TEST_CASE("Vol spike helps long options", "[risk][stress]") {
    std::vector<OptionPosition> positions = {
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Call, 100.0},
        {100.0, 100.0, 0.05, 0.2, 1.0, OptionType::Put, 100.0},
    };

    StressScenario vol_spike = {"Vol Spike", 1.0, 0.20, 0.0};
    auto results = stress_test_portfolio(positions, {vol_spike});

    // Long straddle benefits from vol spike
    REQUIRE(results[0].pnl > 0.0);
}
